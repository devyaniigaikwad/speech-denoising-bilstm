%% SPEECH NOISE SUPPRESSION v3  (complex mask, speaker-independent, streaming data)
% Implements these improvements over v2:
%  1. FULL training set (no 4000-file cap)
%  2. SPEAKER-INDEPENDENT split: validation speakers are never seen in training;
%     the official test set (unseen speakers + unseen noise) is used for testing
%  3. WIDER noise/SNR coverage: original noisy mix, original noise re-scaled,
%     noise bank from other utterances, babble (sum of speakers), coloured
%     noise, SNR -5..20 dB, random gain
%  4. ON-DEMAND LOADING: audio is read as small crops with
%     audioread(file,[start end]); only a small noise/speech bank stays in RAM
%  5. COMPLEX (magnitude + phase) MASK in polar form, with power-compressed
%     magnitude + complex spectral loss, and complex-compressed input features
%     (ALL 3*F input channels - log-magnitude, real, imag - are normalised)
%  6. FULL test-set evaluation: SNR, SI-SDR, STOI, per-input-SNR breakdown,
%     blind listening-test export, real-time latency measurement
%
% Needs : Deep Learning Toolbox, Signal Processing Toolbox (R2022b+ advised).
%         A GPU is strongly recommended. Copy the dataset to a LOCAL drive
%         (e.g. C:\vb_data) for fast random reads, not OneDrive.
% STATUS: UNVERIFIED. This file has NOT been executed in MATLAB by its author.
%         Treat it as a draft until you have run it. First run it with
%         cfg.smokeTest = true (tiny, fast), fix any errors, then do the full run.

clear; clc; close all; rng(0);

%% 0. CONFIG ---------------------------------------------------------------
cfg.root = fullfile(getenv("USERPROFILE"), "OneDrive", "Desktop", ...
                    "project_semv", "ss", "lstm_denoise", "data");
% cfg.root = "C:\Users\Rajeev\Downloads\Dataset";            % <- recommended: local copy of the dataset
if ~isfolder(cfg.root)
    cfg.root = uigetdir(pwd, "Select the folder containing clean_trainset_28spk_wav etc.");
end
cfg.trainClean = fullfile(cfg.root, "clean_trainset_28spk_wav");
cfg.trainNoisy = fullfile(cfg.root, "noisy_trainset_28spk_wav");
cfg.testClean  = fullfile(cfg.root, "clean_testset_wav");
cfg.testNoisy  = fullfile(cfg.root, "noisy_testset_wav");
cfg.outDir     = fullfile(fileparts(cfg.root), "output_v3");
if ~isfolder(cfg.outDir), mkdir(cfg.outDir); end

% Signal / STFT
cfg.fs = 16000; cfg.fftLen = 512; cfg.overlap = 256;
cfg.hop = cfg.fftLen - cfg.overlap;
cfg.win = hann(cfg.fftLen, "periodic");
cfg.numBins = cfg.fftLen/2 + 1;

% Data
cfg.valSpeakerList = ["p226" "p243" "p270"];  % FIXED validation speakers (reproducible)
cfg.seqLen       = 128;                 % frames per training crop (~2 s)
cfg.numVal       = 256;                 % validation crops
cfg.noiseBankN   = 600;                 % utterances used to build the noise bank
cfg.speechBankN  = 300;                 % utterances used to build the babble bank
cfg.snrRange     = [-5 20];             % dB
cfg.gainDb       = 6;
% probabilities: [original noisy, original noise rescaled, noise bank, babble, coloured]
cfg.noiseProbs   = [0.20 0.20 0.30 0.15 0.15];

% Model
cfg.hidden = 256; cfg.numTCN = 4;
% Largest value the mask magnitude can take. 1 = classic bounded mask.
% Experiment: try 1.5 or 2 to let the network amplify bins (a mask > 1 can
% partly undo over-suppression). Keep it identical for training and inference
% (it is saved inside the checkpoint's cfg).
cfg.maxMaskMag = 1;

% Training
cfg.epochs = 40; cfg.itersPerEpoch = 400; cfg.batchSize = 32;
cfg.lr = 1e-3; cfg.lrMin = 1e-5; cfg.warmupIters = 300; cfg.gradClip = 5;
cfg.patience = 8;
cfg.compress = 0.3;                     % power-law compression exponent
cfg.lambdaCplx = 0.3;                   % weight of complex loss vs magnitude loss

% Evaluation / latency
cfg.numTest = inf;                      % inf = complete test set
cfg.ctxFrames = 64;                     % context frames for latency benchmark

% Quick wiring check: set true for a tiny run that exercises every section
cfg.smokeTest = false;
if cfg.smokeTest
    cfg.epochs = 1; cfg.itersPerEpoch = 10; cfg.batchSize = 8; cfg.numVal = 32;
    cfg.numTest = 10; cfg.noiseBankN = 20; cfg.speechBankN = 10; cfg.warmupIters = 5;
end

useGPU = false; try, useGPU = canUseGPU; catch, end
fprintf("GPU in use: %d\n", useGPU);

%% 1. INDEX ALL FILES (cached) -------------------------------------------
idxTrain = fullfile(cfg.outDir, "index_train.mat");
idxTest  = fullfile(cfg.outDir, "index_test.mat");
if isfile(idxTrain), load(idxTrain, "Dall"); else
    Dall = indexDataset(cfg.trainClean, cfg.trainNoisy); save(idxTrain, "Dall"); end
if isfile(idxTest), load(idxTest, "Dtest"); else
    Dtest = indexDataset(cfg.testClean, cfg.testNoisy); save(idxTest, "Dtest"); end
fprintf("Indexed %d train files, %d test files\n", numel(Dall), numel(Dtest));

%% 2. SPEAKER-INDEPENDENT SPLIT -------------------------------------------
spkAll = string({Dall.speaker});
spk = sort(unique(spkAll));
valSpk = cfg.valSpeakerList;
if ~all(ismember(valSpk, spk))
    warning("Some speakers in cfg.valSpeakerList are not in this dataset.\nAvailable: %s\n" + ...
            "Falling back to a deterministic choice (every 10th sorted speaker).", strjoin(spk, ", "));
    valSpk = spk(3:10:end); valSpk = valSpk(1:min(3, numel(valSpk)));
end
assert(numel(valSpk) < numel(spk), "Validation speakers must not include every speaker.");
isVal = ismember(spkAll, valSpk);
Dtrain = Dall(~isVal); Dval = Dall(isVal);
testSpk = unique(string({Dtest.speaker}));
fprintf("Train speakers: %d | Val speakers: %s | Test speakers: %s\n", ...
    numel(unique(string({Dtrain.speaker}))), strjoin(valSpk, ","), strjoin(testSpk, ","));
assert(isempty(intersect(testSpk, unique(string({Dtrain.speaker})))), ...
    "Test speakers overlap with training speakers!");
fprintf("Train utterances: %d | Val utterances: %d\n", numel(Dtrain), numel(Dval));

%% 3. SMALL IN-MEMORY NOISE / SPEECH BANKS (training speakers only) -------
banks = buildBanks(Dtrain, cfg);
fprintf("Noise bank: %d clips | Babble bank: %d clips\n", numel(banks.noise), numel(banks.speech));

%% 4. NORMALISATION STATS + FIXED VALIDATION SET --------------------------
F = cfg.numBins;
% Per-channel mean/std over ALL 3*F input channels (log-mag, real, imag)
idStats = struct("mu", zeros(3*F,1,"single"), "sigma", ones(3*F,1,"single"));
m = zeros(3*F,1); q = zeros(3*F,1); nb = 6;
for b = 1:nb
    bt = makeBatch(Dtrain, randi(numel(Dtrain), 32, 1), banks, cfg, true, idStats);
    m = m + double(mean(bt.X, [2 3])) / nb;
    q = q + double(mean(bt.X.^2, [2 3])) / nb;
end
stats.mu = single(m); stats.sigma = single(sqrt(max(q - m.^2, 1e-6)));

rng(123);
valBatches = {};
for s = 1:cfg.batchSize:cfg.numVal
    valBatches{end+1} = makeBatch(Dval, randi(numel(Dval), cfg.batchSize, 1), ...
                                  banks, cfg, false, stats); %#ok<SAGROW>
end
rng(0);

%% 5. NETWORK --------------------------------------------------------------
net = buildNetwork(3*F, cfg.hidden, cfg.numTCN, 3*F);
fprintf("Model parameters: %.2f M\n", sum(cellfun(@numel, net.Learnables.Value))/1e6);

%% 6. CUSTOM TRAINING LOOP -------------------------------------------------
avgG = []; avgSqG = []; iter = 0; totalIters = cfg.epochs * cfg.itersPerEpoch;
bestVal = inf; bestNet = net; badEpochs = 0;
fig = figure("Name","Training"); ax = axes(fig); hold(ax,"on"); grid(ax,"on");
trainLine = animatedline(ax, "Color", [0.2 0.4 0.9]);
valLine = animatedline(ax, "Color", [0.9 0.2 0.2], "Marker", "o");
legend(ax, ["train", "validation (unseen speakers)"]); xlabel(ax,"iteration"); ylabel(ax,"loss");
tStart = tic;

for epoch = 1:cfg.epochs
    epochLoss = 0;
    for it = 1:cfg.itersPerEpoch
        iter = iter + 1;
        batch = makeBatch(Dtrain, randi(numel(Dtrain), cfg.batchSize, 1), banks, cfg, true, stats);
        [dlX, batch] = toDevice(batch, useGPU);
        [loss, grads] = dlfeval(@modelLoss, net, dlX, batch, cfg);
        grads = clipGradNorm(grads, cfg.gradClip);
        lr = scheduleLR(iter, totalIters, cfg);
        [net, avgG, avgSqG] = adamupdate(net, grads, avgG, avgSqG, iter, lr);
        lossVal = double(gather(extractdata(loss)));
        epochLoss = epochLoss + lossVal;
        addpoints(trainLine, iter, lossVal);
        if mod(it, 25) == 0, drawnow limitrate; end
    end

    vLoss = 0;
    for b = 1:numel(valBatches)
        [dlXv, vb] = toDevice(valBatches{b}, useGPU);
        out = stripdims(predict(net, dlXv));
        vLoss = vLoss + double(gather(extractdata(lossFromOutput(out, vb, cfg))));
    end
    vLoss = vLoss / numel(valBatches);
    addpoints(valLine, iter, vLoss); drawnow;
    fprintf("Epoch %2d/%d | train %.4f | val %.4f | lr %.2e | %.1f min\n", ...
        epoch, cfg.epochs, epochLoss/cfg.itersPerEpoch, vLoss, lr, toc(tStart)/60);

    if vLoss < bestVal - 1e-5
        bestVal = vLoss; bestNet = net; badEpochs = 0;
        netCPU = dlupdate(@gather, bestNet);
        save(fullfile(cfg.outDir, "best_checkpoint.mat"), "netCPU", "stats", "cfg");
    else
        badEpochs = badEpochs + 1;
        if badEpochs >= cfg.patience, fprintf("Early stopping.\n"); break, end
    end
end
net = dlupdate(@gather, bestNet);
save(fullfile(cfg.outDir, "denoiseNet_v3.mat"), "net", "stats", "cfg");

%% 7. FULL TEST-SET EVALUATION --------------------------------------------
% load(fullfile(cfg.outDir, "denoiseNet_v3.mat"));    % <- skip training next time
%   (this also restores the cfg/stats used in training, e.g. cfg.maxMaskMag)
rng(7);
Dt = Dtest(randperm(numel(Dtest)));
Dt = Dt(1:min(cfg.numTest, numel(Dt)));
N = numel(Dt);
methods = ["Noisy", "SpecSub", "Complex-TCN-LSTM"];
R.snr = zeros(N,3); R.sisdr = zeros(N,3); R.stoi = nan(N,3);
inSNR = zeros(N,1); rtf = zeros(N,1);
haveSTOI = exist("stoi", "file") > 0;

for k = 1:N
    [c, n] = loadPair(Dt(k).cleanPath, Dt(k).noisyPath, cfg.fs);
    ySS = spectralSubtraction(n, cfg);
    t0 = tic; yNN = denoiseSignal(net, n, stats, cfg); rtf(k) = toc(t0) / (numel(n)/cfg.fs);
    est = {n, ySS, yNN};
    inSNR(k) = snrdB(c, n);
    for j = 1:3
        R.snr(k,j) = snrdB(c, est{j});  R.sisdr(k,j) = siSDR(c, est{j});
        if haveSTOI
            try, R.stoi(k,j) = stoi(est{j}, c, cfg.fs);   % check arg order in your release's doc
            catch, haveSTOI = false; end
        end
    end
    if mod(k, 50) == 0, fprintf("  evaluated %d / %d\n", k, N); end
end

summary = table(methods', mean(R.snr)', mean(R.sisdr)', mean(R.stoi, "omitnan")', ...
    VariableNames=["Method","SNR_dB","SISDR_dB","STOI"]);
disp(summary);
fprintf("SI-SDR gain of model over noisy: %.2f dB | Real-time factor: %.3f\n", ...
    mean(R.sisdr(:,3) - R.sisdr(:,1)), mean(rtf));

% Breakdown by input SNR
edges = [-inf 5 10 15 inf]; lab = ["<5 dB", "5-10 dB", "10-15 dB", ">15 dB"];
bin = discretize(inSNR, edges);
fprintf("\nSI-SDR gain by input SNR (model vs noisy | specsub vs noisy):\n");
for b = 1:4
    s = bin == b;
    if any(s)
        fprintf("  %-9s n=%3d | %.2f dB | %.2f dB\n", lab(b), nnz(s), ...
            mean(R.sisdr(s,3) - R.sisdr(s,1)), mean(R.sisdr(s,2) - R.sisdr(s,1)));
    end
end

perFile = table(string({Dt.name})', string({Dt.speaker})', inSNR, R.snr, R.sisdr, R.stoi, ...
    VariableNames=["file","speaker","inputSNR","snr","sisdr","stoi"]);
writetable(summary, fullfile(cfg.outDir, "results_summary.csv"));
save(fullfile(cfg.outDir, "results_per_file.mat"), "perFile", "R", "inSNR");

figure("Name","SI-SDR"); boxchart(R.sisdr); xticklabels(methods); ylabel("SI-SDR (dB)"); grid on;
title("SI-SDR on the complete test set (unseen speakers + noises)");
figure("Name","Gain vs input SNR");
scatter(inSNR, R.sisdr(:,3) - R.sisdr(:,1), 12, "filled"); grid on;
xlabel("Input SNR (dB)"); ylabel("SI-SDR gain (dB)"); title("Model improvement vs input SNR");

%% 8. SPECTROGRAMS + LISTENING --------------------------------------------
[c, n] = loadPair(Dt(1).cleanPath, Dt(1).noisyPath, cfg.fs);
yNN = denoiseSignal(net, n, stats, cfg); ySS = spectralSubtraction(n, cfg);
sigs = {c, n, ySS, yNN}; ttl = ["Clean","Noisy","Spectral subtraction","Complex-TCN-LSTM"];
figure("Name","Spectrograms"); tiledlayout(4,1);
for j = 1:4
    nexttile; spectrogram(sigs{j}, cfg.win, cfg.overlap, cfg.fftLen, cfg.fs, "yaxis"); title(ttl(j));
end
for j = [2 3 4 1]
    fprintf("Playing: %s\n", ttl(j)); soundsc(sigs{j}, cfg.fs); pause(numel(c)/cfg.fs + 0.7);
end

%% 9. REAL-TIME LATENCY BENCHMARK -----------------------------------------
% IMPORTANT: this is NOT a full stateful streaming test. It only times ONE
% network call on a window of random data (ctxFrames frames). It does not:
%   - carry LSTM hidden/cell state from one frame to the next,
%   - run the TCN convolutions incrementally,
%   - do frame-by-frame STFT / overlap-add, or audio device I/O,
%   - measure the audio quality of a streaming implementation.
% Treat the numbers as a rough compute-cost estimate (a lower bound on real
% latency). A true real-time test needs a stateful frame-by-frame
% implementation (e.g. predict with network State, or code generation).
lat = measureLatency(net, cfg, cfg.ctxFrames, 200);
hopMs = 1000 * cfg.hop / cfg.fs; winMs = 1000 * cfg.fftLen / cfg.fs;
fprintf("\nPer-hop compute: median %.2f ms, 95th pct %.2f ms (hop budget = %.1f ms)\n", ...
    lat.median, lat.p95, hopMs);
fprintf("Algorithmic latency (one analysis window): %.1f ms | rough total: %.1f ms\n", ...
    winMs, winMs + lat.median);
disp("NOTE: single-window timing only - NOT a full stateful streaming test (see comment above).");
if lat.p95 < hopMs, disp("=> Compute cost per hop is below the hop budget (necessary, not sufficient, for real time).");
else, disp("=> Compute cost exceeds the hop budget: shrink cfg.hidden / numTCN or use a GPU / codegen."); end

%% 10. BLIND LISTENING TEST EXPORT ----------------------------------------
ltDir = fullfile(cfg.outDir, "listening_test");
if ~isfolder(ltDir), mkdir(ltDir); end
keyRows = {};
for k = 1:min(12, N)
    [c, n] = loadPair(Dt(k).cleanPath, Dt(k).noisyPath, cfg.fs);
    variants = {n, spectralSubtraction(n, cfg), denoiseSignal(net, n, stats, cfg)};
    order = randperm(3);
    for j = 1:3
        fname = sprintf("clip%02d_%s.wav", k, char('A' + j - 1));
        audiowrite(fullfile(ltDir, fname), safeScale(variants{order(j)}), cfg.fs);
        keyRows(end+1, :) = {fname, char(methods(order(j)))}; %#ok<SAGROW>
    end
end
writetable(cell2table(keyRows, VariableNames=["file","method"]), ...
           fullfile(ltDir, "KEY_do_not_open_until_scoring_done.csv"));

%% 11. DENOISE YOUR OWN FILE ----------------------------------------------
inFile = "my_noisy_mix.wav";
if isfile(inFile)
    [x, fx] = audioread(inFile); x = resample(mean(x, 2), cfg.fs, fx);
    y = denoiseSignal(net, x, stats, cfg);
    audiowrite(fullfile(cfg.outDir, "denoised_output.wav"), safeScale(y), cfg.fs); soundsc(y, cfg.fs);
end

%% ========================= LOCAL FUNCTIONS ================================
function S = computeSTFT(x, cfg)
    S = stft(x(:), cfg.fs, Window=cfg.win, OverlapLength=cfg.overlap, ...
             FFTLength=cfg.fftLen, FrequencyRange="onesided");
end

function y = reconstruct(S, L, cfg)
    y = istft(S, cfg.fs, Window=cfg.win, OverlapLength=cfg.overlap, ...
              FFTLength=cfg.fftLen, FrequencyRange="onesided");
    y = real(y(:));
    if numel(y) < L, y(end+1:L, 1) = 0; else, y = y(1:L); end
end

function [c, n] = loadPair(cleanPath, noisyPath, fs)
    [c, fc] = audioread(cleanPath); [n, fn] = audioread(noisyPath);
    c = resample(c(:,1), fs, fc); n = resample(n(:,1), fs, fn);
    L = min(numel(c), numel(n)); c = c(1:L); n = n(1:L);
end

function D = indexDataset(cleanDir, noisyDir)
    files = dir(fullfile(noisyDir, "*.wav"));
    assert(~isempty(files), "No wav files found in %s", noisyDir);
    D = struct("name", {}, "speaker", {}, "cleanPath", {}, "noisyPath", {}, "fs", {}, "n", {});
    for i = 1:numel(files)
        cp = fullfile(cleanDir, files(i).name); np = fullfile(noisyDir, files(i).name);
        try
            info = audioinfo(np);
            spk = regexp(files(i).name, '^[^_]+', 'match', 'once');   % e.g. p226 from p226_001.wav
            D(end+1) = struct("name", string(files(i).name), "speaker", string(spk), ...
                "cleanPath", string(cp), "noisyPath", string(np), ...
                "fs", info.SampleRate, "n", info.TotalSamples); %#ok<AGROW>
        catch ME
            warning("Skipping %s (%s)", files(i).name, ME.message);
        end
        if mod(i, 1000) == 0, fprintf("  indexed %d / %d\n", i, numel(files)); end
    end
end

function banks = buildBanks(D, cfg)
    banks.noise = {}; banks.speech = {};
    for i = randperm(numel(D), min(cfg.noiseBankN, numel(D)))
        try
            [c, n] = loadPair(D(i).cleanPath, D(i).noisyPath, cfg.fs);
            banks.noise{end+1} = single(n - c); %#ok<AGROW>
        catch, end
    end
    for i = randperm(numel(D), min(cfg.speechBankN, numel(D)))
        try
            [c, ~] = audioread(D(i).cleanPath);
            banks.speech{end+1} = single(resample(c(:,1), cfg.fs, D(i).fs)); %#ok<AGROW>
        catch, end
    end
end

function [c, n] = readCrop(rec, cropLen, randomStart, cfg)
    % Reads ONLY the needed samples from disk (on-demand loading)
    nRead = round(cropLen * rec.fs / cfg.fs);
    if rec.n <= nRead
        st = 1; en = rec.n;
    else
        if randomStart, st = randi(rec.n - nRead + 1);
        else, st = 1 + floor((rec.n - nRead) * 0.5); end
        en = st + nRead - 1;
    end
    c = audioread(rec.cleanPath, [st en]); n = audioread(rec.noisyPath, [st en]);
    c = resample(c(:,1), cfg.fs, rec.fs); n = resample(n(:,1), cfg.fs, rec.fs);
    L = min(numel(c), numel(n)); c = c(1:L); n = n(1:L);
    if L < cropLen, c(end+1:cropLen,1) = 0; n(end+1:cropLen,1) = 0;
    else, c = c(1:cropLen); n = n(1:cropLen); end
end

function seg = bankSegment(bank, L)
    clip = double(bank{randi(numel(bank))});
    if numel(clip) < L, clip = repmat(clip, ceil(L / numel(clip)) + 1, 1); end
    st = randi(numel(clip) - L + 1); seg = clip(st:st+L-1);
end

function nz = babbleNoise(bank, L)
    nz = zeros(L, 1);
    for k = 1:randi([3 6]), nz = nz + bankSegment(bank, L) * (0.5 + rand); end
end

function n = colouredNoise(L, alpha)
    W = fft(randn(L, 1)); half = floor(L/2) + 1;
    g = (1:half)'.^(-alpha/2);
    G = [g; flipud(g(2:L-half+1))];
    n = real(ifft(W .* G)); n = n / (std(n) + eps);
end

function [c, y] = mixAugment(c, n0, banks, cfg)
    L = numel(c); r = rand; e = cumsum(cfg.noiseProbs / sum(cfg.noiseProbs));
    rescale = true;
    if r < e(1),      nz = n0 - c; rescale = false;          % original mixture
    elseif r < e(2),  nz = n0 - c;                            % original noise, new SNR
    elseif r < e(3),  nz = bankSegment(banks.noise, L);       % other recorded noise
    elseif r < e(4),  nz = babbleNoise(banks.speech, L);      % babble
    else,             nz = colouredNoise(L, 2 * rand);        % white..brown
    end
    if rescale
        snrT = cfg.snrRange(1) + diff(cfg.snrRange) * rand;
        pc = mean(c.^2) + 1e-8; pn = mean(nz.^2) + 1e-10;
        nz = nz * sqrt(pc / (pn * 10^(snrT/10)));
    end
    g = 10^((2*rand - 1) * cfg.gainDb / 20);
    c = c * g; nz = nz * g; y = c + nz;
end

function batch = makeBatch(D, idx, banks, cfg, augment, stats)
    % Arrays are [freq x batch x time], matching a "CBT" dlarray.
    B = numel(idx); F = cfg.numBins; T = cfg.seqLen; cropLen = (T + 1) * cfg.hop;
    X = zeros(3*F, B, T, "single");
    Yr = zeros(F, B, T, "single"); Yi = Yr; Cr = Yr; Ci = Yr;
    for b = 1:B
        [c, n] = readCrop(D(idx(b)), cropLen, augment, cfg);
        if augment, [c, y] = mixAugment(c, n, banks_or_empty(banks), cfg); else, y = n; end
        Sc = computeSTFT(c, cfg); Sy = computeSTFT(y, cfg);
        Sc = Sc(:, 1:T); Sy = Sy(:, 1:T);
        mag = abs(Sy);
        comp = (mag + 1e-5) .^ (cfg.compress - 1);                 % complex-compressed inputs
        feat = ([log(mag + 1e-5); real(Sy) .* comp; imag(Sy) .* comp] - stats.mu) ./ stats.sigma;
        X(:, b, :)  = reshape(single(feat), 3*F, 1, T);
        Yr(:, b, :) = reshape(single(real(Sy)), F, 1, T);  Yi(:, b, :) = reshape(single(imag(Sy)), F, 1, T);
        Cr(:, b, :) = reshape(single(real(Sc)), F, 1, T);  Ci(:, b, :) = reshape(single(imag(Sc)), F, 1, T);
    end
    batch = struct("X", X, "Yr", Yr, "Yi", Yi, "Cr", Cr, "Ci", Ci);
end

function b = banks_or_empty(banks), b = banks; end

function [dlX, batch] = toDevice(batch, useGPU)
    if useGPU, batch = structfun(@gpuArray, batch, UniformOutput=false); end
    dlX = dlarray(batch.X, "CBT");
    batch = rmfield(batch, "X");
end

function net = buildNetwork(Fin, H, numTCN, Fout)
    lg = layerGraph(sequenceInputLayer(Fin, Name="in"));
    lg = addLayers(lg, [fullyConnectedLayer(H, Name="proj")
                        layerNormalizationLayer(Name="projLN")
                        reluLayer(Name="projRelu")]);
    lg = connectLayers(lg, "in", "proj");
    prev = 'projRelu';
    for k = 1:numTCN                                   % causal dilated residual blocks
        cn = sprintf('tcn%d_conv', k); ln = sprintf('tcn%d_ln', k);
        rl = sprintf('tcn%d_relu', k); ad = sprintf('tcn%d_add', k);
        lg = addLayers(lg, [convolution1dLayer(3, H, Padding="causal", DilationFactor=2^(k-1), Name=cn)
                            layerNormalizationLayer(Name=ln)
                            reluLayer(Name=rl)]);
        lg = addLayers(lg, additionLayer(2, Name=ad));
        lg = connectLayers(lg, prev, cn);
        lg = connectLayers(lg, rl, [ad '/in1']);
        lg = connectLayers(lg, prev, [ad '/in2']);
        prev = ad;
    end
    lg = addLayers(lg, [lstmLayer(H, OutputMode="sequence", Name="lstm1")
                        dropoutLayer(0.2, Name="drop1")
                        lstmLayer(H, OutputMode="sequence", Name="lstm2")]);
    lg = connectLayers(lg, prev, "lstm1");
    lg = addLayers(lg, additionLayer(2, Name="lstm_add"));
    lg = connectLayers(lg, "lstm2", "lstm_add/in1");
    lg = connectLayers(lg, prev, "lstm_add/in2");
    lg = addLayers(lg, [fullyConnectedLayer(H, Name="head1")
                        reluLayer(Name="headRelu")
                        fullyConnectedLayer(Fout, Name="head2")]);   % linear: [mag; p; q]
    lg = connectLayers(lg, "lstm_add", "head1");
    net = dlnetwork(lg);
end

function [Mr, Mi] = outputToMask(out, F, maxMag)
    % Polar complex mask: scaled-sigmoid magnitude (range 0..maxMag) and a
    % unit-norm phase direction (p,q). maxMag = 1 -> classic bounded mask.
    mag = maxMag ./ (1 + exp(-out(1:F, :, :)));
    p = out(F+1:2*F, :, :); q = out(2*F+1:3*F, :, :);
    nrm = sqrt(p.^2 + q.^2 + 1e-8);
    Mr = mag .* p ./ nrm; Mi = mag .* q ./ nrm;
end

function L = lossFromOutput(out, batch, cfg)
    F = cfg.numBins; c = cfg.compress;
    [Mr, Mi] = outputToMask(out, F, cfg.maxMaskMag);
    Er = Mr .* batch.Yr - Mi .* batch.Yi;
    Ei = Mr .* batch.Yi + Mi .* batch.Yr;
    magE = sqrt(Er.^2 + Ei.^2 + 1e-8); magC = sqrt(batch.Cr.^2 + batch.Ci.^2 + 1e-8);
    Lmag = mean((magE.^c - magC.^c).^2, "all");
    sE = magE.^(c - 1); sC = magC.^(c - 1);
    Lcx = mean((Er .* sE - batch.Cr .* sC).^2 + (Ei .* sE - batch.Ci .* sC).^2, "all");
    L = (1 - cfg.lambdaCplx) * Lmag + cfg.lambdaCplx * Lcx;
end

function [loss, grads] = modelLoss(net, dlX, batch, cfg)
    out = stripdims(forward(net, dlX));
    loss = lossFromOutput(out, batch, cfg);
    grads = dlgradient(loss, net.Learnables);
end

function grads = clipGradNorm(grads, thr)
    gn = 0;
    for i = 1:height(grads), gn = gn + sum(extractdata(grads.Value{i}).^2, "all"); end
    gn = sqrt(double(gather(gn)));
    if gn > thr
        s = thr / gn; grads.Value = cellfun(@(g) g * s, grads.Value, UniformOutput=false);
    end
end

function lr = scheduleLR(iter, total, cfg)
    if iter <= cfg.warmupIters, lr = cfg.lr * iter / cfg.warmupIters;
    else
        p = (iter - cfg.warmupIters) / max(1, total - cfg.warmupIters);
        lr = cfg.lrMin + 0.5 * (cfg.lr - cfg.lrMin) * (1 + cos(pi * p));
    end
end

function y = denoiseSignal(net, x, stats, cfg)
    x = x(:); F = cfg.numBins;
    S = computeSTFT(x, cfg); mag = abs(S);
    comp = (mag + 1e-5) .^ (cfg.compress - 1);
    feat = single(([log(mag + 1e-5); real(S) .* comp; imag(S) .* comp] - stats.mu) ./ stats.sigma);
    out = double(extractdata(predict(net, dlarray(feat, "CT"))));
    [Mr, Mi] = outputToMask(out, F, cfg.maxMaskMag);
    y = reconstruct((Mr + 1i * Mi) .* S, numel(x), cfg);
end

function lat = measureLatency(net, cfg, ctx, nRuns)
    x = dlarray(randn(3 * cfg.numBins, ctx, "single"), "CT");
    for i = 1:10, extractdata(predict(net, x)); end              % warm-up
    t = zeros(nRuns, 1);
    for i = 1:nRuns, t0 = tic; extractdata(predict(net, x)); t(i) = 1000 * toc(t0); end
    lat.median = median(t); lat.p95 = prctile(t, 95);
end

function y = spectralSubtraction(x, cfg)
    S = computeSTFT(x, cfg); mag2 = abs(S).^2;
    noisePSD = mean(mag2(:, 1:min(10, size(mag2, 2))), 2);
    clean2 = max(mag2 - 2.0 * noisePSD, 0.02 * mag2);
    y = reconstruct(S .* sqrt(clean2 ./ (mag2 + eps)), numel(x), cfg);
end

function v = snrdB(ref, x), v = 10*log10(sum(ref.^2) / (sum((ref - x).^2) + eps)); end

function v = siSDR(ref, est)
    ref = ref(:) - mean(ref); est = est(:) - mean(est);
    s = ((est' * ref) / (ref' * ref + eps)) * ref;
    v = 10*log10(sum(s.^2) / (sum((est - s).^2) + eps));
end

function y = safeScale(y), y = y / max(1, max(abs(y))); end