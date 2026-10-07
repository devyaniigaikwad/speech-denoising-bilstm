%% ================================================================
% TEST.M
% Speech Background Noise Suppression using BiLSTM v7
%
% This script:
%   1. Loads the trained v7 network
%   2. Loads the exact saved feature mean/std
%   3. Selects any WAV file
%   4. Converts to mono
%   5. Resamples to 16 kHz
%   6. Computes STFT
%   7. Normalizes using saved v7 statistics
%   8. Predicts the speech mask
%   9. Reconstructs the enhanced speech
%  10. Saves and plays the denoised audio
%
% NO TRAINING DATA IS READ.
% NO NORMALIZATION STATISTICS ARE RECALCULATED.
%
% ================================================================

clear;
clc;
close all;

fprintf('\n');
fprintf('============================================================\n');
fprintf('       SPEECH DENOISING - BiLSTM v7 FINAL TESTER\n');
fprintf('============================================================\n');


%% ================================================================
% 1. MODEL PATH
% ================================================================

modelFile = ...
    'C:\Users\Rajeev\OneDrive\Documents\MATLAB\data\output_v7\speechDenoisingBiLSTM_v7.mat';

outputFolder = ...
    'C:\Users\Rajeev\OneDrive\Documents\MATLAB\data\output_v7';


%% ================================================================
% 2. CHECK PATHS
% ================================================================

if ~isfile(modelFile)

    error('Model file not found:\n%s',modelFile);

end

if ~isfolder(outputFolder)

    mkdir(outputFolder);

end


%% ================================================================
% 3. LOAD TRAINED MODEL
% ================================================================

fprintf('\nLoading trained model...\n');

modelData = load(modelFile);

fprintf('Variables stored in model file:\n');

disp(fieldnames(modelData));


%% ================================================================
% 4. GET NETWORK
% ================================================================

if isfield(modelData,'net')

    net = modelData.net;

else

    error('Trained network "net" was not found.');

end


fprintf('Network loaded successfully.\n');


%% ================================================================
% 5. LOAD EXACT v7 NORMALIZATION STATISTICS
% ================================================================

if isfield(modelData,'featMean')

    featMean = modelData.featMean;

else

    error('featMean was not found in the model file.');

end


if isfield(modelData,'featStd')

    featStd = modelData.featStd;

else

    error('featStd was not found in the model file.');

end


% Make sure they are column vectors

featMean = featMean(:);

featStd = featStd(:);


% Avoid division by zero

featStd(featStd < 1e-6) = 1;


fprintf('\nExact v7 normalization statistics loaded.\n');

fprintf('Mean range : [%.4f, %.4f]\n', ...
    min(featMean),max(featMean));

fprintf('Std range  : [%.4f, %.4f]\n', ...
    min(featStd),max(featStd));


%% ================================================================
% 6. AUDIO / STFT CONFIGURATION
% ================================================================

targetFs = 16000;

windowLength = 512;

overlapLength = 384;

hopLength = ...
    windowLength - overlapLength;

fftLength = 512;

numBins = ...
    fftLength/2 + 1;

sequenceLength = 100;

window = ...
    hann(windowLength,'periodic');


fprintf('\n');
fprintf('============================================================\n');
fprintf('MODEL / STFT CONFIGURATION\n');
fprintf('============================================================\n');

fprintf('Sampling rate : %d Hz\n',targetFs);

fprintf('Window        : %d samples\n',windowLength);

fprintf('Overlap       : %d samples\n',overlapLength);

fprintf('Hop           : %d samples\n',hopLength);

fprintf('FFT length    : %d\n',fftLength);

fprintf('Frequency bins: %d\n',numBins);

fprintf('Sequence      : %d frames\n',sequenceLength);


%% ================================================================
% 7. VERIFY FEATURE DIMENSION
% ================================================================

if length(featMean) ~= numBins

    error( ...
        'featMean has %d elements, expected %d.', ...
        length(featMean),numBins);

end


if length(featStd) ~= numBins

    error( ...
        'featStd has %d elements, expected %d.', ...
        length(featStd),numBins);

end


%% ================================================================
% 8. SELECT INPUT WAV FILE
% ================================================================

fprintf('\n');
fprintf('============================================================\n');
fprintf('SELECT NOISY AUDIO FILE\n');
fprintf('============================================================\n');

[inputFile,inputPath] = uigetfile( ...
    {'*.wav','WAV Audio Files (*.wav)'}, ...
    'Select noisy speech file');


if isequal(inputFile,0)

    fprintf('No file selected.\n');

    return;

end


inputFullPath = ...
    fullfile(inputPath,inputFile);


fprintf('\nSelected file:\n%s\n',inputFullPath);


%% ================================================================
% 9. LOAD AUDIO
% ================================================================

[noisyAudio,originalFs] = ...
    audioread(inputFullPath);


fprintf('\n');
fprintf('Original audio information:\n');

fprintf('Sampling rate : %d Hz\n',originalFs);

fprintf('Samples       : %d\n',length(noisyAudio));

fprintf('Channels      : %d\n',size(noisyAudio,2));

fprintf('Duration      : %.2f seconds\n', ...
    length(noisyAudio)/originalFs);


%% ================================================================
% 10. CONVERT TO MONO
% ================================================================

if size(noisyAudio,2) > 1

    noisyAudio = mean(noisyAudio,2);

    fprintf('Stereo audio converted to mono.\n');

end


noisyAudio = noisyAudio(:);

noisyAudio = double(noisyAudio);


%% ================================================================
% 11. RESAMPLE TO 16 kHz
% ================================================================

if originalFs ~= targetFs

    fprintf('\n');

    fprintf( ...
        'Resampling %d Hz -> %d Hz...\n', ...
        originalFs,targetFs);

    noisyAudio = ...
        resample(noisyAudio,targetFs,originalFs);

else

    fprintf('\nAudio is already sampled at 16 kHz.\n');

end


%% ================================================================
% 12. INPUT AUDIO DIAGNOSTICS
% ================================================================

inputPeak = ...
    max(abs(noisyAudio));


inputRMS = ...
    sqrt(mean(noisyAudio.^2));


fprintf('\nInput audio:\n');

fprintf('Peak : %.8f\n',inputPeak);

fprintf('RMS  : %.8f\n',inputRMS);


if inputPeak < 1e-8

    error('Input audio contains almost no signal.');

end


%% ================================================================
% 13. COMPUTE STFT
% ================================================================

fprintf('\n');
fprintf('============================================================\n');
fprintf('COMPUTING STFT\n');
fprintf('============================================================\n');


[S,~,~] = spectrogram( ...
    noisyAudio, ...
    window, ...
    overlapLength, ...
    fftLength, ...
    targetFs);


fprintf('STFT size: %d x %d\n', ...
    size(S,1),size(S,2));


if size(S,1) ~= numBins

    error( ...
        'Unexpected STFT size. Expected %d rows, got %d.', ...
        numBins,size(S,1));

end


%% ================================================================
% 14. MAGNITUDE AND PHASE
% ================================================================

magnitude = ...
    abs(S);


phase = ...
    angle(S);


%% ================================================================
% 15. LOG MAGNITUDE FEATURES
% ================================================================

features = ...
    log(magnitude + 1e-6);


%% ================================================================
% 16. NORMALIZE USING SAVED v7 STATISTICS
% ================================================================

features = ...
    (features - featMean) ./ featStd;


fprintf('Feature matrix: %d x %d\n', ...
    size(features,1),size(features,2));


%% ================================================================
% 17. RUN BiLSTM NETWORK
% ================================================================

fprintf('\n');
fprintf('============================================================\n');
fprintf('RUNNING BiLSTM NETWORK\n');
fprintf('============================================================\n');


numFrames = ...
    size(features,2);


predictedMask = ...
    zeros(numBins,numFrames,'single');


numChunks = ...
    ceil(numFrames/sequenceLength);


fprintf('Total frames : %d\n',numFrames);

fprintf('Total chunks : %d\n',numChunks);


for chunk = 1:numChunks

    startFrame = ...
        (chunk-1)*sequenceLength + 1;


    endFrame = ...
        min(chunk*sequenceLength,numFrames);


    currentFeatures = ...
        single(features(:,startFrame:endFrame));


    % Create dlarray:
    %
    % C = frequency bins
    % T = time frames

    dlX = ...
        dlarray(currentFeatures,'CT');


    % Network prediction

    dlY = ...
        predict(net,dlX);


    % Convert to MATLAB array

    currentMask = ...
        extractdata(dlY);


    % Remove singleton dimensions

    currentMask = ...
        squeeze(currentMask);


    % Ensure bins x frames orientation

    if size(currentMask,1) ~= numBins

        currentMask = ...
            currentMask';

    end


    % Keep mask within [0,1]

    currentMask = ...
        max(0,min(1,currentMask));


    % Store mask

    predictedMask(:,startFrame:endFrame) = ...
        currentMask;


    fprintf( ...
        'Chunk %d/%d | frames %d-%d\n', ...
        chunk,numChunks,startFrame,endFrame);

end


%% ================================================================
% 18. MASK DIAGNOSTICS
% ================================================================

fprintf('\n');
fprintf('============================================================\n');
fprintf('MASK INFORMATION\n');
fprintf('============================================================\n');

fprintf('Mask mean : %.4f\n', ...
    mean(predictedMask(:)));

fprintf('Mask min  : %.4f\n', ...
    min(predictedMask(:)));

fprintf('Mask max  : %.4f\n', ...
    max(predictedMask(:)));


%% ================================================================
% 19. APPLY MASK
% ================================================================

enhancedMagnitude = ...
    predictedMask .* magnitude;


fprintf('\nEnhanced magnitude calculated.\n');


%% ================================================================
% 20. RECONSTRUCT COMPLEX SPECTRUM
% ================================================================

enhancedSTFT = ...
    enhancedMagnitude .* exp(1j*phase);


%% ================================================================
% 21. CONVERT ONE-SIDED SPECTRUM TO FULL SPECTRUM
% ================================================================

fprintf('\nReconstructing full spectrum...\n');


positiveSpectrum = ...
    enhancedSTFT;


negativeSpectrum = ...
    conj(enhancedSTFT(end-1:-1:2,:));


enhancedSTFTFull = ...
    [positiveSpectrum; negativeSpectrum];


fprintf('Full STFT size: %d x %d\n', ...
    size(enhancedSTFTFull,1), ...
    size(enhancedSTFTFull,2));


if size(enhancedSTFTFull,1) ~= fftLength

    error( ...
        'Incorrect full-spectrum size.');

end


%% ================================================================
% 22. MANUAL INVERSE STFT
% ================================================================

fprintf('\n');
fprintf('============================================================\n');
fprintf('INVERSE STFT\n');
fprintf('============================================================\n');


numFrames = ...
    size(enhancedSTFTFull,2);


outputLength = ...
    (numFrames-1)*hopLength + windowLength;


enhancedAudio = ...
    zeros(outputLength,1,'double');


windowNormalization = ...
    zeros(outputLength,1,'double');


for frame = 1:numFrames

    % Current frequency-domain frame

    currentSpectrum = ...
        enhancedSTFTFull(:,frame);


    % IFFT

    timeFrame = ...
        real(ifft(currentSpectrum,fftLength));


    % Keep the window-length portion

    timeFrame = ...
        timeFrame(1:windowLength);


    % Apply synthesis window

    timeFrame = ...
        timeFrame .* double(window);


    % Output indices

    startIndex = ...
        (frame-1)*hopLength + 1;


    endIndex = ...
        startIndex + windowLength - 1;


    % Overlap-add

    enhancedAudio(startIndex:endIndex) = ...
        enhancedAudio(startIndex:endIndex) + ...
        timeFrame;


    % Window power normalization

    windowNormalization(startIndex:endIndex) = ...
        windowNormalization(startIndex:endIndex) + ...
        double(window).^2;

end


%% ================================================================
% 23. OVERLAP-ADD NORMALIZATION
% ================================================================

validSamples = ...
    windowNormalization > 1e-8;


enhancedAudio(validSamples) = ...
    enhancedAudio(validSamples) ./ ...
    windowNormalization(validSamples);


enhancedAudio(~validSamples) = 0;


%% ================================================================
% 24. MATCH ORIGINAL AUDIO LENGTH
% ================================================================

desiredLength = ...
    length(noisyAudio);


if length(enhancedAudio) >= desiredLength

    enhancedAudio = ...
        enhancedAudio(1:desiredLength);

else

    enhancedAudio(end+1:desiredLength) = 0;

end


%% ================================================================
% 25. REMOVE DC OFFSET
% ================================================================

enhancedAudio = ...
    enhancedAudio - mean(enhancedAudio);


%% ================================================================
% 26. OUTPUT DIAGNOSTICS
% ================================================================

outputPeak = ...
    max(abs(enhancedAudio));


outputRMS = ...
    sqrt(mean(enhancedAudio.^2));


fprintf('\n');
fprintf('============================================================\n');
fprintf('OUTPUT AUDIO DIAGNOSTICS\n');
fprintf('============================================================\n');

fprintf('Input Peak  : %.10f\n',inputPeak);

fprintf('Output Peak : %.10f\n',outputPeak);

fprintf('Input RMS   : %.10f\n',inputRMS);

fprintf('Output RMS  : %.10f\n',outputRMS);


if outputPeak < 1e-8

    error( ...
        'Denoised output is essentially zero.');

end


%% ================================================================
% 27. NORMALIZE ONLY IF CLIPPING WOULD OCCUR
% ================================================================

if outputPeak > 0.98

    enhancedAudio = ...
        enhancedAudio .* ...
        (0.98/outputPeak);

end


finalPeak = ...
    max(abs(enhancedAudio));


finalRMS = ...
    sqrt(mean(enhancedAudio.^2));


fprintf('\nFinal audio:\n');

fprintf('Peak : %.10f\n',finalPeak);

fprintf('RMS  : %.10f\n',finalRMS);


%% ================================================================
% 28. CREATE OUTPUT FILE NAME
% ================================================================

[~,baseName,~] = ...
    fileparts(inputFile);


outputFile = ...
    fullfile( ...
        outputFolder, ...
        [baseName '_denoised.wav']);


%% ================================================================
% 29. SAVE AUDIO
% ================================================================

audiowrite( ...
    outputFile, ...
    single(enhancedAudio), ...
    targetFs);


fprintf('\n');
fprintf('============================================================\n');
fprintf('DENOISED AUDIO SAVED\n');
fprintf('============================================================\n');

fprintf('%s\n',outputFile);


%% ================================================================
% 30. PLAY INPUT AUDIO
% ================================================================

fprintf('\nPlaying noisy input...\n');

soundsc( ...
    single(noisyAudio), ...
    targetFs);


pause( ...
    length(noisyAudio)/targetFs + 1);


%% ================================================================
% 31. PLAY DENOISED AUDIO
% ================================================================

fprintf('Playing denoised output...\n');

soundsc( ...
    single(enhancedAudio), ...
    targetFs);


%% ================================================================
% 32. WAVEFORM COMPARISON
% ================================================================

timeAxis = ...
    (0:length(noisyAudio)-1)/targetFs;


figure( ...
    'Name','Speech Denoising - Waveforms', ...
    'NumberTitle','off');


subplot(2,1,1);

plot(timeAxis,noisyAudio);

grid on;

xlabel('Time (s)');

ylabel('Amplitude');

title('Noisy Input');


subplot(2,1,2);

plot(timeAxis,enhancedAudio);

grid on;

xlabel('Time (s)');

ylabel('Amplitude');

title('BiLSTM Denoised Output');


%% ================================================================
% 33. SPECTROGRAM COMPARISON
% ================================================================

figure( ...
    'Name','Speech Denoising - Spectrograms', ...
    'NumberTitle','off');


subplot(2,1,1);

spectrogram( ...
    noisyAudio, ...
    window, ...
    overlapLength, ...
    fftLength, ...
    targetFs, ...
    'yaxis');

title('Noisy Input');


subplot(2,1,2);

spectrogram( ...
    enhancedAudio, ...
    window, ...
    overlapLength, ...
    fftLength, ...
    targetFs, ...
    'yaxis');

title('BiLSTM Denoised Output');


%% ================================================================
% 34. FINAL MESSAGE
% ================================================================

fprintf('\n');
fprintf('============================================================\n');
fprintf('TEST COMPLETED SUCCESSFULLY\n');
fprintf('============================================================\n');

fprintf('\nInput:\n%s\n',inputFullPath);

fprintf('\nOutput:\n%s\n',outputFile);

fprintf('\nOutput duration: %.2f seconds\n', ...
    length(enhancedAudio)/targetFs);

fprintf('\nNo training files were read.\n');

fprintf('No normalization statistics were recalculated.\n');

fprintf('Exact v7 featMean and featStd were used.\n');

fprintf('============================================================\n');