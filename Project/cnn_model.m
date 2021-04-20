% load data
load('train_data')
load('trainLabels')
load('test_data')
load('testLabels')


trainLabels=categorical(trainLabels).';
testLabels=categorical(testLabels).';
numcategories=5;

%%%%%% Training model
[height, width, numChannels, ~] = size(train_data);

imageSize = [height width numChannels];

% Model architecture
inputLayer = imageInputLayer(imageSize);
% Convolutional layer parameters
filterSize1 = [5 5];
filterSize=[3 3];
numFilters1=65;
numFilters = 32;

middleLayers = [
    
convolution2dLayer(filterSize, numFilters1, 'Padding', 2)
batchNormalizationLayer
leakyReluLayer()
maxPooling2dLayer(2, 'Stride', 2)

convolution2dLayer(filterSize, numFilters, 'Padding', 2);
batchNormalizationLayer
reluLayer()
maxPooling2dLayer(2, 'Stride',2)

convolution2dLayer(filterSize, 2*numFilters, 'Padding', 2);
batchNormalizationLayer
reluLayer()
maxPooling2dLayer(2, 'Stride',2)

convolution2dLayer(filterSize, numFilters, 'Padding', 2);
batchNormalizationLayer
reluLayer()
maxPooling2dLayer(2, 'Stride',2)

convolution2dLayer(filterSize, numFilters, 'Padding', 2);
batchNormalizationLayer
reluLayer()
maxPooling2dLayer(2, 'Stride',2)
]

finalLayers = [
fullyConnectedLayer(64)
reluLayer
fullyConnectedLayer(numcategories);
softmaxLayer
classificationLayer
]

layers = [
    inputLayer
    middleLayers
    finalLayers
    ]

% hyper-parameters
opts = trainingOptions('rmsprop', ...
    'InitialLearnRate',1e-3, ...
    'MaxEpochs',20, ...
    'Shuffle','every-epoch', ...
    'Verbose',true, ...
    'MiniBatchSize', 100, ...
    'LearnRateSchedule', 'piecewise', ...
    'LearnRateDropFactor', .3, ...
    'LearnRateDropPeriod', 10, ...
    'L2Regularization', 0.001, ...
    'Plots','training-progress');


doTraining = true;

if doTraining
    % Train a network.
    clear rng
    rng('default')
    Net = trainNetwork(train_data,trainLabels, layers, opts);
    save baseIRNetwork Net
else

end


%%%%%%% Run the network on the test set.
[YTest,scores] = classify(Net, test_data);


% Calculate the accuracy.
accuracy = sum(YTest == testLabels)/numel(testLabels)
[M,order]=confusionmat(testLabels,YTest)

% plot confusion matrix
cm = confusionchart(testLabels,YTest)
cm.NormalizedValues
cm.Title = 'Confusion Matrix of CNN Classifier';
cm.RowSummary = 'row-normalized';
% cm.ColumnSummary = 'column-normalized';

stat = confusionmatStats(testLabels,YTest)

avg_fscore = mean(stat.Fscore)
avg_recall = mean(stat.recall)
avg_precision = mean(stat.precision)