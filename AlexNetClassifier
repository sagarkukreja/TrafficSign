myTrainingFolder = 'C:\Users\sk3126\Downloads\BelgiumTSC_Train\BelgiumTSC_Train\Training';
testingFolder = 'C:\Users\sk3126\Downloads\Testing\Testing';


trainingSet = imageDatastore(myTrainingFolder,'IncludeSubfolders', true,  'LabelSource', 'foldernames');
testingSet = imageDatastore(testingFolder,'IncludeSubfolders', true,  'LabelSource', 'foldernames');
net = alexnet;
layersTransfer = net.Layers(1:end-3);
numClasses = numel(categories(trainingSet.Labels));
layers = [...
    layersTransfer
    fullyConnectedLayer(numClasses,'WeightLearnRateFactor',20,'BiasLearnRateFactor',20)
    softmaxLayer
    classificationLayer];

sz = net.Layers(1).InputSize
numImagesTraining = numel(trainingSet.Files)
numImagesTesting = numel(testingSet.Files)


for j = 1 : numImagesTraining
    img = imread(trainingSet.Files{j});
    img = imresize(img, [227 227]);
    imwrite(img,trainingSet.Files{j});
%     img = rgb2gray(img);
%     img = imbinarize(img);
%     trainingFeatures(j,:) = extractHOGFeatures(img,'CellSize',cellSize);
end

for j = 1 : numImagesTesting
    img = imread(testingSet.Files{j});
    img = imresize(img, [227 227]);
    imwrite(img,testingSet.Files{j});
%     img = rgb2gray(img);
%     img = imbinarize(img);
%     trainingFeatures(j,:) = extractHOGFeatures(img,'CellSize',cellSize);
end

options = trainingOptions('sgdm',...
    'MiniBatchSize',5,...
    'MaxEpochs',10,...
    'InitialLearnRate',0.0001);
testLabels = testingSet.Labels;
netTransfer = trainNetwork(trainingSet,layers,options);
predictedLabels = classify(netTransfer,testingSet);
[confMat,order] = confusionmat(testLabels, predictedLabels);
confMatPercent = bsxfun(@rdivide,confMat,sum(confMat,2))
