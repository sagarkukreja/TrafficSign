close all;
clear all;

myTrainingFolder = 'D:\RIT MS\computer vision\traffic sign project\BelgiumTSC_Train\stop\stopSignImages';
testingFolder = 'D:\RIT MS\computer vision\traffic sign project\BelgiumTSC_Train\stop\stopSignTest';


trainingSet = imageDatastore(myTrainingFolder,'IncludeSubfolders', true,  'LabelSource', 'foldernames');
testingSet = imageDatastore(testingFolder,'IncludeSubfolders', true,  'LabelSource', 'foldernames');

%display random training image with hog feature
imageSize = [180 193];
im = imread(trainingSet.Files{8});
figure;
imshow(im);
title('Breaker');
im = imresize(im,imageSize);
im = imbinarize(rgb2gray(im));

[feature, visualization] = extractHOGFeatures(im,'CellSize',[4,4]);
figure;
plot(visualization);
title({'CellSize = [4 4]'; ['Length = ' num2str(length(feature))]});

cellSize = [4,4];
hogFeatureSize = length(feature);


trainingSetLabels = countEachLabel(trainingSet)
numImagesTraining = numel(trainingSet.Files)
numImagesTesting = numel(testingSet.Files)
testingSetLabels = countEachLabel(testingSet)


trainingFeatures = zeros(numImagesTraining, hogFeatureSize);
testFeatures = zeros(numImagesTesting, hogFeatureSize);

for j = 1 : numImagesTraining
    img = imread(trainingSet.Files{j});
    img = imresize(img,imageSize);
    img = rgb2gray(img);
    img = imbinarize(img);
    trainingFeatures(j,:) = extractHOGFeatures(img,'CellSize',cellSize);
end
trainingLabels = trainingSet.Labels;
classifier = fitcecoc(trainingFeatures,trainingLabels)

for l = 1 : numImagesTesting
    img = imread(testingSet.Files{l});
    img = imresize(img,imageSize);
    img = rgb2gray(img);
    img = imbinarize(img);
    testFeatures(l,:) = extractHOGFeatures(img,'CellSize',cellSize);
end
testLabels = testingSet.Labels;
predictedLabels = predict(classifier, testFeatures);
[confMat,order] = confusionmat(testLabels, predictedLabels);
confMatPercent = bsxfun(@rdivide,confMat,sum(confMat,2));


