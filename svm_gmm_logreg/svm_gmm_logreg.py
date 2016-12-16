import arff
import pprint
from pdb import set_trace as t
import numpy as np
from sklearn import linear_model, datasets, svm, mixture, preprocessing


# Load the feature data file
featuresFilePath = "./data/train_data_features_large.arff"
featuresData = arff.load(open(featuresFilePath, 'rb'))
testFeaturesFilePath = "./data/test_data_features_large.arff"
testFeaturesData = arff.load(open(testFeaturesFilePath, 'rb'))

# Below features are used for the baseline method
#baseLineFeatures = [u'pcm_intensity_sma_quartile1', u'pcm_intensity_sma_amean', u'pcm_intensity_sma_quartile3', u'pcm_intensity_sma_stddev',
#    u'pcm_loudness_sma_quartile1', u'pcm_loudness_sma_amean', u'pcm_loudness_sma_quartile3', u'pcm_loudness_sma_stddev', u'F0_sma_quartile1', 
#    u'F0_sma_amean', u'F0_sma_quartile3', u'F0_sma_stddev']

# Below feature set (selected after reviewing literature -- see paper) provides the best results in SVM
features = [
    u'pcm_LOGenergy_sma_amean', u'pcm_LOGenergy_sma_quartile1', u'pcm_LOGenergy_sma_quartile3', u'pcm_LOGenergy_sma_stddev', 
    u'pcm_Mag_melspec_sma[0]_amean', u'pcm_Mag_melspec_sma[0]_quartile1', u'pcm_Mag_melspec_sma[0]_quartile3', u'pcm_Mag_melspec_sma[0]_stddev',
    u'mfcc_sma[0]_amean', u'mfcc_sma[0]_quartile1', u'mfcc_sma[0]_quartile3', u'mfcc_sma[0]_stddev',
    u'mfcc_sma[1]_amean', u'mfcc_sma[1]_quartile1', u'mfcc_sma[1]_quartile3', u'mfcc_sma[1]_stddev',
    u'mfcc_sma[2]_amean', u'mfcc_sma[2]_quartile1', u'mfcc_sma[2]_quartile3', u'mfcc_sma[2]_stddev',
    u'mfcc_sma[3]_amean', u'mfcc_sma[3]_quartile1', u'mfcc_sma[3]_quartile3', u'mfcc_sma[3]_stddev',
    u'mfcc_sma[4]_amean', u'mfcc_sma[4]_quartile1', u'mfcc_sma[4]_quartile3', u'mfcc_sma[4]_stddev',
    u'mfcc_sma[5]_amean', u'mfcc_sma[5]_quartile1', u'mfcc_sma[5]_quartile3', u'mfcc_sma[5]_stddev',
    u'mfcc_sma[6]_amean', u'mfcc_sma[6]_quartile1', u'mfcc_sma[6]_quartile3', u'mfcc_sma[6]_stddev',
    u'mfcc_sma[7]_amean', u'mfcc_sma[7]_quartile1', u'mfcc_sma[7]_quartile3', u'mfcc_sma[7]_stddev',
    u'mfcc_sma[8]_amean', u'mfcc_sma[8]_quartile1', u'mfcc_sma[8]_quartile3', u'mfcc_sma[8]_stddev',
    u'mfcc_sma[9]_amean', u'mfcc_sma[9]_quartile1', u'mfcc_sma[9]_quartile3', u'mfcc_sma[9]_stddev',
    u'mfcc_sma[10]_amean', u'mfcc_sma[10]_quartile1', u'mfcc_sma[10]_quartile3', u'mfcc_sma[10]_stddev',
    u'mfcc_sma[11]_amean', u'mfcc_sma[11]_quartile1', u'mfcc_sma[11]_quartile3', u'mfcc_sma[11]_stddev',
    u'pcm_LOGenergy_sma_de_amean', u'pcm_LOGenergy_sma_de_quartile1', u'pcm_LOGenergy_sma_de_quartile3', u'pcm_LOGenergy_sma_de_stddev', 
    u'pcm_Mag_melspec_sma_de[0]_amean', u'pcm_Mag_melspec_sma_de[0]_quartile1', u'pcm_Mag_melspec_sma_de[0]_quartile3', u'pcm_Mag_melspec_sma_de[0]_stddev',
    u'pcm_Mag_melspec_sma_de_de[0]_amean', u'pcm_Mag_melspec_sma_de_de[0]_quartile1', u'pcm_Mag_melspec_sma_de_de[0]_quartile3', u'pcm_Mag_melspec_sma_de_de[0]_stddev',
    u'mfcc_sma_de[0]_amean', u'mfcc_sma_de[0]_quartile1', u'mfcc_sma_de[0]_quartile3', u'mfcc_sma_de[0]_stddev'
    ]
 
# Top 40 features
# features = [
#     u'pcm_LOGenergy_sma_amean', u'pcm_LOGenergy_sma_stddev', u'pcm_Mag_melspec_sma[0]_amean', u'mfcc_sma[0]_amean', 
#     u'mfcc_sma[1]_amean', u'mfcc_sma[2]_quartile1', u'mfcc_sma[3]_amean', u'pcm_Mag_melspec_sma_de_de[0]_quartile1',
#     u'mfcc_sma[0]_quartile1', u'mfcc_sma[2]_amean',
#     u'mfcc_sma[6]_amean', u'mfcc_sma[1]_quartile1', u'mfcc_sma[8]_amean', u'pcm_Mag_melspec_sma[0]_quartile3',
#     u'mfcc_sma[10]_amean', u'mfcc_sma[10]_quartile3', u'mfcc_sma[0]_stddev', u'mfcc_sma[11]_amean',
#     u'mfcc_sma[3]_stddev', u'mfcc_sma[5]_quartile1',
#     u'mfcc_sma[1]_stddev', u'mfcc_sma[7]_quartile1', u'mfcc_sma[11]_quartile1', u'pcm_Mag_melspec_sma_de[0]_quartile1', 
#     u'pcm_LOGenergy_sma_de_quartile3', u'pcm_Mag_melspec_sma_de[0]_stddev', u'mfcc_sma[6]_quartile1', u'mfcc_sma[9]_quartile1', 
#     u'mfcc_sma[9]_quartile3', u'mfcc_sma[4]_quartile3',
#     u'pcm_Mag_melspec_sma[0]_stddev', u'mfcc_sma[4]_amean', u'mfcc_sma[5]_stddev', u'mfcc_sma[9]_amean', 
#     u'mfcc_sma_de[0]_stddev', u'mfcc_sma[8]_quartile3', u'mfcc_sma[7]_amean', u'mfcc_sma[10]_quartile1', 
#     u'mfcc_sma[0]_quartile3', u'pcm_Mag_melspec_sma_de[0]_quartile3'
#     ]

# Top 30 features
# features = [
#    u'pcm_LOGenergy_sma_amean', u'pcm_LOGenergy_sma_stddev', u'pcm_Mag_melspec_sma[0]_amean', u'mfcc_sma[0]_amean', 
#    u'mfcc_sma[1]_amean', u'mfcc_sma[2]_quartile1', u'mfcc_sma[3]_amean', u'pcm_Mag_melspec_sma_de_de[0]_quartile1',
#    u'mfcc_sma[0]_quartile1', u'mfcc_sma[2]_amean',
#    u'mfcc_sma[6]_amean', u'mfcc_sma[1]_quartile1', u'mfcc_sma[8]_amean', u'pcm_Mag_melspec_sma[0]_quartile3',
#    u'mfcc_sma[10]_amean', u'mfcc_sma[10]_quartile3', u'mfcc_sma[0]_stddev', u'mfcc_sma[11]_amean',
#    u'mfcc_sma[3]_stddev', u'mfcc_sma[5]_quartile1',
#    u'mfcc_sma[1]_stddev', u'mfcc_sma[7]_quartile1', u'mfcc_sma[11]_quartile1', u'pcm_Mag_melspec_sma_de[0]_quartile1', 
#    u'pcm_LOGenergy_sma_de_quartile3', u'pcm_Mag_melspec_sma_de[0]_stddev', u'mfcc_sma[6]_quartile1', u'mfcc_sma[9]_quartile1', 
#    u'mfcc_sma[9]_quartile3', u'mfcc_sma[4]_quartile3'
#    ]

# Top 20 features
# features = [
#    u'pcm_LOGenergy_sma_amean', u'pcm_LOGenergy_sma_stddev', u'pcm_Mag_melspec_sma[0]_amean', u'mfcc_sma[0]_amean', 
#    u'mfcc_sma[1]_amean', u'mfcc_sma[2]_quartile1', u'mfcc_sma[3]_amean', u'pcm_Mag_melspec_sma_de_de[0]_quartile1',
#    u'mfcc_sma[0]_quartile1', u'mfcc_sma[2]_amean',
#    u'mfcc_sma[6]_amean', u'mfcc_sma[1]_quartile1', u'mfcc_sma[8]_amean', u'pcm_Mag_melspec_sma[0]_quartile3',
#    u'mfcc_sma[10]_amean', u'mfcc_sma[10]_quartile3', u'mfcc_sma[0]_stddev', u'mfcc_sma[11]_amean',
#    u'mfcc_sma[3]_stddev', u'mfcc_sma[5]_quartile1'
#    ]

# The top 10 explanatory features
# features = [
#    u'pcm_LOGenergy_sma_amean', u'pcm_LOGenergy_sma_stddev', u'pcm_Mag_melspec_sma[0]_amean', u'mfcc_sma[0]_amean', 
#    u'mfcc_sma[1]_amean', u'mfcc_sma[2]_quartile1', u'mfcc_sma[3]_amean', u'pcm_Mag_melspec_sma_de_de[0]_quartile1',
#    u'mfcc_sma[0]_quartile1', u'mfcc_sma[2]_amean'
#    ]

# # The top 5 features
# features = [
#    u'mfcc_sma[0]_amean',
#    u'pcm_Mag_melspec_sma_de_de[0]_quartile1',
#    u'pcm_Mag_melspec_sma[0]_amean',
#    u'mfcc_sma[1]_amean',
#    u'mfcc_sma[3]_amean',
#    ]


# Get indices for certain features. You can get the list of all features through featuresData['attributes']
# The format of that is a list of tuples: (featureName, featureType), below I call these tuples "feature"
def getFeatureIndices(featureNames):
    #print featureNames
    #print featuresData['attributes']
    featureIndices = []
    for featureName in featureNames:
        #print "Name: " + featureName
        featureIndices.append([index for index, feature in enumerate(featuresData['attributes']) if feature[0] == featureName][0])
    return featureIndices

def createLimitedFeatureVector(featureValues, featureIndices, labelIndex):
    # Structure is featureValues[sample][feature]
    x = []
    y = []
    for currentFeatureValues in featureValues:
        x.append([currentFeatureValues[i] for i in featureIndices])
        y.append(currentFeatureValues[labelIndex])
    return x, y

labelIndex = getFeatureIndices([u"emotion"])[0]
featureIndices = getFeatureIndices(features)

xTrain, yTrain = createLimitedFeatureVector(featuresData['data'], featureIndices, labelIndex)
xTest, yTest = createLimitedFeatureVector(testFeaturesData['data'], featureIndices, labelIndex)
scaler = preprocessing.StandardScaler().fit(xTrain)
xTrainScaled = scaler.transform(xTrain)
xTestScaled = scaler.transform(xTest)


def predictResult(model, predictX, predictY):
    allPredictions = model.predict(predictX) #model.score_samples(predictX)
    right = 0
    wrong = 0
    for index, prediction in enumerate(allPredictions):
        print str(predictY[index]) + ", " + str(prediction) #+ " " + str(predictY[index] == prediction)
        if predictY[index] == prediction:
            right += 1
        else:
            wrong += 1
    print str(right) + " right out of " + str(right + wrong) + " total"
    print "Accuracy: " + str(float(right) / float(right+wrong))

#logReg = linear_model.LogisticRegression(C=1e5)
#logReg.fit(xTrain, yTrain)
#predictResult(logReg, xTrain, yTrain)
#predictResult(logReg, xTest, yTest)

def logRegMethod():
    logReg = linear_model.LogisticRegression(C=1e5)
    logReg.fit(xTrainScaled, yTrain)
    #predictResult(logReg, xTrainScaled, yTrain)
    predictResult(logReg, xTestScaled, yTest)
    #print sum(map(abs, logReg.coef_))

def SVMMethod():
    clf = svm.SVC() #gamma=0.001, C=100
    clf.fit(xTrainScaled, yTrain)
    #predictResult(clf, xTrainScaled, yTrain)
    predictResult(clf, xTestScaled, yTest)
    #print sum(map(abs, clf.coef_))

def GMMMethod():
    emotions = ["anger", "sadness", "happiness", "despair", "neutral"]
    clfAnger = mixture.GaussianMixture(n_components = 1)
    clfSadness = mixture.GaussianMixture(n_components = 1)
    clfHappiness = mixture.GaussianMixture(n_components = 1)
    clfDespair = mixture.GaussianMixture(n_components = 1)
    clfNeutral = mixture.GaussianMixture(n_components = 1)
    xAnger = []
    xSadness = []
    xHappiness = []
    xDespair = []
    xNeutral = []
    for index, label in enumerate(yTrain):
        if label == "anger":
            xAnger.append(xTrainScaled[index])
        if label == "sadness":
            xSadness.append(xTrainScaled[index])
        if label == "happiness":
            xHappiness.append(xTrainScaled[index])
        if label == "despair":
            xDespair.append(xTrainScaled[index])
        if label == "neutral":
            xNeutral.append(xTrainScaled[index])    
    clfAnger.fit(xAnger)
    clfSadness.fit(xSadness)
    clfHappiness.fit(xHappiness)
    clfDespair.fit(xDespair)
    clfNeutral.fit(xNeutral)

    scoreAnger = clfAnger.score_samples(xTestScaled) 
    scoreSadness = clfSadness.score_samples(xTestScaled) 
    scoreHappiness = clfHappiness.score_samples(xTestScaled) 
    scoreDespair = clfDespair.score_samples(xTestScaled) 
    scoreNeutral = clfNeutral.score_samples(xTestScaled) 

    right = 0
    wrong = 0
    for idx, val in enumerate(xTestScaled):
        scores = [scoreAnger[idx], scoreSadness[idx], scoreHappiness[idx], scoreDespair[idx], scoreNeutral[idx]]
        prediction = emotions[scores.index(max(scores))]
        if yTest[idx] == prediction:
            right += 1
        else:
            wrong += 1
        print str(yTest[idx]) + ", " + str(prediction)
    print str(right) + " right out of " + str(right + wrong) + " total"
    print "Accuracy: " + str(float(right) / float(right+wrong))


logRegMethod()
GMMMethod()
SVMMethod()
