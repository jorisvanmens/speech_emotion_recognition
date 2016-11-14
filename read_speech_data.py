import arff
import pprint
from pdb import set_trace as t

# Load the feature data file
featuresFilePath = "../train_data_features.arff"
featuresData = arff.load(open(featuresFilePath, 'rb'))

# Get indices for certain features. You can get the list of all features through featuresData['attributes']
# The format of that is a list of tuples: (featureName, featureType), below I call these tuples "feature"
loudnessIndex = [index for index, feature in enumerate(featuresData['attributes']) if feature[0] == u'pcm_loudness_sma_amean'][0]
labelIndex = [index for index, feature in enumerate(featuresData['attributes']) if feature[0] == u'emotion'][0]

# Get the mean loudness for Anger and Neutral samples
numAngerSamples = 0
numNeutralSamples = 0
angerLoudness = 0
neutralLoudness = 0
for sample in featuresData['data']:
    if sample[labelIndex] == "anger":
        numAngerSamples += 1
        angerLoudness += sample[loudnessIndex]
    elif sample[labelIndex] == "neutral":
        numNeutralSamples += 1
        neutralLoudness += sample[loudnessIndex]

angerMeanLoudness = float(angerLoudness) / float(numAngerSamples)
neutralMeanLoudness = float(neutralLoudness) / float(numNeutralSamples)

print "Mean anger loudness:\t" + str(angerMeanLoudness)
print "Mean neutral loudness:\t" + str(neutralMeanLoudness)
