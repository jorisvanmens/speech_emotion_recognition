from __future__ import print_function
import tensorflow as tf
from tensorflow.python.ops import rnn, rnn_cell

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

# Define baseline features
#baseLineFeatures = [u'pcm_intensity_sma_quartile1', u'pcm_intensity_sma_amean', u'pcm_intensity_sma_quartile3', u'pcm_intensity_sma_stddev',
#    u'pcm_loudness_sma_quartile1', u'pcm_loudness_sma_amean', u'pcm_loudness_sma_quartile3', u'pcm_loudness_sma_stddev', u'F0_sma_quartile1', 
#    u'F0_sma_amean', u'F0_sma_quartile3', u'F0_sma_stddev']
#baseLineFeatures = [u'pcm_intensity_sma_amean']
#baseLineFeatures = [u'pcm_LOGenergy_sma_amean', u'pcm_LOGenergy_sma_stddev', u'pcm_Mag_melspec_sma[0]_amean', u'pcm_Mag_melspec_sma[0]_stddev', 
# u'mfcc_sma[0]_amean', u'mfcc_sma[0]_stddev']

# Below features (selected after reviewing literature) provide great results
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
 
emotions = [
    'anger',
    'despair',
    'happiness',
    'neutral',
    'sadness',
]

def emotionToClassTensor(emotion):
    return map(lambda x: (1 if x==emotion else 0), emotions)

#print featuresData['attributes']

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

#scaler = preprocessing.StandardScaler().fit(xTrain)
#xTrain = scaler.transform(xTrain)
#xTest = scaler.transform(xTest)

yTrainNumeric = map(lambda x: emotionToClassTensor(x), yTrain)
yTestNumeric = map(lambda x: emotionToClassTensor(x), yTest)

xTrainNp = np.array(xTrain)
yTrainNp = np.array(yTrainNumeric)


# Parameters
learning_rate = 0.001
training_iters = 100000
batch_size = 128
display_step = 10

# Network Parameters
n_input = 28 # MNIST data input (img shape: 28*28)
n_steps = 28 # timesteps
n_hidden = 128 # hidden layer num of features
n_classes = 5 # MNIST total classes (0-9 digits)

# tf Graph input
x = tf.placeholder("float", [None, n_steps, n_input])
y = tf.placeholder("float", [None, n_classes])

# Define weights
weights = {
    'out': tf.Variable(tf.random_normal([n_hidden, n_classes]))
}
biases = {
    'out': tf.Variable(tf.random_normal([n_classes]))
}


def RNN(x, weights, biases):

    # Prepare data shape to match `rnn` function requirements
    # Current data input shape: (batch_size, n_steps, n_input)
    # Required shape: 'n_steps' tensors list of shape (batch_size, n_input)

    # Permuting batch_size and n_steps
    x = tf.transpose(x, [1, 0, 2])
    # Reshaping to (n_steps*batch_size, n_input)
    x = tf.reshape(x, [-1, n_input])
    # Split to get a list of 'n_steps' tensors of shape (batch_size, n_input)
    x = tf.split(0, n_steps, x)

    # Define a lstm cell with tensorflow
    lstm_cell = rnn_cell.BasicLSTMCell(n_hidden, forget_bias=1.0)

    # Get lstm cell output
    outputs, states = rnn.rnn(lstm_cell, x, dtype=tf.float32)

    # Linear activation, using rnn inner loop last output
    return tf.matmul(outputs[-1], weights['out']) + biases['out']

pred = RNN(x, weights, biases)

# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Evaluate model
correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initializing the variables
init = tf.initialize_all_variables()

# Launch the graph
with tf.Session() as sess:
    sess.run(init)
    step = 1
    # Keep training until reach max iterations
    while step * batch_size < training_iters:
        batch_x, batch_y = mnist.train.next_batch(batch_size)
        # Reshape data to get 28 seq of 28 elements
        batch_x = batch_x.reshape((batch_size, n_steps, n_input))
        # Run optimization op (backprop)
        sess.run(optimizer, feed_dict={x: batch_x, y: batch_y})
        if step % display_step == 0:
            # Calculate batch accuracy
            acc = sess.run(accuracy, feed_dict={x: batch_x, y: batch_y})
            # Calculate batch loss
            loss = sess.run(cost, feed_dict={x: batch_x, y: batch_y})
            print("Iter " + str(step*batch_size) + ", Minibatch Loss= " + \
                  "{:.6f}".format(loss) + ", Training Accuracy= " + \
                  "{:.5f}".format(acc))
        step += 1
    print("Optimization Finished!")

    # Calculate accuracy for 128 mnist test images
    test_len = 128
    test_data = mnist.test.images[:test_len].reshape((-1, n_steps, n_input))
    test_label = mnist.test.labels[:test_len]
    print("Testing Accuracy:", \
        sess.run(accuracy, feed_dict={x: test_data, y: test_label}))