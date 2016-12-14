from __future__ import print_function
import tensorflow as tf

import arff
import pprint
from pdb import set_trace as t
import numpy as np
from sklearn import linear_model, datasets, svm, mixture, preprocessing, metrics


# Load the feature data file
featuresFilePath = "../data/train_data_features_large.arff"
featuresData = arff.load(open(featuresFilePath, 'rb'))
testFeaturesFilePath = "../data/test_data_features_large.arff"
testFeaturesData = arff.load(open(testFeaturesFilePath, 'rb'))

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

scaler = preprocessing.MinMaxScaler().fit(xTrain)
xTrain = scaler.transform(xTrain)
xTest = scaler.transform(xTest)

yTrainNumeric = map(lambda x: emotionToClassTensor(x), yTrain)
yTestNumeric = map(lambda x: emotionToClassTensor(x), yTest)

xTrainNp = np.array(xTrain)
yTrainNp = np.array(yTrainNumeric)

# Parameters
learning_rate = 0.0001
training_epochs = 9000  
batch_size = 580
display_step = 1

# Network Parameters
n_hidden_1 = 512 # 1st layer number of features
n_hidden_2 = 512 # 2nd layer number of features
n_hidden_3 = 256 # 3rd layer
n_hidden_4 = 16
n_input = len(xTrain[0]) # data input
dropout = 0.7

n_classes = 5 # total classes (0-4 digits)

# tf Graph input
x = tf.placeholder("float", [None, n_input])
y = tf.placeholder("float", [None, n_classes])

np.random.seed(10)

def next_batch(batch_size):
     shuffle_indices = np.random.permutation(np.arange(len(xTrainNp)))
     x_shuffled = xTrainNp[shuffle_indices]
     y_shuffled = yTrainNp[shuffle_indices]
     return x_shuffled[:batch_size], y_shuffled[:batch_size]

# Create model
def multilayer_perceptron(x, weights, biases):
    # Hidden layer with RELU activation
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    layer_1 = tf.nn.relu(layer_1)
    #layer_1 = tf.nn.dropout(layer_1, dropout)

    # Hidden layer with RELU activation
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    layer_2 = tf.nn.relu(layer_2)
    #layer_2 = tf.nn.dropout(layer_2, dropout)

    # Hidden layer with RELU activation
    #layer_3 = tf.add(tf.matmul(layer_2, weights['h3']), biases['b3'])
    #layer_3 = tf.nn.relu(layer_3)

    # Hidden layer with RELI activation
    #layer_4 = tf.add(tf.matmul(layer_3, weights['h4']), biases['b4'])
    #layer_4 = tf.nn.relu(layer_4)

    # Output layer with linear activation
    out_layer = tf.matmul(layer_2, weights['out']) + biases['out']
    return out_layer

# Store layers weight & bias
weights = {
    'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
    'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    'h3': tf.Variable(tf.random_normal([n_hidden_2, n_hidden_3])),
    'h4': tf.Variable(tf.random_normal([n_hidden_3, n_hidden_4])),
    'out': tf.Variable(tf.random_normal([n_hidden_2, n_classes]))
}
biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'b3': tf.Variable(tf.random_normal([n_hidden_3])),
    'b4': tf.Variable(tf.random_normal([n_hidden_4])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}
 
# Construct model
pred = multilayer_perceptron(x, weights, biases)

# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y)
                      # Uncomment the following to apply l2_loss to layers
                      #+ 0.01*tf.nn.l2_loss(weights['h1'])
                      #+ 0.01*tf.nn.l2_loss(weights['h2'])
                      #+ 0.01*tf.nn.l2_loss(weights['out'])
                      #+ 0.01*tf.nn.l2_loss(biases['b1'])
                      #+ 0.01*tf.nn.l2_loss(biases['b2'])
                      #+ 0.01*tf.nn.l2_loss(biases['out'])
                      )
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Initializing the variables
init = tf.initialize_all_variables()

# Launch the graph
with tf.Session() as sess:
    sess.run(init)

    # Training cycle
    for epoch in range(training_epochs):
        avg_cost = 0.
        total_batch = int(len(xTrain)/batch_size)
        # Loop over all batches
        for i in range(total_batch):
            batch_x, batch_y = next_batch(batch_size)
            # Run optimization op (backprop) and cost op (to get loss value)
            _, c = sess.run([optimizer, cost], feed_dict={x: batch_x,
                                                          y: batch_y})
            # Compute average loss
            avg_cost += c / total_batch
        # Display logs per epoch step
        if epoch % display_step == 0:
            print("Epoch:", '%04d' % (epoch+1), "cost=", \
                "{:.9f}".format(avg_cost))
    print("Optimization Finished!")

    # Test model
    y_p = tf.argmax(pred, 1)
    correct_prediction = tf.equal(y_p, tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    
    print ("Traing Data Metrics:")
    val_accuracy, y_pred = sess.run([accuracy, y_p], feed_dict= {x: xTrain, y: yTrainNumeric})
    print ("validation accuracy: " , "{:.9f}".format(val_accuracy))
    y_true = np.argmax(yTrainNumeric,1)
    #print ("Y_true: ", y_true)
    #print ("Y_pred: ", y_pred)
    print ("Precision:", metrics.precision_score(y_true, y_pred, average=None))
    print ("Recall:", metrics.recall_score(y_true, y_pred, average=None))
    print ("f1_score:", metrics.f1_score(y_true, y_pred, average=None))
    print ("confusion_matrix")
    print (metrics.confusion_matrix(y_true, y_pred))

    #metrics
    print ("Test Data Metrics:")
    val_accuracy, y_pred = sess.run([accuracy, y_p], feed_dict= {x: xTest, y: yTestNumeric})
    print ("validation accuracy: " , "{:.9f}".format(val_accuracy))
    y_true = np.argmax(yTestNumeric,1)
    #print ("Y_true: ", y_true)
    #print ("Y_pred: ", y_pred)
    print ("Precision:", metrics.precision_score(y_true, y_pred, average=None))
    print ("Recall:", metrics.recall_score(y_true, y_pred, average=None))
    print ("f1_score:", metrics.f1_score(y_true, y_pred, average=None))
    print ("confusion_matrix")
    print (metrics.confusion_matrix(y_true, y_pred))
