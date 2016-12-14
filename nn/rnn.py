"""Implementation of the simple single-layer LSTM for emotion recognition."""

from bregman.suite import *

from os import listdir
from os.path import isfile, join

import numpy as np
import tensorflow as tf
from tensorflow.python.ops import rnn, rnn_cell

from sklearn import metrics

corpusDir = "/home/gmakarevich/classes/cs221/LDC2002S28_Emotional-Prosody-Speech-and-Transcripts"
trainDir = corpusDir + "/train"
testDir = corpusDir + "/test"

emotions = ['anger', 'despair', 'happiness', 'neutral', 'sadness']

def ReadDataSet(corpusDir):
    data_set = []
    for emotion in emotions:
        dir = corpusDir + "/" + emotion
        files = [f for f in listdir(dir) if isfile(join(dir, f))]
        for file in files:
            ffile = dir + "/" + file
            print ("Processing: %s" % file)
            mfcc = MelFrequencyCepstrum(ffile, nfft=1024, wfft=512, nhop=256).MFCC
            chromagram = Chromagram(ffile, nfft=1024, wfft=512, nhop=256)
            power = chromagram.POWER
            chroma = chromagram.CHROMA
            full_data = np.concatenate((mfcc, chroma, [power]))
            data_set.append((emotion, file, full_data))
    return data_set

def FindMinimumSize(data_set):
    data_sizes = []
    for data_point in data_set:
        emotion, file, data = data_point
        data_sizes.append(len(data[0]))
    return min(data_sizes)

def CutDataSetToMinSize(data_set, min_size):
    unified_data_set = []
    for data_point in data_set:
        emotion, file, data = data_point
        n_data = []
        for v in data:
            v = v[:min_size]
            n_data.append(v)
        unified_data_set.append((emotion, file, n_data))
    return unified_data_set

def EmotionToClassTensor(emotion, emotions):
    return map(lambda x: (1 if x==emotion else 0), emotions)

def DataSetToLabelTensor(dataSet, emotions):
    labels = []
    for data_point in dataSet:
        emotion, file, data = data_point
        labels.append(EmotionToClassTensor(emotion, emotions))
    return labels


trainDataSet = ReadDataSet(trainDir)
trainLabels = DataSetToLabelTensor(trainDataSet, emotions)

testDataSet = ReadDataSet(testDir)
testLabels = DataSetToLabelTensor(testDataSet, emotions)

timeSteps = min(FindMinimumSize(trainDataSet), FindMinimumSize(testDataSet))
featuresSize = len(trainDataSet[0][2])
print ("Minimum time series size is %d" % timeSteps)
print ("Features size is %d" % featuresSize)

trainDataSet = CutDataSetToMinSize(trainDataSet, timeSteps)

testDataSet = CutDataSetToMinSize(testDataSet, timeSteps)

'''
To classify speeaches using a recurrent neural network, we consider every data as 
a sequence of frames.
'''
# Parameters
learning_rate = 0.001
training_iters = 30000
batch_size = 160
display_step = 10

# Network Parameters
n_input = featuresSize
n_steps = timeSteps # timesteps
n_hidden = 256 # hidden layer num of features
n_classes = 5

# tf Graph input
x = tf.placeholder("float", [None, n_input, n_steps])
y = tf.placeholder("float", [None, n_classes])

# Define weights
weights = {
    'out': tf.Variable(tf.random_normal([n_hidden, n_classes]))
}
biases = {
    'out': tf.Variable(tf.random_normal([n_classes]))
}

np.random.seed(10)

def NextBatch(dataSet, labels, batch_size):
     shuffle_indices =list(np.random.permutation(np.arange(len(dataSet))))
     x_shuffled = [dataSet[i] for i in shuffle_indices]
     y_shuffled = [labels[i] for i in shuffle_indices]
     return x_shuffled[:batch_size], y_shuffled[:batch_size]

def RNN(x, weights, biases):

    # Prepare data shape to match `rnn` function requirements
    # Current data input shape: (batch_size, n_input, n_steps)
    # Required shape: 'n_steps' tensors list of shape (batch_size, n_input)

    # Permuting batch_size and n_steps
    x = tf.transpose(x, [2, 0, 1])
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
y_p = tf.argmax(pred, 1)
correct_pred = tf.equal(y_p, tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initializing the variables
init = tf.initialize_all_variables()

# Launch the graph
with tf.Session() as sess:
    sess.run(init)
    step = 1
    # Keep training until reach max iterations
    while step * batch_size < training_iters:
        batch_x, batch_y = NextBatch(trainDataSet, trainLabels, batch_size)
        # Reshape data to get 28 seq of 28 elements
        batch_x_data = map(lambda x: x[2], batch_x)
        # Run optimization op (backprop)
        sess.run(optimizer, feed_dict={x: batch_x_data, y: batch_y})
        if step % display_step == 0:
            # Calculate batch accuracy
            acc = sess.run(accuracy, feed_dict={x: batch_x_data, y: batch_y})
            # Calculate batch loss
            loss = sess.run(cost, feed_dict={x: batch_x_data, y: batch_y})
            print("Iter " + str(step*batch_size) + ", Minibatch Loss= " + \
                  "{:.6f}".format(loss) + ", Training Accuracy= " + \
                  "{:.5f}".format(acc))
        step += 1
    print("Optimization Finished!")

    print ("Traing Data Metrics:")
    trainData = map(lambda x: x[2], trainDataSet)
    val_accuracy, y_pred = sess.run([accuracy, y_p], feed_dict= {x: trainData, y: trainLabels})
    print ("validation accuracy: " , "{:.9f}".format(val_accuracy))
    y_true = np.argmax(trainLabels,1)
    print ("Y_true: ", y_true)
    print ("Y_pred: ", y_pred)
    print ("Precision:", metrics.precision_score(y_true, y_pred, average=None))
    print ("Recall:", metrics.recall_score(y_true, y_pred, average=None))
    print ("f1_score:", metrics.f1_score(y_true, y_pred, average=None))
    print ("confusion_matrix")
    print (metrics.confusion_matrix(y_true, y_pred))

    #metrics
    testData = map(lambda x: x[2], testDataSet)
    print ("Test Data Metrics:")
    val_accuracy, y_pred = sess.run([accuracy, y_p], feed_dict= {x: testData, y: testLabels})
    print ("validation accuracy: " , "{:.9f}".format(val_accuracy))
    y_true = np.argmax(testLabels,1)
    print ("Y_true: ", y_true)
    print ("Y_pred: ", y_pred)
    print ("Precision:", metrics.precision_score(y_true, y_pred, average=None))
    print ("Recall:", metrics.recall_score(y_true, y_pred, average=None))
    print ("f1_score:", metrics.f1_score(y_true, y_pred, average=None))
    print ("confusion_matrix")
    print (metrics.confusion_matrix(y_true, y_pred))
