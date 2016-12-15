# speech_emotion_recognition
## Synopsis

The code here implements different approaches to classification of the emotions from the recorded speech utterances. The project was mostly inspired by the requirements of CS221 class at Stanford.

## Code structure

 - data/  
   The arff feature files extracted from the original dataset using openSmile package.

 - modeltrain/  
   The scripts used to automate feature extraction, training the models and classification of the test set from the original data. It is mostly copied from the corresponding dir from the openSmile packet.  

 - nn/  
   The implementation of the simple multilayer perceptron (tf_nn.py) and single-layer lstm rnn (rnn.py) used in the project to analyze the performance of the NN to classify the utterances into emotion categories in the project.

## Installation

The following packages should be pre-installed to use the code from this project:

- TensorFlow (https://www.tensorflow.org)
- NumPy (http://www.numpy.org/)
- Scikit-learn (http://scikit-learn.org/stable/)
- Arff (https://pypi.python.org/pypi/arff/0.9)
- Bregman Audio-Visual Information Toolbox (http://digitalmusics.dartmouth.edu/~mcasey/bregman/#download-bregman)
- OpenSmile (https://sourceforge.net/projects/opensmile/?source=directory)

## Expected dataset layout

The following is the expected layout of the original wave files dataset on the filesystem:

   rootdir
      train/
            emotion1/
                File1
                File2
                …
            emotion2/
                File1
                File2
                …
            ...
      test/
            emotion1/
                File1
                File2
                …
            emotion2/
                File1
                File2
                …
            ...

## Tests

- To train and test the multilayer perceptron nn classifier, run `cd nn && python tf_nn.py`
- To train and test the LSTM RNN classifier, run `cd nn && python rnn.py`
   Note: you will need to edit the rnn.py to provide the proper path to the original dataset.
