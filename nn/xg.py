"""Implementation of the simple emotion recognition classifier using XGBoost trees.
   We have not reported this approach in the project paper, but this experiment was useful
   at the initial stages of the project.
"""

import arff
import numpy as np
import pandas as pd
import xgboost

from sklearn import preprocessing
from sklearn import metrics

# Load the feature data file
featuresFilePath = "../data/train_data_features_large.arff"
featuresData = arff.load(open(featuresFilePath))
testFeaturesFilePath = "../data/test_data_features_large.arff"
testFeaturesData = arff.load(open(testFeaturesFilePath))

def Sanitize(col_name):
    return col_name.replace('[', '_').replace(']', '_')

columns = [Sanitize(n) for n, _ in featuresData['attributes']]
x_train = pd.DataFrame([tuple(r) for r in featuresData['data']], columns=columns)

y_encoder = preprocessing.LabelEncoder()
y_encoder.fit(x_train['emotion'])
y_train = y_encoder.transform(x_train['emotion'])

x_train = x_train.drop(['name', 'emotion'], axis=1)

model = xgboost.XGBClassifier()
model.fit(x_train, y_train)

x_test = pd.DataFrame([tuple(r) for r in testFeaturesData['data']], columns=columns)
y_test = y_encoder.transform(x_test['emotion'])
x_test = x_test.drop(['name', 'emotion'], axis=1)

print "Train Data Metrics:"
y_pred = model.predict(x_train)
print(metrics.classification_report(y_train, y_pred))


print "Test Data Metrics:"
y_pred = model.predict(x_test)
print(metrics.classification_report(y_test, y_pred))
