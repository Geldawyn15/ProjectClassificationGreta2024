import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import average_precision_score, classification_report, accuracy_score, confusion_matrix

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Assuming df is loaded as before
df = pd.read_csv('card_credit_fraud.csv')
df = df.rename(columns={'oldbalanceOrg':'oldBalanceOrig', 'newbalanceOrig':'newBalanceOrig', 'oldbalanceDest':'oldBalanceDest', 'newbalanceDest':'newBalanceDest'})

X = df.loc[(df.type == 'TRANSFER') | (df.type == 'CASH_OUT')]

Y = X['isFraud']
del X['isFraud']

X = X.drop(['nameOrig', 'nameDest'], axis=1)

X.loc[X.type == 'TRANSFER', 'type'] = 0
X.loc[X.type == 'CASH_OUT', 'type'] = 1
X.type = X.type.astype(int)

trainX, testX, trainY, testY = train_test_split(X, Y, test_size=0.3, random_state=42)

# Configure and train the KNN classifier
knn = KNeighborsClassifier(n_neighbors=5)  # You can adjust n_neighbors based on your dataset size and characteristics
knn.fit(trainX, trainY)  # Fit model to training data
y_pred = knn.predict(testX)

# Calculate and print the average precision score, accuracy, and classification report
print('Precision = {}'.format(average_precision_score(testY, y_pred)))
print("Accuracy:", accuracy_score(testY, y_pred))
print("Classification Report:\n", classification_report(testY, y_pred))

# Calculate and display the confusion matrix
cm = confusion_matrix(testY, y_pred)
print("Confusion Matrix:\n", cm)