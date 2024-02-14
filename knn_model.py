import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import average_precision_score, classification_report, accuracy_score

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

# Define the grid of hyperparameters to search
param_grid = {
    'n_neighbors': [3, 5, 7, 9, 11],
    'weights': ['uniform', 'distance'],
    'metric': ['euclidean', 'manhattan', 'minkowski']
}

# Setup the grid search
grid_search = GridSearchCV(KNeighborsClassifier(), param_grid, cv=5, n_jobs=-1, scoring='accuracy')

# Fit the grid search to the data
grid_search.fit(trainX, trainY)

best_params = grid_search.best_params_

# Print the best parameters and the best accuracy score
print("Best hyperparameters:", grid_search.best_params_)
print("Best cross-validation accuracy:", grid_search.best_score_)

# You can now evaluate the best model on the test set
best_model = grid_search.best_estimator_
y_pred_best = best_model.predict(testX)
print("Accuracy of the best model on the test set:", accuracy_score(testY, y_pred_best))


knn = KNeighborsClassifier(n_neighbors=best_params['n_neighbors'], weights=best_params['weights'], metric=best_params['metric'] )  # You can adjust n_neighbors based on your dataset size and characteristics
knn.fit(trainX, trainY)  # Fit model to training data
y_pred = knn.predict(testX)

# Calculate and print the average precision score, accuracy, and classification report
print('Precision = {}'.format(average_precision_score(testY, y_pred)))
print("Accuracy:", accuracy_score(testY, y_pred))
print("Classification Report:\n", classification_report(testY, y_pred))