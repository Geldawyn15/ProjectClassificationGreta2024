import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from xgboost.sklearn import XGBClassifier
from sklearn.metrics import average_precision_score, classification_report, accuracy_score, confusion_matrix
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Load dataset
df = pd.read_csv('card_credit_fraud.csv')
df = df.rename(columns={'oldbalanceOrg':'oldBalanceOrig', 'newbalanceOrig':'newBalanceOrig', 'oldbalanceDest':'oldBalanceDest', 'newbalanceDest':'newBalanceDest'})

X = df.loc[(df.type == 'TRANSFER') | (df.type == 'CASH_OUT')]

Y = X['isFraud']
X = X.drop(['isFraud', 'nameOrig', 'nameDest'], axis=1)

X.loc[X.type == 'TRANSFER', 'type'] = 0
X.loc[X.type == 'CASH_OUT', 'type'] = 1
X.type = X.type.astype(int)

# Split the dataset (Note: We split before resampling to avoid data leakage)
trainX, testX, trainY, testY = train_test_split(X, Y, test_size=0.3, random_state=42)

# Define the pipeline with SMOTE and XGBClassifier
pipeline = Pipeline([
    ('smote', SMOTE(random_state=42)),
    ('classifier', XGBClassifier(use_label_encoder=False, eval_metric='logloss'))
])

# Define the parameter grid
param_grid = {
    'classifier__n_estimators': [100, 200],
    'classifier__max_depth': [3, 5, 7],
    'classifier__learning_rate': [0.01, 0.1],
    'classifier__scale_pos_weight': [1, (Y == 0).sum() / (Y == 1).sum()]  # Adjust for imbalance
}

# Configure GridSearchCV
grid_search = GridSearchCV(pipeline, param_grid, cv=3, scoring='roc_auc', verbose=2, n_jobs=-1)

# Fit GridSearchCV to the training data
grid_search.fit(trainX, trainY)

# Best parameters found by GridSearchCV
print("Best Parameters: ", grid_search.best_params_)

# Predictions
y_pred = grid_search.predict(testX)

# Performance metrics
print('Precision = {}'.format(average_precision_score(testY, y_pred)))
print("Accuracy:", accuracy_score(testY, y_pred))
print("Classification Report:\n", classification_report(testY, y_pred))

# Confusion Matrix
cm = confusion_matrix(testY, y_pred)
print("Confusion Matrix:\n", cm)
