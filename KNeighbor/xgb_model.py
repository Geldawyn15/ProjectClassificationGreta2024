import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from xgboost.sklearn import XGBClassifier
from sklearn.metrics import average_precision_score, classification_report, accuracy_score, confusion_matrix
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Load dataset
df = pd.read_csv("card_credit_fraud.csv")
df = df.rename(columns={'oldbalanceOrg':'oldBalanceOrig', 'newbalanceOrig':'newBalanceOrig', 'oldbalanceDest':'oldBalanceDest', 'newbalanceDest':'newBalanceDest'})

df['nameOrig'] = df['nameOrig'].str.replace('C', '0', regex=False)
df['nameOrig'] = df['nameOrig'].str.replace('M', '1', regex=False)

df['nameDest'] = df['nameDest'].str.replace('C', '0', regex=False)
df['nameDest'] = df['nameDest'].str.replace('M', '1', regex=False)

df['nameOrig'] = df['nameOrig'].astype('category').cat.codes
df['nameDest'] = df['nameDest'].astype('category').cat.codes

Y = df['isFraud']
X = df.drop(['isFraud'], axis=1)

X.loc[X.type == 'TRANSFER', 'type'] = 0
X.loc[X.type == 'CASH_OUT', 'type'] = 1
X.loc[X.type == 'PAYMENT', 'type'] = 2
X.loc[X.type == 'CASH_IN', 'type'] = 3
X.loc[X.type == 'DEBIT', 'type'] = 4

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
    'classifier__max_depth': [1, 3, 5, 7, 10],
    'classifier__learning_rate': [0.01, 0.1],
    'classifier__scale_pos_weight': [1, (Y == 0).sum() / (Y == 1).sum()]  # Adjust for imbalance
}

# Configure GridSearchCV
n_splits = 5
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

grid_search = GridSearchCV(pipeline, param_grid, cv=cv, scoring='roc_auc', verbose=2, n_jobs=-1)

# Fit GridSearchCV to the training data
grid_search.fit(trainX, trainY)

# Analyze the results
cv_results = pd.DataFrame(grid_search.cv_results_)
#print(cv_results[['param_classifier__n_estimators', 'param_classifier__max_depth', 'mean_train_score', 'mean_test_score']])

# Best parameters found by GridSearchCV
print("Best Parameters: ", grid_search.best_params_)


print("-----------------------------------------------------------------------------------g-",)
y_pred = grid_search.predict(testX)
# Performance metrics
print("Testing Data ",)
print('Precision = {}'.format(average_precision_score(testY, y_pred)))
print("Accuracy:", accuracy_score(testY, y_pred))
print("Classification Report:\n", classification_report(testY, y_pred))

# Confusion Matrix
cm = confusion_matrix(testY, y_pred)
print("Confusion Matrix:\n", cm)

import pickle
filename='model.pkl'
pickle.dump(grid_search, open(filename, 'wb'))