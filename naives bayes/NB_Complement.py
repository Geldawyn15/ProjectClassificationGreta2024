import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.naive_bayes import ComplementNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Chargement des données
data = pd.read_csv('card_credit_fraud.csv')

# Encodage de la colonne 'type'
encoder = OneHotEncoder(handle_unknown='ignore')
type_encoded = encoder.fit_transform(data[['type']])
type_encoded_df = pd.DataFrame(type_encoded.toarray(), columns=encoder.get_feature_names_out())
data_encoded = pd.concat([data.drop(columns=['type']), type_encoded_df], axis=1)

# Sélectionner les colonnes à inclure dans les caractéristiques
features = ['step', 'amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest']

# Séparation des données en fonction de la variable cible
X = data_encoded[features]  # Caractéristiques
y = data_encoded['isFraud']  # Variable cible

# Normalisation des caractéristiques avec MinMaxScaler
scaler = MinMaxScaler()
X_normalized = scaler.fit_transform(X)

# Séparation des données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X_normalized, y, test_size=0.3, random_state=42)

# Entraînement du modèle ComplementNB
complement_nb_model = ComplementNB()
complement_nb_model.fit(X_train, y_train)

# Évaluation du modèle
y_pred = complement_nb_model.predict(X_test)

# Calcul de l'exactitude (accuracy)
accuracy = accuracy_score(y_test, y_pred)
print("Exactitude (Accuracy) : {:.2f}%".format(accuracy * 100))

# Matrice de confusion
conf_matrix = confusion_matrix(y_test, y_pred)
print("Matrice de confusion :")
print(conf_matrix)

# Rapport de classification
class_report = classification_report(y_test, y_pred)
print("Rapport de classification :")
print(class_report)
