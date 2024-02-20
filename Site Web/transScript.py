import pandas as pd
from app import db
from app.models import Transaction, Client


csv_file_path = '../card_credit_fraud.csv'

def load_100():
    n = 100
    df = pd.read_csv(csv_file_path)
    df_sampled = df.sample(n=n, random_state=42)
    print(df_sampled.head())
    print(len(df_sampled))
    return df_sampled
    
def send_to_DB(df):
    #loop to send to database line by line
    for row in df.itertuples():
        print(f"Index: {row.transactionId}")
    
    
df_sampled = load_100()
send_to_DB(df_sampled)