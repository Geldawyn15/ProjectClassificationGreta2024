import pandas as pd
from app import app, db
from app.models import Transaction, Client
import sqlalchemy as sa

csv_file_path = '../card_credit_fraud.csv'

def replace_letters(input_string):
    # Replace 'C' with '0' and 'M' with '1'
    return input_string.replace('C', '0').replace('M', '1')

def replace_type(input_string):
    types = {'TRANSFER': 0, 'CASH_OUT': 1, 'PAYMENT': 2, 'CASH_IN': 3, 'DEBIT': 4}
    return types.get(input_string, -1)  # Returns -1 if type is not found

def clear_database():
    try:
        # Deleting all transactions first to avoid foreign key constraint violations
        db.session.query(Transaction).delete()
        # Then, deleting all clients
        db.session.query(Client).delete()
        db.session.commit()
        print("Database cleared successfully.")
    except Exception as e:
        db.session.rollback()
        print(f"Error clearing database: {e}")

def load_100():
    n1 = 60
    n2 = 40
    df = pd.read_csv(csv_file_path)
    # Now filter after loading the df
    df_not_fraud = df[df['isFraud'] == 0]
    df_is_fraud = df[df['isFraud'] == 1]
    
    df_sampledNot = df_not_fraud.sample(n=n1, random_state=42)
    df_sampledis = df_is_fraud.sample(n=n2, random_state=42)
    
    df_sampled = pd.concat([df_sampledNot, df_sampledis])
    
    return df_sampled

def send_to_DB(df):
    for row in df.itertuples():
        try:
            print(f"Index: {row.transactionId}")
            nameOrig = replace_letters(row.nameOrig)
            nameDest = replace_letters(row.nameDest)

            trans_type = replace_type(row.type)

            co = Client(name=nameOrig, oldBalance=row.oldbalanceOrg, newBalance=row.newbalanceOrig)
            cd = Client(name=nameDest, oldBalance=row.oldbalanceDest, newBalance=row.newbalanceDest)
            
            db.session.add(co)
            db.session.add(cd)
            db.session.commit()
            
            queryOrig = sa.select(Client).filter_by(name=nameOrig)
            queryDest = sa.select(Client).filter_by(name=nameDest)
            
            orig = db.session.scalars(queryOrig).first()
            dest = db.session.scalars(queryDest).first()
            
            trans = Transaction(
                name=row.transactionId,
                step=row.step,
                type=trans_type,
                amount=row.amount,
                isFraud=row.isFraud,
                id_client_originaire=orig.id if orig else None,
                id_client_destinaire=dest.id if dest else None
            )

            
            db.session.add(trans)
            db.session.commit()
        except Exception as e:
            print(f"Error processing transaction {row.transactionId}: {e}")
            db.session.rollback()

def query_transactions():
    query = sa.select(Transaction)
    transquery = db.session.scalars(query).all()
    for u in transquery:
        print(u.id, u.name)

if __name__ == '__main__':
    with app.app_context():
        # Clear the database before starting
        clear_database()

        # Load 100 random lines from the csv
        df_sampled = load_100()

        # Load the lines into the database
        send_to_DB(df_sampled)

        # Query transactions
        query_transactions()