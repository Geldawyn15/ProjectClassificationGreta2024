from urllib.parse import urlsplit
from flask import render_template, flash, redirect, url_for, request
from flask_login import login_user, logout_user, current_user, login_required
import sqlalchemy as sa
from sqlalchemy.sql.expression import func
import random
from app import app, db
from app.forms import LoginForm, RegistrationForm
from app.models import User, Transaction, Client
import numpy as np 
import pickle

model_path = '../model.pkl'
loaded_model = pickle.load(open(model_path, 'rb'))


@app.route('/')
@app.route('/selection')
@login_required
def selection():
    non_fraud_query = sa.select(Transaction).where(Transaction.isFraud == False).order_by(func.random()).limit(3)
    non_fraud_transactions = db.session.scalars(non_fraud_query).all()
    
    # Query 2 fraudulent transactions
    fraud_query = sa.select(Transaction).where(Transaction.isFraud == True).order_by(func.random()).limit(2)
    fraud_transactions = db.session.scalars(fraud_query).all()
    
    # Combine the transactions and shuffle them 
    transactions = non_fraud_transactions + fraud_transactions
    random.shuffle(transactions)
    
    types = {0: 'Transfer', 1: 'Cash Out', 2: 'Payment', 3: 'Cash In', 4: 'Debit'}
    
    return render_template('selection.html', title='Selections', transactions=transactions, types=types)

def ValuePredictor(transaction, orig, dest):
    transaction_list = [
        transaction.id, transaction.step, transaction.type, 
        transaction.amount, orig.id, orig.oldBalance, 
        orig.newBalance, dest.id, dest.oldBalance, dest.newBalance
    ]
    
    to_predict = np.array(transaction_list).reshape(1, -1)
    result = loaded_model.predict_proba(to_predict)
    return result

@app.route('/predict/<int:transaction_id>')
def predict(transaction_id):
    transaction = db.session.query(Transaction).filter_by(id=transaction_id).first()
    if not transaction:
        return "Transaction not found", 404
      
    query = sa.select(Transaction).filter_by(id=transaction_id)
    transaction = db.session.scalars(query).first()
    orig = db.session.query(Client).filter_by(id=transaction.id_client_originaire).first()
    dest = db.session.query(Client).filter_by(id=transaction.id_client_destinaire).first()
    if not orig or not dest:
        return "Client information not found", 404
    
  
    prediction = ValuePredictor(transaction, orig, dest)
    return render_template("predict.html", title='Prediction', transaction=transaction, prediction=prediction)

@app.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('selection'))
    form = LoginForm()
    if form.validate_on_submit():
        user = db.session.scalar(
            sa.select(User).where(User.username == form.username.data))
        if user is None or not user.check_password(form.password.data):
            flash('Invalid username or password')
            return redirect(url_for('login'))
        login_user(user, remember=form.remember_me.data)
        next_page = request.args.get('next')
        if not next_page or urlsplit(next_page).netloc != '':
            next_page = url_for('selection')
        return redirect(next_page)
    return render_template('login.html', title='Sign In', form=form)


@app.route('/logout')
def logout():
    logout_user()
    return redirect(url_for('selection'))


@app.route('/register', methods=['GET', 'POST'])
def register():
    if current_user.is_authenticated:
        return redirect(url_for('selection'))
    form = RegistrationForm()
    if form.validate_on_submit():
        user = User(username=form.username.data, email=form.email.data)
        user.set_password(form.password.data)
        db.session.add(user)
        db.session.commit()
        flash('Congratulations, you are now a registered user!')
        return redirect(url_for('login'))
    return render_template('register.html', title='Register', form=form)

