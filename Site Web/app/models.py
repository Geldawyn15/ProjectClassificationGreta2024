from typing import Optional
import sqlalchemy as sa
import sqlalchemy.orm as so
from flask_login import UserMixin
from werkzeug.security import generate_password_hash, check_password_hash
from app import db, login



class User(UserMixin, db.Model):
    id: so.Mapped[int] = so.mapped_column(primary_key=True)
    username: so.Mapped[str] = so.mapped_column(sa.String(64), index=True,
                                                unique=True)
    email: so.Mapped[str] = so.mapped_column(sa.String(120), index=True,
                                             unique=True)
    password_hash: so.Mapped[Optional[str]] = so.mapped_column(sa.String(256))
    

    def __repr__(self):
        return '<User {}>'.format(self.username)

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)
    
@login.user_loader
def load_user(id):
    return db.session.get(User, int(id))
    

class Client(db.Model):
    id: so.Mapped[int] = so.mapped_column(primary_key=True)
    
    name: so.Mapped[str] = so.mapped_column(sa.String(64))
    
    oldBalance: so.Mapped[int] = so.mapped_column(sa.Integer())
    
    newBalance: so.Mapped[int] = so.mapped_column(sa.Integer())
    
    def __repr__(self):
        return '<Client {}>'.format(self.body)

class Transaction(db.Model):
    id: so.Mapped[int] = so.mapped_column(primary_key=True)
    
    step: so.Mapped[int] = so.mapped_column(sa.Integer())
    
    type: so.Mapped[str] = so.mapped_column(sa.String(50))
    
    amount: so.Mapped[int] = so.mapped_column(sa.Integer())
    
    isFraud: so.Mapped[bool] = so.mapped_column(sa.Boolean())
    
    id_client_destinaire: so.Mapped[int] = so.mapped_column(sa.ForeignKey(Client.id), index=True)
    
    id_client_originaire: so.Mapped[int] = so.mapped_column(sa.ForeignKey(Client.id), index=True)


    def __repr__(self):
        return '<Transaction {}>'.format(self.body)
    
    
