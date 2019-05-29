from datetime import datetime
from flask_login import UserMixin
from itsdangerous import TimedJSONWebSignatureSerializer as Serializer
from flask import current_app
from app import db, ma, login_manager
from sqlalchemy_utils import PhoneNumber

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

class User(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(20), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    image_file = db.Column(db.String(20), nullable=False, default='default.jpg')
    password = db.Column(db.String(60), nullable=False)
    ownership = db.relationship('Ownership', backref='owner', lazy=True)
    #sms_user = db.relationship('Sms', backref='user', lazy=True)

    def get_reset_token(self, expires_sec=1800):
        s = Serializer(current_app.config['SECRET_KEY'], expires_sec)
        return s.dumps({'user_id': self.id}).decode('utf-8')

    @staticmethod
    def verify_reset_token(token):
        s = Serializer(current_app.config['SECRET_KEY'])
        try:
            user_id = s.loads(token)['user_id']
        except:
            return None
        return User.query.get(user_id)


    def __repr__(self):
        return f"User('{self.username}', '{self.email}','{self.image_file}')"

class Ownership(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    raspi_id = db.Column(db.String(20), unique=True, nullable=False)
    date_created = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)
    phone = db.Column(db.String(20), nullable=False)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    raspberry = db.relationship('Gps', backref='raspberry')

    def __repr__(self):
        return f"Post('{self.raspi_id}','{self.phone}')"

class Gps(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    latitude = db.Column(db.Float(20))
    longitude = db.Column(db.Float(20))
    time_created = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)
    raspberry_id = db.Column(db.Integer, db.ForeignKey('ownership.id'))

    def __init__(self, latitude, longitude, raspberry_id):

        self.latitude = latitude
        self.longitude = longitude
        self.raspberry_id = raspberry_id

    def __repr__(self):
        return f"Gps('{self.latitude}', '{self.longitude}')"


class GpsSchema(ma.Schema):
    class Meta:
        fields = ('id', 'latitude', 'longitude', 'time_created', 'raspberry_id')

gps_schema = GpsSchema(strict=True)
gpsall_schema = GpsSchema(many=True, strict=True)


class Sms(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    #user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    last_sent_chainsaw = db.Column(db.DateTime, nullable=False, default=0)
    last_sent_vehicle = db.Column(db.DateTime, nullable=False, default=0)

    def __init__(self, last_sent_chainsaw, last_sent_vehicle):
        self.last_sent_chainsaw = last_sent_chainsaw
        self.last_sent_vehicle = last_sent_vehicle



