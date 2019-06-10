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
    rangername = db.Column(db.String(20), nullable=False)
    phone = db.Column(db.String, nullable=False)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    raspberry = db.relationship('Gps', backref='raspberry')

    def __repr__(self):
        return f"Post('{self.raspi_id}','{self.rangername}','{self.phone}')"

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


class Status(db.Model):
    __tablename__ = 'statuses'

    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(120))

    def __init__(self, name):
        self.name = name

    def __repr__(self):
        return "%s(%s)" % (self.__class__.__name__, self.name)


class Sms(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    #user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    pred_label = db.Column(db.String(20), nullable=False)
    prob_label = db.Column(db.Float(20), nullable=False)
    last_sent_chainsaw = db.Column(db.DateTime, nullable=False, default=datetime.utcnow())
    last_sent_vehicle = db.Column(db.DateTime, nullable=False, default=datetime.utcnow())
    latitude = db.Column(db.Float(20))
    longitude = db.Column(db.Float(20))
    rangername = db.Column(db.String(20), nullable=False)

    def __init__(self, pred_label, prob_label, last_sent_chainsaw, last_sent_vehicle, lat, lng, responsible):
        self.pred_label = pred_label
        self.prob_label = prob_label
        self.last_sent_chainsaw = last_sent_chainsaw
        self.last_sent_vehicle = last_sent_vehicle
        self.latitude = lat
        self.longitude = lng
        self.rangername = responsible

    def __repr__(self):
        return "%s(%s)" % (self.__class__.__name__, self.id)


class SmsStatus(db.Model):

    __tablename__ = 'sms_statuses'

    sms_id = db.Column(db.Integer, db.ForeignKey('sms.id'), primary_key=True)
    status_id = db.Column(db.Integer, db.ForeignKey('statuses.id'), primary_key=True)

    created_date = db.Column(db.DateTime, default=db.func.current_timestamp())
    updated_date = db.Column(db.DateTime, default=db.func.current_timestamp(), onupdate=db.func.current_timestamp())

    def __init__(self, sms_id=None, status_id=None):
        self.sms_id = sms_id
        self.status_id = status_id

    def __repr__(self):
        return "%s(%s, %s)" % (self.__class__.__name__, self.sms_id, self.status_id)
        # return "{class_name}({sms_id}, {status_id})".format(class_name=self.__class__.__name__, sms_id=self.sms_id, status_id=self.status_id)