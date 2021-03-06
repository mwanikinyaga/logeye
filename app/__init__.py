
from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from flask_marshmallow import Marshmallow
from flask_bcrypt import Bcrypt
from flask_login import LoginManager
from flask_mail import Mail

from app import momentjs
from app.config import Config


db = SQLAlchemy()
ma = Marshmallow()
bcrypt = Bcrypt()
login_manager = LoginManager()
login_manager.login_view = 'users.login'
login_manager.login_message_category = 'info'
mail = Mail()




def create_app(config_class=Config):
    app = Flask(__name__)
    app.config.from_object(Config)

    db.init_app(app)
    ma.init_app(app)
    bcrypt.init_app(app)
    login_manager.init_app(app)
    mail.init_app(app)
    app.jinja_env.globals['momentjs'] = momentjs

    from app.users.routes import users
    from app.pis.routes import pis
    from app.main.routes import main
    from app.errors.handlers import errors
    from app.gps.routes import gps
    from app.model.routes import ml
    app.register_blueprint(users)
    app.register_blueprint(pis)
    app.register_blueprint(main)
    app.register_blueprint(errors)
    app.register_blueprint(gps)
    app.register_blueprint(ml)
    return app


