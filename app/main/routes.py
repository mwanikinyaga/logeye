from flask import render_template, request, Blueprint, jsonify, request
from app.models import Ownership
from random import sample


main = Blueprint('main', __name__)



@main.route('/')
@main.route('/home')
def home():
    page = request.args.get('page', 1, type=int)
    posts = Ownership.query.order_by(Ownership.date_created.desc()).paginate(page=page, per_page=5)
    return render_template('home.html', posts=posts)



@main.route('/dashboard')
def dashboard():
    return render_template('about.html', title = 'Dashboard')

@main.route('/data')
def data():
    id = request.args.get('id')

    

    return jsonify({'results' : sample(range(1,10), 5)})


