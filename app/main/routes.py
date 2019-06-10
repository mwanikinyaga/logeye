from flask import render_template, request, Blueprint, jsonify, request, redirect, url_for
from app.models import Ownership
from random import sample
from app.models import Sms, SmsStatus
from flask_login import login_required
from app import db
import time
import datetime


main = Blueprint('main', __name__)



@main.route('/add/<int:id>', methods=['POST'])
@login_required
def add(id):
    x = db.session.query(SmsStatus).filter_by(sms_id=id).first()
    x.status_id = 2
    x.updated_date = datetime.datetime.now()
    db.session.commit()

    return redirect(url_for('main.dashboard'))

@main.route('/home')
def home():
    page = request.args.get('page', 1, type=int)
    posts = Ownership.query.order_by(Ownership.date_created.desc()).paginate(page=page, per_page=5)
    return render_template('home.html', posts=posts)


@main.route('/')



@main.route('/dashboard')
@login_required
def dashboard():
    legend = 'Monthly Data'
    labels = ["January", "February", "March", "April", "May", "June", "July", "August"]
    values = [10, 9, 8, 7, 6, 4, 7, 8]
    total_chainsaws = len(Sms.query.filter_by(pred_label='Chainsaw').all())
    total_vehicles = len(Sms.query.filter_by(pred_label='Vehicle').all())
    stat = SmsStatus.query.filter_by(status_id=1).join(Sms, SmsStatus.sms_id == Sms.id).all()
    pending = len(stat)
    addr = SmsStatus.query.filter_by(status_id=2).join(Sms, SmsStatus.sms_id == Sms.id).all()
    addressed = len(addr)
    points = Sms.query.all()
    coords = [[point.pred_label, point.latitude, point.longitude, point.rangername] for point in points]

    my_qry = """
        SELECT *
        FROM sms
        INNER JOIN sms_statuses
        ON sms.id = sms_statuses.sms_id
        INNER JOIN statuses
        ON sms_statuses.status_id = statuses.id;
    """

    connection = db.session.connection()
    data = connection.execute(my_qry)
    data = data.fetchall()





    for i in range(0, pending):
        x = stat[i].status_id

    return render_template('about.html', title = 'Dashboard', stat=stat, pending=pending, addressed=addressed, data=data, total_chainsaws=total_chainsaws, total_vehicles=total_vehicles, values=values, labels=labels, legend=legend, coords=coords)



@main.route('/data')
def points():
    points = Sms.query.all()
    coords = [[point.latitude, point.longitude] for point in points]
    return jsonify({"data": coords})

@main.route('/smscount')
def smscount():
    points = Sms.query.all()
    smscount = [[point.pred_label, point.last_sent_chainsaw, point.last_sent_vehicle] for point in points]

    data = [
        [0,0],
        [0,0],
        [0,0],
        [0,0],
        [0,0],
        [0,0],
        [0,0],
        [0,0],
        [0,0],
        [0,0],
        [0,0],
        [0,0]
    ]
    for item in smscount:
        print(str(item[1]))

        if str(item[1])[5:7] == "06": # month is June
            if str(item[1])[0:4] != "1970": # real data
                data[5][0] += 1
        elif str(item[2])[5:7] == "06": # month is June
            if str(item[2])[0:4] != "1970": # real data
                data[5][1] += 1


        # if str(item[1])[0:4] != "1970":   #populate vehicle
        #     if str(item[1])[5:7] == "03": # month March
        #         data[2][0] += 1
        #     elif str(item[1])[5:7] == "06":            # Month June
        #         data[5][0] += 1
        #
        # if str(item[2])[0:4] != "1970":   #populate chainsaw
        #      if str(item[1])[5:7] == "03": # month March
        #          data[2][1] += 1
        #      elif str(item[1])[5:7] == "06":            # Month June
        #          data[5][1] += 1
    # return jsonify({"data": data})
    return jsonify({"data": data})

