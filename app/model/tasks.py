import json, os, time, requests, datetime, pytz
from os import listdir
from pathlib import Path
import sqlite3
from sqlite3 import Error
import africastalking
from operator import itemgetter
from app import db, create_app
from app.models import Sms, Gps, Ownership, SmsStatus


wd = os.path.realpath(__file__)
p = (os.path.dirname(os.path.abspath(os.path.dirname(os.path.abspath(wd)))))
BASE_DIR = os.getcwd()
directory_root = str(p) + '/assets/live/'
username = 'logeye'
api_key = '82de6dd2ea7618a0ad4b392f3a5ac4d5e8c7cc56cf3345102b1a83e7a05c0aea'
africastalking.initialize(username, api_key)
sms = africastalking.SMS
tdelta = datetime.timedelta(minutes=2)
tdelta_sms_count = datetime.timedelta()
count_sms_since = datetime.datetime(2019, 5, 26, 0, 00, 00, 000)


def model_run(begin):

    start = time.time()

    app = create_app()
    with app.app_context():
        #pi = db.session.query(Ownership).order_by(
         #   Ownership.phone.desc()).all()
        o = Ownership.query.filter_by(user_id=1).first()
        texting = o.phone

        ranger = o.rangername


        while True:
            print("Checking for audio files to process")
            if len(listdir(directory_root)) > 0:
                try:
                    files_in_directory = sorted(listdir(directory_root), reverse=False)

                    for file in files_in_directory:
                        file_name = str(directory_root) + str(file)

                        since = datetime.datetime.now() - tdelta

                        if (file is not None and file.lower().endswith('.wav')):
                            print("Processing audio file " + str(file))
                            file_to_send = open(directory_root + str(file), 'rb')

                            files = {
                                'audio': (file_to_send),
                            }

                            response = requests.post('http://52.170.80.136:5000/model/predict', files=files)

                            file_to_send.close()

                            json_array = json.loads(response.text)
                            jpredictions = json_array['predictions']
                            maxPrediction = max(jpredictions, key=itemgetter('probability'))['probability']

                            for jsarr in json_array["predictions"]:
                                if (jsarr["label_id"] == "/m/01j4z9" and jsarr["probability"] == maxPrediction):
                                    print(str(jsarr) + "\n\n")
                                    # Set the numbers you want to send to in international format

                                    recipients = [texting]
                                    responsible = ranger

                                    try:

                                        print(since)
                                        q = (db.session.query(Sms).filter(
                                            Sms.last_sent_chainsaw > since).all())
                                        #print(q)

                                        if not q:
                                            coordinates = db.session.query(Gps).order_by(
                                                Gps.time_created.desc()).first()
                                            lat = str(coordinates.latitude)
                                            lng = str(coordinates.longitude)

                                            pred_label = jsarr["label"]
                                            prob_label = jsarr["probability"]
                                            last_sent_chainsaw = datetime.datetime.now()
                                            last_sent_vehicle = datetime.datetime(1970, 3, 22, 16, 24, 45, 10000)

                                            time_obj = Sms(pred_label, prob_label, last_sent_chainsaw, last_sent_vehicle, lat, lng, responsible)
                                            db.session.add(time_obj)
                                            db.session.commit()

                                            # Set your message
                                            message = "Chainsaws active here:\n\n" "http://maps.google.com/?q=" + lat + "," + lng

                                            print(message)


                                            response = sms.send(message, recipients)

                                            m = db.session.query(Sms).order_by(Sms.id.desc()).first()
                                            db.session.add(SmsStatus(sms_id=m.id, status_id=1))
                                            db.session.commit()



                                    except Exception as e:
                                        print('Encountered an error while sending: %s' % str(e))

                                elif (jsarr["label_id"] == "/m/07yv9" and jsarr["probability"] == maxPrediction):

                                    print(str(jsarr) + "\n\n")
                                    # Set the numbers you want to send to in international format
                                    recipients = [texting]


                                    try:
                                        q = (db.session.query(Sms).filter(
                                            Sms.last_sent_vehicle > since).all())
                                        #print(q)

                                        if not q:
                                            coordinates = db.session.query(Gps).order_by(
                                                Gps.time_created.desc()).first()
                                            lat = str(coordinates.latitude)
                                            lng = str(coordinates.longitude)
                                            pred_label = jsarr["label"]
                                            prob_label = jsarr["probability"]
                                            last_sent_chainsaw = datetime.datetime(1970, 3, 22, 16, 24, 45, 10000)
                                            last_sent_vehicle = datetime.datetime.now()
                                            #print(last_sent_vehicle)
                                            time_obj_2 = Sms(pred_label, prob_label, last_sent_chainsaw, last_sent_vehicle, lat, lng, responsible)
                                            db.session.add(time_obj_2)
                                            db.session.commit()



                                            vehicle_send_count = (db.session.query(Sms).filter(
                                                Sms.last_sent_vehicle > count_sms_since).all())
                                            #print(vehicle_send_count)

                                            # Set your message
                                            message = "Vehicle presence detected here:\n\n" "http://maps.google.com/?q=" + lat + "," + lng
                                            print(message)
                                            response = sms.send(message, recipients)

                                            m = db.session.query(Sms).order_by(Sms.id.desc()).first()
                                            db.session.add(SmsStatus(sms_id=m.id, status_id=1))
                                            db.session.commit()



                                    except Exception as e:
                                        print('Encountered an error while sending: %s' % str(e))

                        time.sleep(float(.15))
                        os.remove(file_name)
                except Exception as e:
                    print("Error: " + str(e))
            else:
                print("No audio files found for processing")
            time.sleep(float(15))

    end = time.time()

    time_elapsed = end - start
    print(f"Time elapsed: {time_elapsed} ")


