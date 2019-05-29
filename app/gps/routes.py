from app import db
from app.models import Gps, gps_schema, gpsall_schema
from flask import Blueprint, request, jsonify, render_template

gps = Blueprint('gps', __name__)

@gps.route('/gps', methods=['POST'])
def add_gps():
    latitude = request.json['latitude']
    longitude = request.json['longitude']
    raspberry_id = request.json['raspberry_id']

    new_gps = Gps(latitude, longitude, raspberry_id)

    db.session.add(new_gps)
    db.session.commit()

    return gps_schema.jsonify(new_gps)


# Get all co-ordinates
@gps.route('/gps', methods=['GET'])
def get_gps_all():
    all_gps = Gps.query.all()
    result = gpsall_schema.dump(all_gps)
    return jsonify(result.data)

# Get single co-ordinates
@gps.route('/gps/<id>', methods=['GET'])
def get_gps(id):
    one_gps = Gps.query.get(id)
    return gps_schema.jsonify(one_gps)

# Update coordinates
@gps.route('/gps/<id>', methods=['PUT'])
def update_gps(id):
    coordinate = Gps.query.get(id)
    raspi_id = request.json['raspi_id']
    latitude = request.json['latitude']
    longitude = request.json['longitude']

    coordinate.raspi_id = raspi_id
    coordinate.latitude = latitude
    coordinate.longitude = longitude

    db.session.commit()

    return gps_schema.jsonify(coordinate)

# Delete single co-ordinates
@gps.route('/gps/<id>', methods=['DELETE'])
def delete_gps(id):
    one_gps = Gps.query.get(id)
    db.session.delete(one_gps)
    db.session.commit()

    return gps_schema.jsonify(one_gps)