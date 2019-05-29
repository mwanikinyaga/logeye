from flask_wtf import FlaskForm
from wtforms import StringField, SubmitField, ValidationError
from wtforms.validators import DataRequired
import phonenumbers


class PiForm(FlaskForm):
    raspi_id = StringField('Raspberry Pi Name', validators=[DataRequired()])
    phone_no = StringField('Ranger Phonenumber', validators=[DataRequired()])

    def validate_phone_no(form, field):
        if len(field.data) < 10 or len(field.data) > 13:
            raise ValidationError('Invalid phone number')
        try:
            input_number = phonenumbers.parse(field.data)
            if not (phonenumbers.is_valid_number(input_number)):
                raise ValidationError('Invalid phone number')
        except:
            input_number = phonenumbers.parse("+1" + field.data)
            if not (phonenumbers.is_valid_number(input_number)):
                raise ValidationError('Invalid phone number. Please use International Format')

    submit = SubmitField('Create')

