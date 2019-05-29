from flask import (render_template, url_for, flash,
                   redirect, request, abort, Blueprint)
from flask_login import current_user, login_required
from app import db
from app.models import Ownership
from app.pis.forms import PiForm



pis = Blueprint('pis', __name__)

@pis.route('/pi/new', methods=['GET', 'POST'])
@login_required
def new_pi():
    form = PiForm()
    if form.validate_on_submit():
        post = Ownership(raspi_id=form.raspi_id.data, phone=form.phone_no.data, owner=current_user)
        db.session.add(post)
        db.session.commit()
        flash('Your Pi has been added!','success')
        return redirect(url_for('main.home'))
    return render_template('create_pi.html', title='Add Pi',
                           form=form, legend='Add Pi')



@pis.route('/pi/<int:post_id>')
def pi(post_id):
    post = Ownership.query.get_or_404(post_id)
    return render_template('pi.html', title=post.raspi_id, post=post)


@pis.route('/pi/<int:post_id>/update', methods=['GET', 'POST'])
@login_required
def update_pi(post_id):
    post = Ownership.query.get_or_404(post_id)
    if post.owner != current_user:
        abort(403)
    form = PiForm()
    if form.validate_on_submit():
        post.raspi_id = form.raspi_id.data
        post.phone = form.phone_no.data
        db.session.commit()
        flash('Your Pi has been updated!', 'success')
        return redirect(url_for('pis.pi', post_id=post.id))
    elif request.method == 'GET':
        form.raspi_id.data = post.raspi_id
        form.phone_no.data = post.phone
    return render_template('create_pi.html', title='Update Pi',
                           form=form, legend='Update Pi')


@pis.route('/pi/<int:post_id>/delete', methods=['POST'])
@login_required
def delete_pi(post_id):
    post = Ownership.query.get_or_404(post_id)
    if post.owner != current_user:
        abort(403)
    db.session.delete(post)
    db.session.commit()
    flash('Your Pi has been deleted!', 'success')
    return redirect(url_for('main.home'))