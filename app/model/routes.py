from flask import request, Blueprint, render_template
import redis, time, requests
from rq import Queue
from time import strftime
from app.model.tasks import model_run


ml = Blueprint('ml', __name__)


r = redis.Redis(socket_connect_timeout=86400)
q = Queue(connection=r)


@ml.route("/add-task", methods=["GET", "POST"])
def add_task():
    jobs = q.jobs  # Get a list of jobs in the queue
    message = None

    if request.args:  # Only run if a query string is sent in the request

        begin = request.args.get("url")  # Gets the URL coming in as a query string

        task = q.enqueue(model_run, begin, job_timeout='24h')  # Send a job to the task queue

        jobs = q.jobs  # Get a list of jobs in the queue

        q_len = len(q)  # Get the queue length

        message = f"Task queued at {task.enqueued_at.strftime('%a, %d %b %Y %H:%M:%S')}. {q_len} jobs queued"

    return render_template("add_task.html", title='Add Task', message=message, jobs=jobs)