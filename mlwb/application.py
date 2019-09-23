
from flask import Flask, render_template, request, redirect, url_for, session  #,flash
import os
import json
import boto3
import subprocess
import atexit
from bokeh.embed import server_document
#from werkzeug.utils import secure_filename


app = Flask(__name__)
app.secret_key = 'AWEDSDOLASJ'

bokeh_process = subprocess.Popen(
    ['bokeh', 'serve', '--allow-websocket-origin=127.0.0.1:5000',
        'classifier/dt.py', 'classifier/gbc.py', 'classifier/knn.py',
        'classifier/lr.py', 'classifier/mlp.py', 'classifier/rf.py',
        'classifier/sgd.py',
        'regressor/ll.py', 'regressor/nb.py', 'regressor/rdg.py',
        'regressor/svm.py'])

def main():
    port = int(os.environ.get('PORT', 5000))

    app.run(host='127.0.0.1', debug=True)

@atexit.register
def kill_server():
    bokeh_process.kill()


# Listen for GET requests to yourdomain.com/account/
@app.route("/")
def account():
    return render_template('index.html')


@app.route('/classification', methods=['GET', 'POST'])
def classification():
    if request.method == 'POST':
        return redirect(url_for('index'))
    return render_template('first.html')


@app.route('/regression', methods=['GET', 'POST'])
def regression():
    if request.method == 'POST':
        return redirect(url_for('index'))
    return render_template('third.html') 

#
@app.route("/letsplay_classification", methods=['POST'])
def play():
    if request.method == 'POST':
        f = request.files['file-input']
        selected_algo = request.form.get('algo')
        print(selected_algo)
        fname = f.filename
        print(fname)
        dict = {'dsname': fname}
        session['dsname'] = fname
        bokeh_script=server_document(url="http://127.0.0.1:5006/"+selected_algo,arguments=dict)
        return render_template('second.html',bokeh_script=bokeh_script,dsname=fname)


@app.route("/nextpage_classification", methods=['POST'])
def nxtpage():
    if request.method == 'POST':
        selected_algo = request.form.get('algo')
        print(selected_algo)
        dsname = session.get('dsname', None)
        dict = {'dsname': dsname}
        bokeh_script=server_document(url="http://127.0.0.1:5006/"+selected_algo,arguments=dict)
        return render_template('second.html',bokeh_script=bokeh_script)


@app.route("/letsplay_regression", methods=['POST'])
def play_reg():
    if request.method == 'POST':
        f = request.files['file-input']
        selected_algo = request.form.get('algo')
        print(selected_algo)
        fname = f.filename
        print(fname)
        dict = {'dsname': fname}
        session['dsname'] = fname
        bokeh_script=server_document(url="http://127.0.0.1:5006/"+selected_algo,arguments=dict)
        return render_template('fourth.html',bokeh_script=bokeh_script,dsname=fname)

@app.route("/nextpage_regression", methods=['POST'])
def nxtpage_reg():
    if request.method == 'POST':
        selected_algo = request.form.get('algo')
        print(selected_algo)
        dsname = session.get('dsname', None)
        dict = {'dsname': dsname}
        bokeh_script = server_document(url="http://127.0.0.1:5006/{}".format(selected_algo), arguments=dict)
        return render_template('fourth.html', bokeh_script=bokeh_script)


@app.route('/sign-s3/')
def sign_s3():
    # Load necessary information into the application
    #S3_BUCKET = os.environ.get('S3_BUCKET')
    S3_BUCKET = "cs297-mlplayground"
    # Load required data from the request
    file_name = request.args.get('file-name')
    file_type = request.args.get('file-type')

    # Initialise the S3 client
    s3 = boto3.client('s3')

    # Generate and return the presigned URL
    presigned_post = s3.generate_presigned_post(
        Bucket=S3_BUCKET,
        Key=file_name,
        Fields={"acl": "public-read", "Content-Type": file_type},
        Conditions=[
            {"acl": "public-read"},
            {"Content-Type": file_type}
        ],
        ExpiresIn = 3600
    )

    # Return the data to the client
    return json.dumps({
        'data': presigned_post,
        'url': 'https://%s.s3.amazonaws.com/%s' % (S3_BUCKET, file_name)
    })


# Main code
if __name__ == '__main__':
    main()
