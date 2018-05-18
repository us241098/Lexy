import os
from flask import Flask, request, redirect, url_for
from werkzeug.utils import secure_filename
from headers import *
from regression import *
from multireg import *
from flask import render_template
UPLOAD_FOLDER = ''
ALLOWED_EXTENSIONS = set(['txt', 'csv'])


app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER



def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS




@app.route('/', methods=['GET', 'POST'])
def uplile():
    return render_template('index.html')


@app.route('/lin', methods=['GET', 'POST'])
def upload_file():
    global filename
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # if user does not select file, browser also
        # submit a empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
    return render_template('input.html')


@app.route('/headers',methods = ['POST', 'GET'])
def result():
    q=show_headers(filename)
    return render_template("headers.html",users=q)


@app.route('/result',methods = ['POST', 'GET'])
def final():
   if request.method == 'POST':
      result = request.form.getlist("selected")
      print result
      maino(filename,result)
      return render_template("result.html")



@app.route('/multiple', methods=['GET', 'POST'])
def upload_file2():
    global filename2
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # if user does not select file, browser also
        # submit a empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename2 = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename2))
    return render_template('input.html')



@app.route('/headers2',methods = ['POST', 'GET'])
def result2():
    q=show_headers(filename2)
    print q
    return render_template("headers2.html",users=q)


@app.route('/result2',methods = ['POST', 'GET'])
def final2():
   if request.method == 'POST':
      result2 = request.form.getlist("selected")
      print result2
      mainmulti(filename2,result2)
      return render_template("result2.html")



if __name__ == '__main__':
	app.run(debug=True)
