from flask import Flask, render_template, request, redirect, url_for
from PIL import Image
import cv2
import numpy as np
import os
from model.process import Loader
ld = Loader()
from model import inference
app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        try:
            f = request.files['user_image']
            if f:
                # Update the path to your 'uploads' directory
                upload_path = '/home/andry/HACKHAHAHAHAH/plajta/src/static/images/uploaded_file.jpg'
                f.save(upload_path)
                return redirect(url_for('page1'))
            else:
                return "No file selected."
        except Exception as e:
            return "An error occurred: " + str(e)
    return render_template('index.html')

@app.route('/page1', methods=['GET','POST'])
def page1():
    if request.method == 'POST':
        return redirect(url_for('page2'))
    full_filename = os.path.join('static','images', 'uploaded_file.jpg')
    return render_template("page1.html", user_image = full_filename)

@app.route('/page2', methods=['GET','POST'])
def page2():
    full_filename = os.path.join('static','images', 'uploaded_file.jpg')
    npimg = ld.resizing(cv2.imread("/home/andry/HACKHAHAHAHAH/plajta/src/static/images/uploaded_file.jpg"))
    out = inference.infer_CNN(npimg)

    return out


if __name__ == '__main__':
    app.run(debug=True)
