from flask import Flask, render_template, request, redirect, url_for, session
from PIL import Image
import cv2
import numpy as np
import os
from model.processing import Loader
from model import inference
import uuid
import yaml


# clean Directory
folder_path = "src/static/images"
files = os.listdir(folder_path)
for file in files:
    file_path = os.path.join(folder_path, file)
    try:
        if os.path.isfile(file_path):
            os.remove(file_path)
    except Exception as e:
        print(f"Error deleting {file}: {str(e)}")


models = []
models_path = {}

last_file = False

with open('models.yaml') as f:
    data = yaml.load(f, Loader=yaml.Loader)
    for model in data:
        models.append(model)
        mdl_pathes = []
        for model_path in data[model]:
            mdl_pathes.append([os.path.basename(model_path), model_path])
        models_path[model] = mdl_pathes

ld = Loader()
app = Flask(__name__)
app.secret_key = b'_5#F4Q8zepiofjewo4j579c-4=mcuF4Q8z_5#F4Q8z"F4Q8z\n\xec]/'


def Process_Anal(img_path, model, model_path):
    global last_file
    last_file = img_path
    img = cv2.imread(img_path)
    npimg = (ld.resizing(img))
    out = inference.infer([npimg], model, model_path)
    if out[0] == 0:
        return "štěpný lom"
    elif out[0] == 1:
        return "tvárný lom"


@app.route('/', methods=['GET', 'POST'])
def upload_file():
    global last_file
    # POST Method
    if request.method == 'POST':
        try:

            f = request.files['user_image']
            if f:
                image_uuid = str(uuid.uuid4())
                session["img_path"] = 'src/static/images/'+image_uuid+'.jpg'
                f.save(session["img_path"])
                return render_template('index.html', models=models, models_path=models_path, ansver=Process_Anal(session["img_path"],request.form['model'],request.form['model_path']), image="static/images/" + image_uuid + ".jpg")
                
            else:
                return render_template('index.html', models=models, models_path=models_path, error="No File Selected")
        
        except Exception as e:
            return "An error occurred: " + str(e)


    # GET Method    
    if last_file:
        os.remove(last_file)
        last_file = False
    return render_template('index.html', models=models, models_path=models_path, error=False)



if __name__ == '__main__':
    app.run(debug=True)
