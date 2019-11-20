from__future__ import division, print_function

import sys
import os
import glob
import re
#import numpy as np

#Keras
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.models import load_model
from keras.preprocesiing import image

#Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

#define a flask app
app=Flask(__name__)

MODEL_PATH='models/model_keras.h5'
model=load_model(MODEL_PATH)
model.make_predict_function()
print('Model yüklendi. Hizmet başlatılıyor..')

def model_predict(img_path,model):
    img=image.load_img(img_path,target_size=(150,150))

    #preprocessing image
    x=image.img_to_array(img)
    #x=np.true_divide(x,255)
    x=np.expand_dims(x,axis=0)
    #modelin girişine dikkat edin, aksi takdirde doğru tahmin yapamaz
    x=preprocess_input(x,mode='caffe')

    preds=model.predict(x)
    return preds


@app.route('/',methods=['GET'])
def index():
    #Ana sayfa
    return render_template('index.html')

@app.route('/predict',methods=['GET','POST'])
def upload():
    if request.method=='POST':
        #get the file from post request
        f=request.files['file']

        #save the file to ./uploads
        basepath=os.path.dirname(__file__)
        file_path=os.path.join(basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        #make prediction
        preds=model_predict(file_path,model)

        #process your result
        pred_class=decode_predictions(preds,top=1)
        result=str(pred_class[0][0][1])
        return result
    return None

if __name__=='__main__':
    http_server=WSGIServer(('',5000),app)
    http_server.serve_forever()

    


                        
