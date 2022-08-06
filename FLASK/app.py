# -*- coding: utf-8 -*-
"""
Created on Sat Aug  6 12:06:08 2022

@author: Anitha
"""

import numpy as np
import pandas as pd
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from flask import Flask,render_template,request

app=Flask(__name__)
model=load_model("model_fruit.h5")
model1=load_model("model_vegetable.h5")

@app.route('/')
def index():
    return render_template("predict.html")

@app.route('/predict',methods=['GET','POST'])
def upload():
    if request.method=='POST':
        f=request.files['image']
        basepath=os.path.dirname(__file__)
        filepath=os.path.join(basepath,'uploads',f.filename)
        f.save(filepath)
        img=image.load_img(filepath,target_size=(64,64))
        x=image.img_to_array(img)
        x=np.expand_dims(x,axis=0)
        plant=request.form['plant']
        print(plant)
    if(plant=="fruit"):
            pred=np.argmax(model.predict(x),axis=1)
            print(pred)
            df=pd.read_excel('precautions - fruits.xlsx')
            print(df.iloc[pred[0]]['caution'])
            index=['Apple___Black_rot', 'Apple___healthy', 'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy','Peach___Bacterial_spot','Peach___healthy']
            text="The classified Fruit is " +str(index[pred[0]])
    else:
           pred=np.argmax(model1.predict(x),axis=1)
           print(pred)
           df=pd.read_excel('precautions - veg.xlsx')
           print(df.iloc[pred[0]]['caution'])
           index= ['Pepper,_bell___Bacterial_spot','Pepper,_bell___healthy','Potato___Early_blight','Potato___Late_blight','Potato___healthy','Tomato___Bacterial_spot','Tomato___Late_blight','Tomato___Leaf_Mold','Tomato___Septoria_leaf_spot']
           text="The classified Vegetable is " +str(index[pred[0]])
    return text

if __name__== '__main__':
    app.run(debug=False)