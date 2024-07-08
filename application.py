import pickle
import numpy as np
import pandas as pd
from flask import Flask,request,render_template,jsonify
from sklearn.preprocessing import StandardScaler

application=Flask(__name__)
app=application

ridge_model=pickle.load(open('models/RidgeRegresson.pkl','rb'))
standard_scaler=pickle.load(open('models/Standarization.pkl','rb'))

@app.route("/")
def index():
    return render_template('index1.html')

@app.route("/predicted",methods=['GET','POST'])
def predicted_datapoints():
    if request.method=='POST':
        Temperature = float(request.form.get(('Temperature')))
        RH = float(request.form.get(('RH')))
        Ws = float(request.form.get(('Ws')))
        Rain = float(request.form.get(('Rain')))
        FFMC = float(request.form.get(('FFMC')))
        DMC = float(request.form.get(('DMC')))
        ISI = float(request.form.get(('ISI')))
        Classes = float(request.form.get(('Classes')))
        Region = float(request.form.get(('Region')))

        new_data=standard_scaler.transform([[Temperature,RH,Ws,Rain,FFMC,DMC,ISI,Classes,Region]])
        result=ridge_model.predict(new_data)

        return render_template('home.html',results=result[0])


    else:
        return render_template('home.html')
    

if __name__=="__main__":
    app.run(host="0.0.0.0")



