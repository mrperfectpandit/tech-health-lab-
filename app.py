
from __future__ import division, print_function
from flask import Flask, render_template, request,url_for,flash,redirect,send_from_directory
import pickle
import numpy as np
import joblib
from PIL import Image

from flask_restful import Resource, Api
from package.patient import Patients, Patient
from package.doctor import Doctors, Doctor
from package.appointment import Appointments, Appointment
from package.common import Common
from package.medication import Medication, Medications
from package.department import Departments, Department
from package.nurse import Nurse, Nurses
from package.room import Room, Rooms
from package.procedure import Procedure, Procedures 
from package.prescribes import Prescribes, Prescribe
from package.undergoes import Undergoess, Undergoes

import json
import os

# temporary-------------------------------------------------------------
from tensorflow.keras.applications.imagenet_utils import preprocess_input, decode_predictions
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

import sys
import os
import glob
import re
import numpy as np
from werkzeug.utils import secure_filename


with open('config.json') as data_file:
    config = json.load(data_file)



app = Flask(__name__, static_url_path='')
app.config['SECRET_KEY'] = '5791628bb0b13ce0c676dfde280ba242'

api = Api(app)

api.add_resource(Patients, '/patient')
api.add_resource(Patient, '/patient/<int:id>')
api.add_resource(Doctors, '/doctor')
api.add_resource(Doctor, '/doctor/<int:id>')
api.add_resource(Appointments, '/appointment')
api.add_resource(Appointment, '/appointment/<int:id>')
api.add_resource(Common, '/common')
api.add_resource(Medications, '/medication')
api.add_resource(Medication, '/medication/<int:code>')
api.add_resource(Departments, '/department')
api.add_resource(Department, '/department/<int:department_id>')
api.add_resource(Nurses, '/nurse')
api.add_resource(Nurse, '/nurse/<int:id>')
api.add_resource(Rooms, '/room')
api.add_resource(Room, '/room/<int:room_no>')
api.add_resource(Procedures, '/procedure')
api.add_resource(Procedure, '/procedure/<int:code>')
api.add_resource(Prescribes, '/prescribes')
api.add_resource(Undergoess, '/undergoes')


dir_path = os.path.dirname(os.path.realpath(__file__))
# UPLOAD_FOLDER = dir_path + '/uploads'
# STATIC_FOLDER = dir_path + '/static'
UPLOAD_FOLDER = 'uploads'
STATIC_FOLDER = 'static'

model = load_model('models/model_malaria.h5')  #malaria
model222=load_model("models/my_model.h5") #pneumonia
model_tumor=load_model("models/tumor_prediction.h5") #tumor
#FOR THE FIRST MODEL

# call model to predict an image malaria
def api_malaria(full_path):
    data = image.load_img(full_path, target_size=(50, 50, 3))
    data = np.expand_dims(data, axis=0)
    data = data * 1.0 / 255

    #with graph.as_default():
    predicted = model.predict(data)
    return predicted

#FOR THE SECOND MODEL pneumonia
def api_pneumonia(full_path):
    data = image.load_img(full_path, target_size=(64, 64, 3))
    data = np.expand_dims(data, axis=0)
    data = data * 1.0 / 255

    #with graph.as_default():
    predicted = model222.predict(data)
    return predicted

# def api_tumor(full_path):
#     data = image.load_img(full_path, target_size=(50, 50, 3))
#     data = np.expand_dims(data, axis=0)
#     data = data * 1.0 / 255

#     #with graph.as_default():
#     predicted = model_tumor.predict(data)
#     return predicted
def api_tumor(full_path):
    data = image.load_img(full_path, target_size=(224,224, 3))
    data = image.img_to_array(data)
    data = np.expand_dims(data, axis=0)
    data = preprocess_input(data)
    # data = data * 1.0 / 255

    #with graph.as_default():
    predicted = model_tumor.predict(data)
    return predicted

# malaria
@app.route('/upload_malaria', methods=['POST','GET'])
def upload_file():

    if request.method == 'GET':
        return render_template('Malaria.html')
    else:
        try:
            file = request.files['image']
            full_name = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(full_name)

            # indices = {0: 'PARASITIC', 1: 'Uninfected', 2: 'Invasive carcinomar', 3: 'Normal'}
            indices = {0: 'Uninfected', 1: 'PARASITIC'}
            result = api_malaria(full_name)
            print(result)

            predicted_class = np.asscalar(np.argmax(result, axis=1))
            accuracy = round(result[0][predicted_class] * 100, 2)
            label = indices[predicted_class]
            return render_template('resultMalaria.html', image_file_name = file.filename, label = label, accuracy = accuracy)
        except:
            flash("Please select the image first !!", "danger")      
            return redirect(url_for("Malaria"))
#Pneumonia
@app.route('/upload_pneumonia', methods=['POST','GET'])
def upload11_file():

    if request.method == 'GET':
        return render_template('Pnuemonia.html')
    else:
        try:
            file = request.files['image']
            full_name = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(full_name)

            indices = {0: 'Normal', 1: 'Pneumonia'}
            result = api_pneumonia(full_name)

            #predicted_class = np.asscalar(np.argmax(result, axis=1))
            #accuracy = round(result[0][predicted_class] * 100, 2)
            #label = indices[predicted_class]
            if(result>50):
                label= indices[1]
                accuracy= result
            else:
                label= indices[0]
                accuracy= 100-result
            return render_template('resultpneumonia.html', image_file_name = file.filename, label = label, accuracy = accuracy)
        except:
            flash("Please select the image first !!", "danger")      
            return redirect(url_for("Pnuemonia"))

@app.route('/upload_tumor', methods=['POST','GET'])
def upload111_file():

    if request.method == 'GET':
        return render_template('tumor.html')
    else:
        try:
            file = request.files['image']
            full_name = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(full_name)

            # indices = {0: 'PARASITIC', 1: 'Uninfected', 2: 'Invasive carcinomar', 3: 'Normal'}
            indices = {0: 'not_tumor', 1: 'Tumor'}
            result = api_tumor(full_name)
            print(result)

            predicted_class = np.asscalar(np.argmax(result, axis=1))
            accuracy = round(result[0][predicted_class] * 100, 2)
            label = indices[predicted_class]
            return render_template('resulttumor.html', image_file_name = file.filename, label = label, accuracy = accuracy)
        except:
            flash("Please select the image first !!", "danger")      
            return redirect(url_for("tumor"))



@app.route('/uploads/<filename>')
def send_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)




@app.route("/")
def home():
    return render_template('index.html')

@app.route("/about")
def about():
    return render_template('about.html')

@app.route("/Diabetes")
def Diabetes():
	return render_template('diabetes.html')

@app.route("/Cancer")
def Cancer():
    return render_template('cancer.html')

@app.route("/Heart")
def Heart():
    return render_template('heart.html')

@app.route("/Kidney")
def Kidney():
    return render_template('kidney.html')

@app.route("/Liver")
def Liver():
    return render_template('liver.html')

@app.route("/Malaria")
def Malaria():
    return render_template('Malaria.html')

@app.route("/Pnuemonia")
def Pnuemonia():
    return render_template('Pnuemonia.html')

@app.route("/tumor")
def tumor():
    return render_template('tumor.html')

@app.route('/')
def index():
    return app.send_static_file('hospitalmanagement.html')

def ValuePredictor(to_predict_list, size):
    to_predict = np.array(to_predict_list).reshape(1,size)
    # diabetes
    if(size==8):
        loaded_model = joblib.load('diabetes-prediction-model.pkl')
        result = loaded_model.predict(to_predict)
    
    # cancer
    if(size==23):
        loaded_model = joblib.load('models/cancer-prediction-model.pkl')
        result = loaded_model.predict(to_predict)

    # heart
    if(size==7):
        loaded_model = joblib.load('models/heart_model.pkl')
        result = loaded_model.predict(to_predict)

    # kidney
    if(size==22):
        loaded_model = joblib.load('models/kidney-prediction-model.pkl')
        result = loaded_model.predict(to_predict)

    # liver
    if(size==10):
        loaded_model = joblib.load('models/liver-prediction-model.pkl')
        result = loaded_model.predict(to_predict)

    return result[0]

@app.route('/predict', methods = ["POST"])
def predict():
    if request.method == "POST":
        to_predict_list = request.form.to_dict()
        to_predict_list = list(to_predict_list.values())
        to_predict_list = list(map(float, to_predict_list))
         #diabetes
        if(len(to_predict_list)==8):
            result = ValuePredictor(to_predict_list,8)
            my_prediction = result
            return render_template('resultDiabetes.html', prediction=my_prediction)
        # cancer
        if(len(to_predict_list)==23):
            result = ValuePredictor(to_predict_list,23)
            my_prediction = result
            return render_template('resultCancer.html', prediction=my_prediction)
        # heart
        if(len(to_predict_list)==7):
            result = ValuePredictor(to_predict_list,7)
            my_prediction = result
            return render_template('resultHeart.html', prediction=my_prediction)
        # kidney
        if(len(to_predict_list)==22):
            result = ValuePredictor(to_predict_list,22)
            my_prediction = result
            return render_template('resultkidney.html', prediction=my_prediction)
        # liver
        if(len(to_predict_list)==10):
            result = ValuePredictor(to_predict_list,10)
            my_prediction = result
            return render_template('resultliver.html', prediction=my_prediction)

    # if(int(result)==1):
    #     prediction = "Sorry you chances of getting the disease. Please consult the doctor immediately"
    # else:
    #     prediction = "No need to fear. You have no dangerous symptoms of the disease"
    # return(render_template("result.html", prediction_text=prediction))       
    
    # my_prediction = result
    # return render_template('resultDiabetes.html', prediction=my_prediction)






if __name__ == "__main__":
    app.run(debug=True)