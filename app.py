
from __future__ import division, print_function
from flask import Flask, render_template, request,url_for,flash,redirect
import pickle
import numpy as np
import joblib
from PIL import Image



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






app = Flask(__name__)
app.config['SECRET_KEY'] = '5791628bb0b13ce0c676dfde280ba242'

dir_path = os.path.dirname(os.path.realpath(__file__))
# UPLOAD_FOLDER = dir_path + '/uploads'
# STATIC_FOLDER = dir_path + '/static'
UPLOAD_FOLDER = 'uploads'
STATIC_FOLDER = 'static'

model = load_model('models/model111.h5')  #malaria
model222=load_model("models/my_model.h5") #pneumonia

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

            indices = {0: 'PARASITIC', 1: 'Uninfected', 2: 'Invasive carcinomar', 3: 'Normal'}
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
        return render_template('index2.html')
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

@app.route("/dashboard")
def dashboard():
    return render_template('dashboard.html')

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