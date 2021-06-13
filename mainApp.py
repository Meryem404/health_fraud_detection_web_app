from flask import Flask, render_template, jsonify, request
from pycaret import *
from pycaret.classification import *
import pickle
# from joblib import load
import numpy as np
import pandas as pd
# from webapp import app
# from app.main.constant import constants as Constants

app = Flask(__name__)
loaded_model = pickle.load(open('insurance_predictfraud_flask.pkl', 'rb'))
#loaded_model =load('model.joblib')

cols=['BeneID', 'ClaimID', 'InscClaimAmtReimbursed', 'DeductibleAmtPaid',
       'RenalDiseaseIndicator', 'ChronicCond_Alzheimer',
       'ChronicCond_Heartfailure', 'ChronicCond_KidneyDisease',
       'ChronicCond_Cancer', 'ChronicCond_ObstrPulmonary',
       'ChronicCond_Depression', 'ChronicCond_Diabetes',
       'ChronicCond_IschemicHeart', 'ChronicCond_Osteoporasis',
       'ChronicCond_rheumatoidarthritis', 'ChronicCond_stroke', 'WhetherDead',
       'NumPhysicians', 'NumProc', 'NumUniqueClaims', 'ExtraClm',
       'AdmissionDays', 'ClaimDays', 'Hospt', 'NoOfMonths_PartACov',
       'NoOfMonths_PartBCov', 'IPAnnualReimbursementAmt',
       'IPAnnualDeductibleAmt', 'OPAnnualReimbursementAmt',
       'OPAnnualDeductibleAmt', 'Age']
##########
@app.route('/predict',methods=['POST'])

def predict():
    int_features = [x for x in request.form.values()]
    print("==============int_features====", int_features)
    final = np.array(int_features)
    print("============final=======", final)
    # data_unseen = pd.DataFrame(final, columns=cols).transpose()
    data_unseen = pd.DataFrame([final], columns=cols)
    print("=========data_unseen====", data_unseen)
    # prediction = predict_model(loaded_model, data=data_unseen, round=0)
    prediction = loaded_model.predict(data_unseen)
    #prediction = predict_model(loaded_model,[final] , round=0)
    # prediction = int(prediction.Label[0])
    prediction = int(prediction)
    if format(prediction) == 1:
        return render_template('base.html', pred='This patient\'s claim is fraudulent')
    else:
        return render_template('base.html', pred='This patient\'s claim is not fraudulent')

@app.route('/predict_api',methods=['POST'])

def predict_api():
    data = request.get_json(force=True)
    data_unseen = pd.DataFrame([data])
    prediction = predict_model(loaded_model, data=data_unseen)
    output = prediction.Label[0]
    return jsonify(output)
##########
@app.route('/')

def home():
    return render_template("index.html")

if __name__ == '__main__':
    app.run(debug=True)
