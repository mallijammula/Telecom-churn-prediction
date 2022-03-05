# -*- coding: utf-8 -*-
"""
Created on Sat Feb 19 12:58:28 2022

@author: malli
"""
from flask import Flask, jsonify, request
import numpy as np
import joblib
# https://www.tutorialspoint.com/flask
import flask
app = Flask(__name__)


#@app.route('/')
#def hello_world():
 #   return 'Hello World!'


@app.route('/')
def index():
    return flask.render_template('input.html')


@app.route('/predict', methods=['POST'])
def predict():
    xgb = joblib.load('model_final.pkl')
    to_predict_list = request.form.to_dict()
    to_predict_list = list(to_predict_list.values())
    to_predict_list = np.array(list(map(float, to_predict_list))).reshape(1, -1)
    print(to_predict_list)
    pred = xgb.predict(to_predict_list)
    pred = list( ["%.2f" % x for x in pred])
    if pred[0]:
        prediction = "Not-churn"
    else:
        ptrediction = "churn"
        
    return jsonify({'prediction': prediction})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
#if __name__ == "__main__":    
        #socketio.run(app)
        #app.run(host='0.0.0.0', port=8080)

#python3 __init__.py


    #app.run(debug = True)
    
   
    
    
 



