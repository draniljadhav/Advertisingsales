import json
import pickle
from flask import Flask,request,app,jsonify,url_for,render_template
import numpy as np
import pandas as pd

app=Flask(__name__)
regmodel=pickle.load(open('d:/anil/DS & DA Symbi/MLdeployment/Advertisingsales/regmodel.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict_api', methods=['POST'])



def predict_api():
    try:
        if request.is_json:
            data = request.get_json()
            print(f"Received data: {data}")
            new_data = np.array(list(data['data'].values())).reshape(1, -1)
            print(f"Reshaped data: {new_data}")
            output = regmodel.predict(new_data)
            print(f"Prediction output: {output[0]}")
            return jsonify({'prediction': output[0]})
        else:
            return jsonify({"error": "Request content-type must be application/json"}), 400
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({'error': str(e)}), 400


'''
@app.route('/predict',methods=['POST'])
def predict():
    data=[float(x) for x in request.form.values()]
    final_input=scalar.transform(np.array(data).reshape(1,-1))
    print(final_input)
    output=regmodel.predict(final_input)[0]
    return render_template("home.html",prediction_text="sales prediction is {}".format(output))
'''

if __name__ == "__main__":
    app.run(debug=True)
