from flask import Flask, render_template, request
from sklearn.linear_model import LogisticRegression
import numpy as np
import pickle

app = Flask(__name__)
model = pickle.load(open("Model/model-0/model.pickle", 'rb'))

@app.route('/',methods=["POST", "GET"])
def index():
    if request.method == "POST":
        try:
            age = int(request.form['txtAge'])
            sex = int(request.form['ddlSex'])
            exang = int(request.form['exang'])
            ca = int(request.form['ddlCa'])
            cp = int(request.form['ddlCp'])
            bp = int(request.form['txtBP'])
            chol = int(request.form['txtChol'])
            fbs = int(request.form['Fbs'])
            rest_ecg = int(request.form['ddlRest_ecg'])
            thalach = int(request.form['txtThalach'])
            thal = int(request.form['ddlThal'])
            old_peak = 0.62
            slope = 2
            
            arr = [
                age, sex, cp, bp, chol, fbs, rest_ecg,
                thalach, exang, old_peak, slope, ca, thal
            ]
            
            pred = model.predict([arr])[0]
            pred = 'HIGH' if pred ==1 else 'LOW'
            return render_template('index.html', msg=pred)
        except Exception as e:
            print(e)
            return render_template('index.html', msg='')
    return render_template('index.html', msg='')

if __name__ == '__main__':
    app.run(debug=True)