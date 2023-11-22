import numpy as np
from flask import Flask, request, render_template
import pickle

app=Flask(__name__)
model=pickle.load(open("model.pkl","rb"))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=["POST"])
def predict():
    float_features = [float(x) for x in request.form.values()]
    features = [np.array(float_features)]
    pred = model.predict(features)
    return render_template("index.html", prediction_text="The wine quality  is {}".format(pred))

if __name__=="__main__":
    app.run(debug=True)