from flask import Flask, render_template, request
import pickle
import numpy as np


app = Flask(__name__)

filename = 'final_model_TimeSeries1.pkl'
model = pickle.load(open(filename, 'rb'))

@app.route("/")
def template():
    return render_template('index.html')

@app.route("/submit", methods=['GET', 'POST'])
def predict():
    global prediction
    if request.method == "POST":
        p1 = request.form['Periods taken for auto regressive model']
        d = request.form['Difference']
        p2 = request.form['Periods in moving average model']
        result = np.array([[p1, d, p2]])
        prediction = model.predict(result)

    return render_template("submit.html", n=prediction)

if __name__ == "__main__":
    app.run(debug=True)


