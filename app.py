import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib
from flask import Flask, request, render_template
import numpy as np
from training_model import Model as m
from database import Database as D

model = joblib.load('sales_prediction_model.pkl')
mdl = m()
df = mdl.df
dtbs = D()
app = Flask(__name__)

@app.route('/')
def Home():
    return render_template('home.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    prediction = None  # Default value

    if request.method == 'POST':
        # Retrieve form values and convert to floats
        features = [float(x) for x in request.form.values()]
        X_new = np.array([features])
        # Prediction
        prediction = model.predict(X_new)[0]  # Get scalar from array

    return render_template('predict_sale.html', prediction=prediction)

@app.route('/graphs', methods=['GET', 'POST'])
def graphs():
    image_file = None
    selected_x = None
    selected_y = None
    selected_plot = None
    title = ''
    try:
        df = pd.read_csv('Walmart_Sales.csv')
        print(df.head())
    except Exception as e:
        print("Failed to read CSV:", e)
        df = pd.DataFrame()  

    columns = df.columns.tolist()
    if request.method == 'POST':
        selected_x = request.form.get('xColumn')
        selected_y = request.form.get('yColumn')
        selected_plot = request.form.get('plotType')
        if selected_plot == 'bar':
            plt.bar(df[selected_x], df[selected_y], color='teal')
            title = f'Bar Chart {selected_x} VS {selected_y}'
            plt.title(title)
            plt.xlabel(selected_x)
            plt.ylabel(selected_y)
            plt.savefig("static/images/bar.png")
            plt.close()
            image_file = 'images/bar.png'
        elif selected_plot == 'line':
            df.sort_values(by=selected_x, inplace=True)
            plt.plot(df[selected_x], df[selected_y], marker='o', linestyle='-', color='blue')
            title = f'Line Chart {selected_x} VS {selected_y}'
            plt.title(title)
            plt.xlabel(selected_x)
            plt.ylabel(selected_y)
            plt.grid(True)
            plt.savefig('static/images/line.png')
            plt.close()
            image_file = 'images/line.png'
    return render_template('graphs.html', columns=columns, image_file=image_file)

@app.route('/train-model', methods = ['GET', 'POST'])
def train_model():
    s = None
    if request.method == 'POST':
        s = mdl.train()
    return render_template('train.html', score=s)

@app.route('/add-data', methods = ['GET', 'POST'])
def add_data():
    columns = list(zip(df.columns, df.dtypes))
    input_data = {}
    result = None
    message = None
    if request.method == 'POST':
        input_data = request.form.to_dict()
        for name, dtype in columns:
            if dtype in ['int']:
                input_data[name] = int(input_data[name])
            elif dtype in ['float']:
                input_data[name] = float(input_data[name])
        result = dtbs.add_single_document(input_data)
        if result:
            message = 'Insertion Successful'
            print(message)
    return render_template('Add_data.html', columns=columns, message=message)

if __name__ == '__main__':
    app.run(debug=True)