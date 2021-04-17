from flask import Flask, request, render_template
import pickle
import numpy as np
import pandas as pd
from src.pull_new_data import pull_data

app = Flask(__name__)

with open('data/best_model.pkl', 'rb') as f:
    model = pickle.load(f)

@app.route('/')
def index():
    return '''
        <form action="/predict" method='POST'>
            <input type="text" name="city_ids"/>
            
            <input type="text" name="lookback_days"/>

            <input type="text" name="threshold"/>
            <input type='submit' value="Predict Churn">
        </form>
    '''

@app.route('/predict', methods=['GET', 'POST'])
def predictions():
    city_ids = str(request.form['city_ids'])
    lookback_days = str(request.form['lookback_days'])
    X = pull_data(city_ids, lookback_days)
    print(X)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8105, debug=True, threaded=True)