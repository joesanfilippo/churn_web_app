from flask import Flask, request, render_template
import pickle
import numpy as np
import pandas as pd
from src.pull_new_data import pull_data

with open('data/best_model.pkl', 'rb') as f:
        model = pickle.load(f)

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('cover.html')

@app.route('/inputs')
def inputs():
    return '''
        <form action="/predict" method='POST'>
            <input type="text" name="city_ids"/>
            
            <input type="text" name="lookback_days"/>

            <input type="text" name="threshold"/>
            <input type='submit' value="Predict Churn">
        </form>
    '''

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    city_ids = str(request.form['city_ids'])
    lookback_days = str(request.form['lookback_days'])
    threshold = float(request.form['threshold'])
    X, user_ids = pull_data(city_ids, lookback_days)
    y_preds = model.predict_proba(X)[:,1]
    
    filtered_preds = np.around(y_preds[y_preds >= threshold] * 100, decimals=1)
    filtered_users = np.array(user_ids)[y_preds >= threshold]
    
    sorted_preds = filtered_preds[np.argsort(filtered_preds)[::-1]]
    sorted_users = filtered_users[np.argsort(filtered_preds)[::-1]]

    return render_template('predictions.html', data=zip(sorted_users, sorted_preds))

if __name__ == '__main__':
    
    app.run(host='0.0.0.0', port=8105, debug=True, threaded=True)