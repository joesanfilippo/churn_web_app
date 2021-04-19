from flask import Flask, request, render_template
import pickle
import numpy as np
import pandas as pd
import psycopg2 as pg2

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('cover.html')

@app.route('/inputs')
def inputs():
    return render_template('inputs.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    city_ids = str(request.form['city_ids'])
    lookback_days = int(request.form['lookback_days'])
    threshold = float(request.form['threshold'])
    
    query = """
            SELECT 
                user_id 
                ,churn_prediction

            FROM churn_predictions

            WHERE 1=1
                AND city_id in ({city_ids})
                AND days_since_signup <= {lookback_days}
                AND churn_prediction >= {threshold}

            ORDER BY churn_prediction DESC
            """.format(city_ids=city_ids, lookback_days=lookback_days, threshold=threshold)

    cursor.execute(query)
    results = cursor.fetchall()
    users = [result[0] for result in results]
    churn_predictions = [round(result[1]*100) for result in results]

    return render_template('predictions.html', data=zip(users, churn_predictions))

if __name__ == '__main__':
    print('Connecting to Database...')
    conn = pg2.connect(dbname='churn_database'
                      ,user='postgres'
                      ,password='galvanize'
                      ,host='localhost'
                      ,port='5432')
    
    cursor = conn.cursor()
    
    app.run(host='0.0.0.0', port=8105, debug=True, threaded=True)

    if conn:
        print('Disconnecting from Database...')
        cursor.close()
        conn.close()