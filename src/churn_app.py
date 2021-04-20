from flask import Flask, request, render_template
import boto3
import pickle
import numpy as np
import pandas as pd
from train_test import retrain_model
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
    cursor = conn.cursor()
    try:
        city_ids = str(request.form['city_ids'])
        city_splits = city_ids.split(',')
        city_tup = tuple([int(x) for x in city_splits])
        lookback_days = int(request.form['lookback_days'])
        threshold = float(request.form['threshold'])
        cursor.execute(
                """
                SELECT 
                    user_id 
                    ,city_name
                    ,days_since_signup
                    ,first_30_day_orders
                    ,churn_prediction

                FROM churn_predictions

                WHERE 1=1
                    AND city_id in %(city_ids)s
                    AND days_since_signup <= %(lookback_days)s
                    AND churn_prediction >= %(threshold)s

                ORDER BY churn_prediction DESC
                """, {'city_ids':city_tup, 'lookback_days':lookback_days, 'threshold':threshold}
        )

        results = cursor.fetchall()
        users = [result[0] for result in results]
        cities = [result[1] for result in results]
        days = [result[2] for result in results]
        orders = [result[3] for result in results]
        churn_predictions = [round(result[4]*100, 1) for result in results]
        cursor.close()
        return render_template('predictions.html', data=zip(users, cities, days, orders, churn_predictions))
    
    except (Exception, pg2.DatabaseError) as error:
        print("Error: %s" % error)
        conn.rollback()
        cursor.close()
        return '''These are not valid SQL parameters'''

@app.route('/retrain')
def retrain():
    return render_template('retrain.html')

@app.route('/retrain_model')
def retrain_churn_model():
    name, hyper_params, score = retrain_model()
    return render_template('retrain_model.html', name=name, params=hyper_params, score=round(score, 3))

if __name__ == '__main__':
    is_remote=True

    print('Connecting to Database...')
    
    if is_remote: 
        ssm = boto3.client('ssm', region_name='us-east-2')
        api_key = ssm.get_parameter(Name='REDASH_API_KEY', WithDecryption=True)['Parameter']['Value']
        query_url = ssm.get_parameter(Name='REDASH_LINK', WithDecryption=True)['Parameter']['Value']
        db_password = ssm.get_parameter(Name='POSTGRES_PASSWORD', WithDecryption=True)['Parameter']['Value']
    else:
        api_key = os.environ['REDASH_API_KEY']
        query_url = os.environ['REDASH_LINK']

    conn = pg2.connect(dbname='churn_database'
                      ,user='postgres'
                      ,password=db_password
                      ,host='localhost'
                      ,port='5432')
    
    app.run(host='0.0.0.0', port=8105, debug=True, threaded=True)

    if conn:
        print('Disconnecting from Database...')
        conn.close()