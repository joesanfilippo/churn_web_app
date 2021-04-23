from flask import Flask, request, render_template, make_response
import boto3
import pickle
import numpy as np
import pandas as pd
from train_test import retrain_model
import psycopg2 as pg2
import time 

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
        if city_ids and city_ids != 'City IDs':
            city_splits = city_ids.split(',')
            city_tup = tuple([int(x) for x in city_splits])
        else:
            city_tup = tuple([int(x) for x in np.arange(1,100)])

        lookback_days = int(request.form['lookback_days'])
        threshold = float(request.form['threshold'])
        cursor.execute(
                """
                SELECT 
                    user_id 
                    ,city_name
                    ,days_since_signup
                    ,first_30_day_orders
                    ,signup_to_order_hours
                    ,DATE_PART('day', now() - last_order_time_utc) as days_since_last_order
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
        results_df = pd.DataFrame(results, columns=['user_id'
                                                   ,'city_name'
                                                   ,'days_since_signup'
                                                   ,'first_30_day_orders'
                                                   ,'signup_to_order_hours'
                                                   ,'days_since_last_order'
                                                   ,'churn_prediction'])
        
        users = results_df['user_id']
        cities = results_df['city_name']
        days = results_df['days_since_signup']
        orders = results_df['first_30_day_orders']
        signup_hrs = round(results_df['signup_to_order_hours'],1)
        days_since_order = results_df['days_since_last_order'].astype(int)
        churn_predictions = round(results_df['churn_prediction']*100,1)
        cursor.close()
        return render_template('predictions.html'
                              ,data=zip(users
                                       ,cities
                                       ,days
                                       ,orders
                                       ,signup_hrs
                                       ,days_since_order
                                       ,churn_predictions)
                              ,dataframe=results_df.to_csv(index=False))
    
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

@app.route('/download', methods=['GET', 'POST'])
def download():
    timestamp = int(time.time())
    filename = 'churn_prediction_results_{}.csv'.format(timestamp)

    df_response = request.form['results_df']
    resp = make_response(df_response)
    resp.headers["Content-Disposition"] = "attachment; filename={}".format(filename)
    resp.headers["Content-Type"] = "text/csv"
    return resp

@app.route('/questions')
def questions():
    return render_template('questions.html')

if __name__ == '__main__':
    
    print('Connecting to Database...')
    ssm = boto3.client('ssm', region_name='us-east-2')
    api_key = ssm.get_parameter(Name='REDASH_API_KEY', WithDecryption=True)['Parameter']['Value']
    query_url = ssm.get_parameter(Name='REDASH_LINK', WithDecryption=True)['Parameter']['Value']
    db_password = ssm.get_parameter(Name='POSTGRES_PASSWORD', WithDecryption=True)['Parameter']['Value']

    conn = pg2.connect(dbname='churn_database'
                      ,user='postgres'
                      ,password=db_password
                      ,host='localhost'
                      ,port='5432')
    
    app.run(host='0.0.0.0', port=8105, debug=True, threaded=True)

    if conn:
        print('Disconnecting from Database...')
        conn.close()