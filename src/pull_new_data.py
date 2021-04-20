import os
import boto3
import pickle
import psycopg2 as pg2
import psycopg2.extras as extras
from clean_data import Query_results
from sklearn.preprocessing import StandardScaler

def convert_cat_to_int(df):

    object_cols = df.select_dtypes(include=['object']).columns.tolist()

    for col in object_cols:
        col_dict = {}

        for idx, category in enumerate(df[col].unique()):
            col_dict[category] = idx

        df[col] = df[col].map(lambda x: col_dict[x])
        
        # scaler = StandardScaler()
        # df = scaler.fit_transform(df, 1)

    return df 

def pull_data(is_remote=False):

    if is_remote: 
        ssm = boto3.client('ssm', region_name='us-east-2')
        api_key = ssm.get_parameter(Name='REDASH_API_KEY', WithDecryption=True)['Parameter']['Value']
        query_url = ssm.get_parameter(Name='REDASH_LINK', WithDecryption=True)['Parameter']['Value']
    else:
        api_key = os.environ['REDASH_API_KEY']
        query_url = os.environ['REDASH_LINK']
    
    clean_dict = {'datetime_cols': ['signup_time_utc', 'last_order_time_utc']
                ,'target_column': 'last_order_time_utc'
                ,'days_to_churn': 30
                }
    
    churn_query_id = 714507
    churn_data = Query_results(query_url, churn_query_id, api_key, params={})
    churn_data.clean_data(clean_dict)
    
    X = churn_data.df.drop(['user_id', 'city_id', 'signup_time_utc', 'last_order_time_utc'], axis=1)
    X = convert_cat_to_int(X)

    return X, churn_data.df, churn_data.target

def execute_batch(conn, df, table, page_size=100):
    # Code example adapted from: 
    # https://naysan.ca/2020/05/09/pandas-to-postgresql-using-psycopg2-bulk-insert-performance-benchmark/
    
    tuples = [tuple(x) for x in df.to_numpy()]
    cols = ','.join(list(df.columns))
    query  = """INSERT INTO %s(%s) VALUES(%%s,%%s,%%s,%%s,%%s,%%s,%%s,%%s,%%s,%%s
                                         ,%%s,%%s,%%s,%%s,%%s,%%s,%%s,%%s,%%s,%%s
                                         ,%%s,%%s,%%s,%%s,%%s)""" % (table, cols)
    cursor = conn.cursor()
    try:
        extras.execute_batch(cursor, query, tuples, page_size)
        conn.commit()
    except (Exception, pg2.DatabaseError) as error:
        print("Error: %s" % error)
        conn.rollback()
        cursor.close()
        return 1
    cursor.close()

if __name__ == '__main__':

    print('Connecting to Database...')
    postgres_pw = ssm.get_parameter(Name='REDASH_API_KEY', WithDecryption=True)['Parameter']['Value']
    conn = pg2.connect(dbname='churn_database', user='postgres', password=postgres_pw, host='localhost', port='5432')

    print('Loading model...')
    with open('data/best_model.pkl', 'rb') as f:
        model = pickle.load(f)

    print('Pulling new data...')
    X, X_df, y = pull_data(is_remote=True)
    X_df['churn_prediction'] = model.predict_proba(X)[:,1]
    
    print('Deleting all rows...')
    cursor = conn.cursor()
    cursor.execute("DELETE FROM churn_predictions")
    conn.commit()

    print('Updating table with new predictions...')
    execute_batch(conn, X_df, 'churn_predictions')

    print('Done.')
