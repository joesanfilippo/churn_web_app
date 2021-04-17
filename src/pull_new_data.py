import os
import boto3
import pickle
from src.clean_data import Query_results
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

def pull_data(city_ids, lookback_days):

    is_remote = False

    params = {'city_ids': city_ids
             ,'lookback_days': lookback_days}

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

    dynamic_churn_query_id = 744861
    dynamic_churn_data = Query_results(query_url, dynamic_churn_query_id, api_key, params)
    dynamic_churn_data.clean_data(clean_dict)
    
    user_ids = dynamic_churn_data.df.pop('user_id')
    dynamic_churn_data.df.drop(['signup_time_utc', 'last_order_time_utc'], axis=1, inplace=True)
    X = convert_cat_to_int(dynamic_churn_data.df)

    return X, user_ids

if __name__ == '__main__':

    with open('data/best_model.pkl', 'rb') as f:
        model = pickle.load(f)

    X = pull_data('1', '90')
    print(model.predict(X['user_id']))
