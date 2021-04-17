import os
from src.clean_data import Query_results

def pull_data(city_ids, lookback_days):

    params = {'city_ids': city_ids
             ,'lookback_days': lookback_days}

    ssm = boto3.client('ssm')

    api_key = ssm.get_parameter(Names=['REDASH_API_KEY'], WithDecryption=True)
    query_url = ssm.get_parameter(Names=['REDASH_LINK'], WithDecryption=True)

    dynamic_churn_query_id = 744861
    dynamic_churn_data = Query_results(query_url, dynamic_churn_query_id, api_key, params)
    
    clean_dict = {'datetime_cols': ['signup_time_utc', 'last_order_time_utc']
                ,'target_column': 'last_order_time_utc'
                ,'days_to_churn': 30
                }

    dynamic_churn_data.clean_data(clean_dict)

    return dynamic_churn_data.df


if __name__ == '__main__':

    X = pull_data('1, 10', '90')
    print(X)