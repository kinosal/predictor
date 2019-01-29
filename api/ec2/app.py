from flask import Flask, request, jsonify
from sklearn.externals import joblib
from boto.s3.connection import S3Connection
import pandas as pd
import config

app = Flask(__name__)

@app.route('/ping', methods=['POST'])
def ping():
    data = request.json
    return jsonify({'ping': 'successful'})

@app.route('/cpx', methods=['POST'])
def cpx():
    data = request.json
    cpi = predict(data, 'cost_per_impression')
    cpc = predict(data, 'cost_per_click')
    return jsonify({'cpi': cpi, 'cpc': cpc})

@app.route('/cpi', methods=['POST'])
def cpi():
    data = request.json
    prediction = predict(data, 'cost_per_impression')
    return jsonify({'cpi': prediction})

@app.route('/cpc', methods=['POST'])
def cpc():
    data = request.json
    prediction = predict(data, 'cost_per_click')
    return jsonify({'cpc': prediction})

def predict(data, cpx):
    model = load_from_bucket(cpx + '_model.pkl')
    columns = load_from_bucket(cpx + '_columns.pkl')
    set = pd.DataFrame(data).reindex(columns=columns, fill_value=0)
    prediction = model.predict(set)[0].round(4)
    return prediction

def load_from_bucket(key):
    connection = S3Connection(config.aws_access_key_id, config.aws_secret_access_key)
    bucket = connection.get_bucket('cpx-prediction')
    local_file = '/tmp/' + key
    bucket.get_key(key).get_contents_to_filename(local_file)
    model = joblib.load(local_file)
    return model

if __name__ == '__main__':
    app.run()

# scp -i ~/.ssh/stagelink.pem app.py ubuntu@ec2-34-252-110-37.eu-west-1.compute.amazonaws.com:/home/ubuntu/cpx-prediction/app.py

# Example JSON payload
# [
#     {
#         "total_budget": 10000,
#         "start_month": 7,
#         "end_month": 12,
#         "days_before": 0,
#         "budget_before": 0,
#         "days": 184,
#         "facebook": 1,
#         "instagram": 1,
#         "google_search": 1,
#         "google_display": 0,
#         "twitter": 0,
#         "facebook_likes": 1000,
#         "region_germany": 1,
#         "region_switzerland": 0,
#         "locality_single": 1,
#         "category_comedy": 0,
#         "category_conference": 0,
#         "category_music": 1,
#         "category_tradefair": 0,
#         "shop_actnews": 0,
#         "shop_eventbrite": 0,
#         "shop_eventim": 0,
#         "shop_reservix": 0,
#         "shop_showare": 0,
#         "shop_stagelink": 1,
#         "tracking_yes": 1
#     }
# ]
