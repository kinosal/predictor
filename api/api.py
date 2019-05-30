from flask import Flask, request, jsonify
from boto.s3.connection import S3Connection
import pandas as pd
import joblib
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
    return jsonify({'cpi': cpi[0], 'cpc': cpc[0]})


@app.route('/cpi', methods=['POST'])
def cpi():
    data = request.json
    prediction = predict(data, 'cost_per_impression')
    cpi = prediction[0]
    ctr = prediction[1]
    budget = data[0]['total_budget']
    impressions = (budget / cpi).round(0)
    clicks = (impressions * ctr).round(0)
    return jsonify({'cpi': cpi, 'impressions': impressions, 'clicks': clicks})


@app.route('/cpc', methods=['POST'])
def cpc():
    data = request.json
    prediction = predict(data, 'cost_per_click')
    cpc = prediction[0]
    ctr = prediction[1]
    budget = data[0]['total_budget']
    clicks = (budget / cpc).round(0)
    impressions = (clicks / ctr).round(0)
    return jsonify({'cpc': cpc, 'impressions': impressions, 'clicks': clicks})


def predict(data, output):
    filter = 'pay_per_' + output[9:]
    cpx_model = load_from_bucket(output + '_' + filter + '_model.pkl')
    cpx_columns = load_from_bucket(output + '_' + filter + '_columns.pkl')
    ctr_model = load_from_bucket('click_rate_' + filter + '_model.pkl')
    ctr_columns = load_from_bucket('click_rate_' + filter + '_columns.pkl')
    cpx_set = pd.DataFrame(data).reindex(columns=cpx_columns, fill_value=0)
    ctr_set = pd.DataFrame(data).reindex(columns=ctr_columns, fill_value=0)
    cpx_prediction = cpx_model.predict(cpx_set)[0].round(4)
    ctr_prediction = ctr_model.predict(ctr_set)[0].round(4)
    return [cpx_prediction, ctr_prediction]


def load_from_bucket(key):
    connection = S3Connection(aws_access_key_id=config.aws_access_key_id,
                              aws_secret_access_key=config.aws_secret_access_key,
                              is_secure=False)
    bucket = connection.get_bucket('cpx-prediction')
    local_file = '/tmp/' + key
    bucket.get_key(key).get_contents_to_filename(local_file)
    model = joblib.load(local_file)
    return model


if __name__ == '__main__':
    app.run()


# Example JSON payload
# [
#     {
#         "total_budget": 10000,
#         "start_month": 7,
#         "end_month": 12,
#         "days_before": 0,
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
