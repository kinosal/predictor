from flask import Flask, request, jsonify
from boto.s3.connection import S3Connection
import pandas as pd
import joblib
import config


app = Flask(__name__)


@app.route('/ping', methods=['POST'])
def ping():
    return jsonify({'ping': 'successful'})


@app.route('/<metric>', methods=['POST'])
def process(metric):
    if metric not in ['impressions', 'clicks', 'cost_per_impression',
                      'cost_per_click']:
        return 'Metric "' + metric + '" not supported. Currently ' + \
               'supported metrics are "impressions", "clicks", ' + \
               '"cost_per_impression" and "cost_per_click".'
    data = request.json
    prediction = predict(data, metric)
    return jsonify({metric: prediction})


def predict(data, output):
    model = load_from_bucket(output + '_model.pkl')
    columns = load_from_bucket(output + '_columns.pkl')
    data = pd.DataFrame(data).reindex(columns=columns, fill_value=0)
    prediction = model.predict(data)[0].round(4)
    return prediction


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
#         "cost": 10000,
#         "start_month": 7,
#         "end_month": 12,
#         "start_week": 25,
#         "end_week": 50,
#         "days": 184,
#         "ticket_capacity": 10000,
#         "average_ticket_price": 50,
#         "facebook": 1,
#         "instagram": 1,
#         "google_search": 1,
#         "google_display": 0,
#         "facebook_likes": 1000,
#         "region_germany": 1,
#         "region_switzerland": 0,
#         "locality_single": 1,
#         "category_comedy": 0,
#         "category_concert": 0,
#         "category_conference": 1,
#         "category_ecommerce": 0,
#         "category_festival": 0,
#         "category_theatre": 0,
#         "shop_actnews": 0,
#         "shop_eventbrite": 0,
#         "shop_eventim": 0,
#         "shop_reservix": 0,
#         "shop_showare": 0,
#         "shop_stagelink": 1,
#         "tracking_yes": 1
#     }
# ]
