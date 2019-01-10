from flask import Flask, request, jsonify
from boto.s3.connection import S3Connection
from sklearn.externals import joblib
import pandas as pd

app = Flask(__name__)

@app.route('/cpi', methods=['POST'])
def cpi():
    data = request.json
    prediction = predict(data, 'cpi')
    return jsonify({'cpi': prediction})

@app.route('/cpc', methods=['POST'])
def cpc():
    data = request.json
    prediction = predict(data, 'cpc')
    return jsonify({'cpc': prediction})

def predict(data, cpx):
    model = load_from_bucket(cpx + '_model.pkl')
    columns = load_from_bucket(cpx + '_columns.pkl')
    set = pd.DataFrame(data).reindex(columns = columns, fill_value = 0)
    prediction = model.predict(set)[0].round(4)
    return prediction

def load_from_bucket(key):
    connection = S3Connection()
    bucket = connection.get_bucket('zappa-cpx')
    local_file = '/tmp/' + key
    bucket.get_key(key).get_contents_to_filename(local_file)
    model = joblib.load(local_file)
    return model

if __name__ == '__main__':
    app.run(host='0.0.0.0')

# Example JSON payload
# [
#     {
#         "start_date": 190701,
#         "end_date": 191231,
#         "total_budget": 10000,
#         "start_month": 7,
#         "end_month": 12,
#         "campaign_phase": 1,
#         "days": 184,
#         "facebook": 1,
#         "instagram": 1,
#         "google_search": 1,
#         "google_display": 0,
#         "twitter": 0,
#         "region_germany": 1,
#         "locality_single": 1,
#         "category_music": 1,
#         "shop_eventbrite": 1
#     }
# ]
