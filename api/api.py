import datetime
import math
from flask import Flask, request, jsonify, render_template
import boto3
import pandas as pd
import joblib


app = Flask(__name__)


@app.route('/ping', methods=['GET', 'POST'])
def ping():
    return 'Server is here'


@app.route('/', methods=['GET', 'POST'])
def root():
    if request.method == 'POST':
        data = {}
        data['cost'] = int(request.form['budget'])
        start_month, start_day, start_year = \
            [int(i) for i in request.form['start_date'].split('/')]
        end_month, end_day, end_year = \
            [int(i) for i in request.form['end_date'].split('/')]
        data['start_month'] = start_month
        data['end_month'] = end_month
        start_date = datetime.date(start_year, start_month, start_day)
        end_date = datetime.date(end_year, end_month, end_day)
        data['start_week'] = start_date.isocalendar()[1]
        data['end_week'] = end_date.isocalendar()[1]
        data['days'] = (end_date - start_date).days
        data['ticket_capacity'] = int(request.form['capacity'])
        data['average_ticket_price'] = int(request.form['price'])
        for channel in ['facebook', 'instagram',
                        'google_search', 'google_display']:
            try:
                if request.form[channel] == 'on':
                    data[channel] = 1
            except KeyError:
                data[channel] = 0
        data['facebook_likes'] = int(request.form['likes'])
        data['region_' + request.form['region'].lower()] = 1
        # try:
        #     if request.form['targets'] == 'on':
        #         data['locality_single'] = 0
        # except KeyError:
        #     data['locality_single'] = 1
        data['category_' + request.form['category'].lower()] = 1
        data['shop_' + request.form['shop'].lower()] = 1
        predictions = predict_metrics(data)
        impressions_low = int(round_down(predictions['impressions'] * 0.8, -4))
        impressions_high = int(round_up(predictions['impressions'] * 1.2, -4))
        clicks_low = int(round_down(predictions['clicks'] * 0.8, -2))
        clicks_high = int(round_up(predictions['clicks'] * 1.2, -2))
        purchases_low = int(round_down(predictions['purchases'] * 0.8, -1))
        purchases_high = int(round_up(predictions['purchases'] * 1.2, -1))
        return render_template('index.html', scroll='results',
                               impressions_low=f'{impressions_low:,}',
                               impressions_high=f'{impressions_high:,}',
                               clicks_low=f'{clicks_low:,}',
                               clicks_high=f'{clicks_high:,}',
                               purchases_low=f'{purchases_low:,}',
                               purchases_high=f'{purchases_high:,}')

    return render_template('index.html')


@app.route('/<metric>', methods=['POST'])
def metric_prediction(metric):
    if metric not in ['impressions', 'clicks', 'purchases',
                      'cost_per_impression', 'cost_per_click',
                      'cost_per_purchase']:
        return 'Metric "' + metric + '" not supported.'
    data = request.json
    data = format_categoricals(data)
    prediction = int(predict([data], metric))
    return jsonify({metric: prediction})


@app.route('/campaign', methods=['POST'])
def campaign_prediction():
    data = request.json
    data = format_categoricals(data)
    predictions = predict_metrics(data)
    return jsonify(predictions)


def round_up(x, decimals=0):
    multiplier = 10 ** decimals
    return math.ceil(x * multiplier) / multiplier


def round_down(x, decimals=0):
    multiplier = 10 ** decimals
    return math.floor(x * multiplier) / multiplier


def format_categoricals(data):
    categoricals = ['category', 'region', 'shop', 'locality']
    for cat in categoricals:
        if cat in data:
            data[cat + '_' + data[cat].lower()] = 1
            del data[cat]
    return data


def predict_metrics(data):
    predictions = {}
    for metric in ['impressions', 'clicks', 'purchases']:
        direct = int(predict([data], metric))
        cpx = int(data['cost'] / predict([data], 'cost_per_' + metric[0:-1]))
        trans = int(predict([{'direct': direct, 'cpx': cpx}],
                            metric + '_transfer'))
        predictions[metric] = trans
    return predictions


def predict(data, output):
    model = load_from_bucket(output + '_model.pkl')
    columns = load_from_bucket(output + '_columns.pkl')
    data = pd.DataFrame(data).reindex(columns=columns, fill_value=0)
    prediction = model.predict(data)[0]
    return prediction


def load_from_bucket(key):
    local_file = '/tmp/' + key
    s3_resource = boto3.resource('s3')
    bucket = s3_resource.Bucket('cpx-prediction')
    bucket.download_file(key, local_file)
    model = joblib.load(local_file)
    return model


if __name__ == '__main__':
    app.run()
