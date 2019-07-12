import datetime
from flask import Flask, request, jsonify, render_template
from boto.s3.connection import S3Connection
import pandas as pd
import joblib
import config


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
        # data['facebook_likes'] = int(request.form['followers'])
        data['region_' + request.form['region'].lower()] = 1
        try:
            if request.form['tour'] == 'on':
                data['locality_single'] = 0
        except KeyError:
            data['locality_single'] = 1
        data['category_' + request.form['category'].lower()] = 1
        data['shop_' + request.form['shop'].lower()] = 1
        # try:
        #     if request.form['tracking'] == 'on':
        #         data['tracking'] = 1
        # except KeyError:
        #     data['tracking'] = 0
        impressions = (predict([data], 'impressions') + data['cost'] /
                       predict([data], 'cost_per_impression')) / 2
        clicks = (predict([data], 'clicks') + data['cost'] /
                  predict([data], 'cost_per_click')) / 2
        purchases = (predict([data], 'purchases') + data['cost'] /
                     predict([data], 'cost_per_purchase')) / 2
        impressions_lower = f'{int((impressions * 0.7).round(-4)):,}'
        impressions_higher = f'{int((impressions * 1.2).round(-4)):,}'
        clicks_lower = f'{int((clicks * 0.7).round(-2)):,}'
        clicks_higher = f'{int((clicks * 1.2).round(-2)):,}'
        purchases_lower = f'{int((purchases * 0.7).round(-1)):,}'
        purchases_higher = f'{int((purchases * 1.2).round(-1)):,}'
        return render_template('index.html', scroll='results',
                               impressions_lower=impressions_lower,
                               impressions_higher=impressions_higher,
                               clicks_lower=clicks_lower,
                               clicks_higher=clicks_higher,
                               purchases_lower=purchases_lower,
                               purchases_higher=purchases_higher)

    return render_template('index.html')


@app.route('/<metric>', methods=['POST'])
def process(metric):
    if metric not in ['impressions', 'clicks', 'purchases',
                      'cost_per_impression', 'cost_per_click',
                      'cost_per_purchase']:
        return 'Metric "' + metric + '" not supported.'
    data = request.json
    print(data)
    categoricals = ['category', 'region', 'shop', 'locality']
    for cat in categoricals:
        if cat in data:
            data[cat + '_' + data[cat].lower()] = 1
            del data[cat]
    prediction = predict([data], metric)
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
