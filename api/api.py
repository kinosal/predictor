from flask import Flask, request, jsonify
from boto.s3.connection import S3Connection
from sklearn.externals import joblib
import traceback
import pandas as pd
import numpy as np

app = Flask(__name__)

@app.route('/', methods=['POST'])
def index():
    json_ = request.json
    query = pd.get_dummies(pd.DataFrame(json_))
    model = load_from_bucket('model.pkl')
    model_columns = load_from_bucket('model_columns.pkl')
    query = query.reindex(columns=model_columns, fill_value=0)
    prediction = list(model.predict(query))
    return jsonify({'prediction': str(prediction)})

def load_from_bucket(key):
    connection = S3Connection()
    bucket = connection.get_bucket('zappa-cpx')
    local_file = '/tmp/' + key
    bucket.get_key(key).get_contents_to_filename(local_file)
    model = joblib.load(local_file)
    return model

if __name__ == '__main__':
    app.run(host='0.0.0.0')
