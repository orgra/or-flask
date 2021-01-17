# app.py
import json

from flask import Flask, request, jsonify
from flask_restful import reqparse
import numpy as np
from joblib import load
import sys


app = Flask(__name__)
model = load('assets/rfc_or.joblib')

parser = reqparse.RequestParser()


@app.route('/predict_single/', methods=['GET'])
def respond():
    data = []
    try:
        for feature in request.args:
            data.append(int(request.args.get(feature)))

        pred = model.predict(np.array(data).reshape(1, -1))

        result = "predicted salary <50k" if pred[0] == 0 else "predicted salary >50k"
        return result

    except Exception as e:
        print("Unexpected error:", sys.exc_info()[0])
        response = str(e)
        return jsonify(response)



@app.route("/predict_mul/", methods=["POST"])
def mul_respond():
    request_json = request.get_json()
    request_dict = json.loads(request_json)

    xin = np.array(list(request_dict.values())).T

    prediction = model.predict(xin)
    return json.dumps(prediction.tolist())

# A welcome message to test our server
@app.route('/')
def index():
    return "<h1>Welcome to Ors server !!</h1>"


if __name__ == '__main__':
    # Threaded option to enable multiple instances for multiple user access support
    app.run(threaded=True, port=5000)
