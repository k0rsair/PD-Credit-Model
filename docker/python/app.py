from flask import Flask, request, jsonify
import numpy as np
import joblib


app = Flask(__name__)


tfidf = joblib.load('tfidf_vectorizer-2.joblib')
model_tfidf = joblib.load('logistic_regression_model-2.joblib')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Получение данных из запроса
        data = request.get_json()
        # Предсказываем
        predictions = model_tfidf.predict(data)
        return predictions
    except Exception as e:
        return jsonify({'error': str(e)}), 400
@app.route('/test', methods=['get'])
def test():
    return '123'

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
