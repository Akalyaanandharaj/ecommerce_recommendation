from flask import Flask, jsonify
from flask import request
from algorithm import get_recommendations
from datacleaning import df
import pickle

app = Flask(__name__)


@app.route('/product', methods=['GET', 'POST'])
def home():
    if (request.method == 'GET'):
        occasion = request.args.get('occasion')
        style = request.args.get('style')
        color = request.args.get('color')
        details = request.args.get('details')

        # Load TF-IDF Vectorizer
        with open('tfidf_vectorizer.pkl', 'rb') as f:
            tfidf_vectorizer = pickle.load(f)

        with open('tfidf_matrix.pkl', 'rb') as f:
            tfidf_matrix = pickle.load(f)

        results = get_recommendations(occasion, details, style, color, df, tfidf_vectorizer, tfidf_matrix)
        return jsonify(results)

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8080, debug=True)
