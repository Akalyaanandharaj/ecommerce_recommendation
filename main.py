from flask import Flask, jsonify
from flask import request
from algorithm import get_recommendations,df
import pickle

app = Flask(__name__)


@app.route('/product', methods=['GET', 'POST'])
def home():
    if (request.method == 'GET'):
        occasion = request.args.get('occasion')
        material = request.args.get('material')
        style = request.args.get('style')
        color = request.args.get('color')

        # Load TF-IDF Vectorizer
        with open('tfidf_vectorizer.pkl', 'rb') as f:
            tfidf_vectorizer = pickle.load(f)

        # Load Cosine Similarity Matrix
        with open('cosine_sim.pkl', 'rb') as f:
            cosine_sim = pickle.load(f)

        results = get_recommendations(occasion, material, style, color, df, tfidf_vectorizer)
        return jsonify(results)

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8080, debug=True)
