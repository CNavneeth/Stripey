from flask import Flask, send_from_directory, request, jsonify
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
import os

app = Flask(__name__)

# Load your dataset
df = pd.read_csv('data/dataset.csv')

# Function to recommend products based on category
def recommend_products(category, num_recommendations=3):
    # Filter products by category
    category_products = df[df['CATEGORY'] == category]
    
    if category_products.empty:
        return []

    # Features: Price, Rating (standardized), and Quantity
    features = category_products[['PRICE', 'RATING', 'QUANTITY']].copy()
    
    # Standardize features
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)

    # Compute similarity
    similarity_matrix = cosine_similarity(scaled_features)

    # Get similarity scores
    mean_similarity = similarity_matrix.mean(axis=0)
    similar_indices = mean_similarity.argsort()[-num_recommendations:][::-1]

    # Return recommended products
    recommendations = category_products.iloc[similar_indices]
    return recommendations[['PRODUCT', 'PRICE', 'RATING']].to_dict(orient='records')

@app.route('/')
def serve_static_html():
    return send_from_directory('static', 'recommend.html')

@app.route('/recommend', methods=['GET'])
def recommend():
    category = request.args.get('category', default='ELECTRONICS', type=str)
    recommendations = recommend_products(category)
    return jsonify(recommendations)

if __name__ == '__main__':
    app.run(debug=True)
