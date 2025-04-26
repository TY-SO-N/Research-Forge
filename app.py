from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import requests
import json
from urllib.parse import quote
import nltk
from nltk.corpus import stopwords
import spacy
import re
import numpy as np
import torch
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Initialize global variables and models
def initialize_models():
    # Download necessary NLTK data
    nltk.download('stopwords')
    nltk.download('punkt')
    
    # Load spaCy model
    try:
        nlp = spacy.load('en_core_web_sm')
    except:
        # If model not found, download it
        import subprocess
        subprocess.call(['python', '-m', 'spacy', 'download', 'en_core_web_sm'])
        nlp = spacy.load('en_core_web_sm')
    
    # Load sentence transformer model
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    return nlp, model

# Function to fetch journal data from OpenAlex API
def fetch_journal_data(query="computer science", limit=100):
    # Base URL for the OpenAlex API
    base_url = "https://api.openalex.org/works"
    
    # For searching venues specifically, we can use a filter
    encoded_query = quote(query)
    url = f"{base_url}?filter=default.search:{encoded_query}&per-page={limit}"
    
    # Add a user-agent header as some APIs require this
    headers = {
        "User-Agent": "Mozilla/5.0 Academic Research Project"
    }
    
    response = requests.get(url, headers=headers)
    
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Error: {response.status_code} - {response.text}")
        
        # Try an alternative approach
        venues_url = f"https://api.openalex.org/venues?search={encoded_query}&per-page={limit}"
        alt_response = requests.get(venues_url, headers=headers)
        
        if alt_response.status_code == 200:
            return alt_response.json()
        else:
            print(f"Alternative request failed: {alt_response.status_code}")
            return None

# Preprocess text
def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    
    # Remove special characters and numbers
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Tokenize and remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = nltk.word_tokenize(text)
    filtered_tokens = [w for w in tokens if w not in stop_words]
    
    # Join tokens back into a string
    return ' '.join(filtered_tokens)

# Extract key topics from text using spaCy
def extract_topics(text, nlp, top_n=5):
    doc = nlp(text)
    
    # Extract noun phrases as potential topics
    noun_phrases = [chunk.text for chunk in doc.noun_chunks]
    
    # Extract named entities
    entities = [ent.text for ent in doc.ents]
    
    # Combine and get unique topics
    all_topics = noun_phrases + entities
    # Get most frequent topics
    topic_freq = {}
    for topic in all_topics:
        if topic.lower() in topic_freq:
            topic_freq[topic.lower()] += 1
        else:
            topic_freq[topic.lower()] = 1
    
    # Sort by frequency and get top N
    sorted_topics = sorted(topic_freq.items(), key=lambda x: x[1], reverse=True)
    return [topic for topic, freq in sorted_topics[:top_n]]

# Generate embeddings for text
def generate_embeddings(text, model):
    return model.encode(text)

# Calculate similarity between paper and journals
def find_matching_journals(paper_embedding, journals_data, model, top_n=3):
    journal_scores = []
    
    # Extract unique journals from the data
    journals = {}
    
    # Process based on the structure of the API response
    if 'results' in journals_data:
        for item in journals_data['results']:
            # For works endpoint
            if 'host_venue' in item and item['host_venue']:
                journal_info = item['host_venue']
                if 'display_name' in journal_info and journal_info['display_name']:
                    journal_name = journal_info['display_name']
                    journal_desc = item['title'] if 'title' in item else ""
                    
                    if journal_name not in journals:
                        journals[journal_name] = journal_desc
            # For venues endpoint
            elif 'display_name' in item:
                journal_name = item['display_name']
                journal_desc = item.get('works_count', "") or item.get('description', "")
                
                if journal_name not in journals:
                    journals[journal_name] = str(journal_desc)
    
    # Calculate similarity scores
    for journal_name, journal_desc in journals.items():
        try:
            journal_embedding = model.encode(journal_desc)
            similarity = cosine_similarity([paper_embedding], [journal_embedding])[0][0]
            journal_scores.append((journal_name, similarity))
        except Exception as e:
            print(f"Error processing journal {journal_name}: {e}")
    
    # Sort by similarity score and return top N
    journal_scores.sort(key=lambda x: x[1], reverse=True)
    return journal_scores[:top_n]

# Initialize models at startup
nlp, model = initialize_models()

@app.route('/api/recommend', methods=['POST'])
def recommend_journals():
    try:
        # Get data from request
        data = request.json
        title = data.get('title', '')
        abstract = data.get('abstract', '')
        
        # Combine title and abstract
        paper_text = f"{title} {abstract}"
        
        # Preprocess text
        processed_text = preprocess_text(paper_text)
        
        # Extract key topics
        topics = extract_topics(paper_text, nlp)
        
        # Generate embeddings for the paper
        paper_embedding = generate_embeddings(processed_text, model)
        
        # Fetch journal data using the topics
        search_query = " ".join(topics[:3])  # Use top 3 topics for search
        journals_data = fetch_journal_data(search_query)
        
        if not journals_data:
            return jsonify({
                'error': 'Failed to fetch journal data',
                'recommendations': []
            }), 500
        
        # Find matching journals
        journal_matches = find_matching_journals(paper_embedding, journals_data, model)
        
        # Format recommendations
        recommendations = [
            {
                'journal': journal,
                'score': float(score),  # Convert numpy float to Python float for JSON
                'match_percentage': f"{float(score) * 100:.1f}%"
            }
            for journal, score in journal_matches
        ]
        
        return jsonify({
            'recommendations': recommendations,
            'paper': {
                'title': title,
                'abstract': abstract,
                'topics': topics
            }
        })
    
    except Exception as e:
        return jsonify({
            'error': str(e),
            'recommendations': []
        }), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy"})

@app.route('/', methods=['GET'])
def index():
    return jsonify({
        "message": "Journal Recommendation API is running",
        "endpoints": {
            "/api/recommend": "POST - Get journal recommendations based on paper title and abstract",
            "/api/health": "GET - Check API health status"
        }
    })    

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)