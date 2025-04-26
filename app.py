from flask import Flask, request, jsonify, render_template
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
    
    # Load scientific text-specific model for better results
    model = SentenceTransformer('allenai/specter')
    
    return nlp, model

# Updated function to fetch data from OpenAlex API
def fetch_journal_data(query="computer science", limit=100):
    # Base URL for the OpenAlex API
    base_url = "https://api.openalex.org/works"
    
    # For searching venues specifically, we can use a filter
    encoded_query = quote(query)
    url = f"{base_url}?filter=default.search:{encoded_query}&per-page={limit}"
    
    print(f"Requesting URL: {url}")
    
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
        print("Trying alternative endpoint...")
        venues_url = f"https://api.openalex.org/venues?search={encoded_query}&per-page={limit}"
        print(f"Requesting URL: {venues_url}")
        alt_response = requests.get(venues_url, headers=headers)
        
        if alt_response.status_code == 200:
            return alt_response.json()
        else:
            print(f"Alternative request failed: {alt_response.status_code}")
            return None

# Improved text preprocessing focusing on technical terms
def preprocess_text(text):
    # Remove newlines and extra spaces
    text = re.sub(r'\s+', ' ', text)
    # Keep important technical characters like hyphens
    text = re.sub(r'[^\w\s\-]', '', text)
    # Convert to lowercase
    text = text.lower()
    
    # Keep important technical words that might be in stopwords
    custom_stopwords = set(stopwords.words('english')) - {'not', 'no', 'against', 'between', 'through', 
                                                         'above', 'below', 'up', 'down', 'under', 'over'}
    words = text.split()
    filtered_text = ' '.join([word for word in words if word not in custom_stopwords])
    return filtered_text

# Function to create optimized embeddings for similarity
def generate_optimized_embedding(text, model):
    """Generate embeddings optimized for similarity matching"""
    # Preprocess
    processed_text = preprocess_text(text)
    
    # Normalize the embedding for better cosine similarity
    embedding = model.encode(processed_text)
    normalized_embedding = embedding / np.linalg.norm(embedding)
    
    return normalized_embedding.reshape(1, -1)

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

# Enhanced paper processing with focus on technical terms
def process_paper_for_similarity(title, abstract, nlp, model):
    """Process paper with focus on maximizing similarity to relevant journals"""
    # Give more weight to title by repeating it
    combined_text = title + " " + title + " " + abstract
    
    # Extract and add technical keywords to boost similarity
    doc = nlp(title + " " + abstract)
    technical_terms = []
    
    # Extract technical noun phrases and entities
    for chunk in doc.noun_chunks:
        technical_terms.append(chunk.text)
    
    # Add extracted terms to boost similarity
    augmented_text = combined_text + " " + " ".join(technical_terms)
    
    # Generate embedding from this augmented text
    embedding = generate_optimized_embedding(augmented_text, model)
    
    # Extract topics
    topics = extract_topics(combined_text, nlp)
    
    return {
        "text": augmented_text,
        "embedding": embedding,
        "topics": topics
    }

# Match paper to journals focusing on cosine similarity
def find_matching_journals(paper_embedding, journals_data, model, top_n=5):
    journal_scores = []
    
    # Extract unique journals from the data
    journals = {}
    journal_urls = {}  # Add this to store URLs
    
    # Process based on the structure of the API response
    if 'results' in journals_data:
        for item in journals_data['results']:
            # For works endpoint
            if 'host_venue' in item and item['host_venue']:
                journal_info = item['host_venue']
                if 'display_name' in journal_info and journal_info['display_name']:
                    journal_name = journal_info['display_name']
                    
                    # Create comprehensive journal description
                    journal_desc = item['title'] if 'title' in item else ""
                    
                    # Store URL if available - FIX URL CONSTRUCTION HERE
                    if 'id' in item:
                        # Check if DOI is available first
                        if 'doi' in item and item['doi']:
                            journal_urls[journal_name] = item['doi']
                        # Otherwise use OpenAlex ID but ensure we don't duplicate the base URL
                        elif item['id'].startswith('https://'):
                            journal_urls[journal_name] = item['id']
                        else:
                            # If ID doesn't include https://, add the OpenAlex base URL
                            journal_urls[journal_name] = f"https://openalex.org/{item['id'].replace('https://openalex.org/', '')}"
                    
                    # Add abstract if available
                    if 'abstract_inverted_index' in item and isinstance(item['abstract_inverted_index'], dict):
                        # Reconstruct abstract from inverted index
                        abstract_words = []
                        for word, positions in item['abstract_inverted_index'].items():
                            for pos in positions:
                                while len(abstract_words) <= pos:
                                    abstract_words.append("")
                                abstract_words[pos] = word
                        journal_desc += " " + " ".join(abstract_words)
                    
                    if journal_name not in journals:
                        journals[journal_name] = journal_desc
            # For venues endpoint
            elif 'display_name' in item:
                journal_name = item['display_name']
                journal_desc = item.get('works_count', "") or item.get('description', "")
                
                # Store URL if available - FIX URL CONSTRUCTION HERE
                if 'id' in item:
                    if 'homepage_url' in item and item['homepage_url']:
                        journal_urls[journal_name] = item['homepage_url']
                    elif item['id'].startswith('https://'):
                        journal_urls[journal_name] = item['id']  
                    else:
                        # Ensure we don't duplicate the base URL
                        item_id = item['id'].replace('https://openalex.org/', '')
                        journal_urls[journal_name] = f"https://openalex.org/{item_id}"
                
                if journal_name not in journals:
                    journals[journal_name] = str(journal_desc)
    
    # Calculate similarity scores
    for journal_name, journal_desc in journals.items():
        try:
            # Generate optimized embedding for journal
            journal_embedding = generate_optimized_embedding(journal_desc, model)
            similarity = cosine_similarity(paper_embedding, journal_embedding)[0][0]
            url = journal_urls.get(journal_name, '')  # Get URL or empty string if not found
            journal_scores.append((journal_name, similarity, url))
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
        
        # Process paper with enhanced similarity approach
        paper_data = process_paper_for_similarity(title, abstract, nlp, model)
        
        # Fetch journal data using the extracted topics
        search_query = " ".join(paper_data["topics"][:3])  # Use top 3 topics for search
        journals_data = fetch_journal_data(search_query)
        
        if not journals_data:
            return jsonify({
                'error': 'Failed to fetch journal data',
                'recommendations': []
            }), 500
        
        # Find matching journals with optimized similarity approach
        journal_matches = find_matching_journals(paper_data["embedding"], journals_data, model)
        
        # Format recommendations
        recommendations = [
            {
                'journal': journal,
                'score': float(score),  # Convert numpy float to Python float for JSON
                'match_percentage': f"{float(score) * 100:.1f}%",
                'url': url or '#'  # Include URL, default to # if not available
            }
            for journal, score, url in journal_matches  # Now unpacking 3 values
        ]
        
        return jsonify({
            'recommendations': recommendations,
            'paper': {
                'title': title,
                'abstract': abstract,
                'topics': paper_data["topics"]
            }
        })
    
    except Exception as e:
        return jsonify({
            'error': str(e),
            'recommendations': []
        }), 500
    try:
        # Get data from request
        data = request.json
        title = data.get('title', '')
        abstract = data.get('abstract', '')
        
        # Process paper with enhanced similarity approach
        paper_data = process_paper_for_similarity(title, abstract, nlp, model)
        
        # Fetch journal data using the extracted topics
        search_query = " ".join(paper_data["topics"][:3])  # Use top 3 topics for search
        journals_data = fetch_journal_data(search_query)
        
        if not journals_data:
            return jsonify({
                'error': 'Failed to fetch journal data',
                'recommendations': []
            }), 500
        
        # Find matching journals with optimized similarity approach
        journal_matches = find_matching_journals(paper_data["embedding"], journals_data, model)
        
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
                'topics': paper_data["topics"]
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
    return render_template('index.html')
    
@app.route('/api', methods=['GET'])
def api_info():
    return jsonify({
        "message": "Journal Recommendation API is running",
        "endpoints": {
            "/api/recommend": "POST - Get journal recommendations based on paper title and abstract",
            "/api/health": "GET - Check API health status"
        }
    })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)