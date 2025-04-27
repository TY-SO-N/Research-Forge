from flask import Flask, request, jsonify, render_template, redirect, url_for, session, flash
from flask_cors import CORS
from flask_login import LoginManager, UserMixin, login_user, logout_user, current_user, login_required
from werkzeug.security import generate_password_hash, check_password_hash
import sqlite3
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
import os
from collections import defaultdict

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

app.secret_key = 'a9e3c1f4b6d982fc7a3b5d6cfe1234adf9087c2d3b6e9f0124c5d6a7b8c9e0f1'
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

# User model
class User(UserMixin):
    def __init__(self, id, username, email):
        self.id = id
        self.username = username
        self.email = email

# Database functions
def get_db_connection():
    conn = sqlite3.connect('journal_recommender.db')
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Create users table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE NOT NULL,
        email TEXT UNIQUE NOT NULL,
        password_hash TEXT NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    ''')
    
    # Create sessions table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS sessions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER NOT NULL,
        session_id TEXT UNIQUE NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (user_id) REFERENCES users (id)
    )
    ''')
    
    # Create ratings table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS ratings (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER NOT NULL,
        journal_name TEXT NOT NULL,
        rating INTEGER NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (user_id) REFERENCES users (id)
    )
    ''')

    cursor.execute('''
    CREATE TABLE IF NOT EXISTS paper_ratings (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER NOT NULL,
        paper_title TEXT NOT NULL,
        paper_abstract TEXT NOT NULL,
        rating INTEGER NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (user_id) REFERENCES users (id)
    )
    ''')
    
    conn.commit()
    conn.close()

def save_paper_rating_to_db(user_id, paper_title, paper_abstract, rating):
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Check if rating already exists
    cursor.execute(
        "SELECT id FROM paper_ratings WHERE user_id = ? AND paper_title = ?",
        (user_id, paper_title)
    )
    existing = cursor.fetchone()
    
    if existing:
        # Update existing rating
        cursor.execute(
            "UPDATE paper_ratings SET rating = ? WHERE id = ?",
            (rating, existing['id'])
        )
    else:
        # Insert new rating
        cursor.execute(
            "INSERT INTO paper_ratings (user_id, paper_title, paper_abstract, rating) VALUES (?, ?, ?, ?)",
            (user_id, paper_title, paper_abstract, rating)
        )
    
    conn.commit()
    conn.close()

def get_user_paper_ratings(user_id):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(
        "SELECT paper_title, paper_abstract, rating, created_at FROM paper_ratings WHERE user_id = ? ORDER BY created_at DESC",
        (user_id,)
    )
    ratings = cursor.fetchall()
    conn.close()
    
    return ratings

def get_specific_paper_rating(user_id, paper_title):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(
        "SELECT rating FROM paper_ratings WHERE user_id = ? AND paper_title = ?",
        (user_id, paper_title)
    )
    result = cursor.fetchone()
    conn.close()
    
    return result['rating'] if result else 0

@app.route('/api/rate-paper', methods=['POST'])
@login_required
def rate_paper():
    try:
        data = request.json
        paper_title = data.get('title')
        paper_abstract = data.get('abstract', '')  # Provide default empty abstract if missing
        rating = data.get('rating')
        
        if not paper_title or not rating or not isinstance(rating, int) or rating < 1 or rating > 5:
            return jsonify({'error': 'Invalid rating data'}), 400
            
        # Store the rating associated with the user
        save_paper_rating_to_db(current_user.id, paper_title, paper_abstract, rating)
        
        return jsonify({'success': True, 'message': 'Paper rating saved'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/get-paper-ratings', methods=['GET'])
@login_required
def get_paper_ratings():
    try:
        ratings = get_user_paper_ratings(current_user.id)
        
        formatted_ratings = []
        for rating in ratings:
            formatted_ratings.append({
                'title': rating['paper_title'],
                'abstract': rating['paper_abstract'],
                'rating': rating['rating'],
                'date': rating['created_at']
            })
        
        return jsonify({'ratings': formatted_ratings})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/get-paper-rating', methods=['GET'])
@login_required
def get_paper_rating():
    try:
        paper_title = request.args.get('title')
        if not paper_title:
            return jsonify({'error': 'Paper title is required'}), 400
        
        rating = get_specific_paper_rating(current_user.id, paper_title)
        
        return jsonify({'rating': rating})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/get-journal-rating', methods=['GET'])
@login_required
def get_journal_rating():
    try:
        journal_name = request.args.get('journal')
        if not journal_name:
            return jsonify({'error': 'Journal name is required'}), 400
        
        rating = get_specific_journal_rating(current_user.id, journal_name)
        
        return jsonify({'rating': rating})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def get_specific_journal_rating(user_id, journal_name):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(
        "SELECT rating FROM ratings WHERE user_id = ? AND journal_name = ?",
        (user_id, journal_name)
    )
    result = cursor.fetchone()
    conn.close()
    
    return result['rating'] if result else 0

# User loader function for Flask-Login
@login_manager.user_loader
def load_user(user_id):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT id, username, email FROM users WHERE id = ?", (user_id,))
    user_data = cursor.fetchone()
    conn.close()
    
    if user_data:
        return User(user_data['id'], user_data['username'], user_data['email'])
    return None

# User management functions
def create_user(username, email, password):
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Check if username or email already exists
        cursor.execute("SELECT id FROM users WHERE username = ? OR email = ?", (username, email))
        if cursor.fetchone():
            conn.close()
            return False, "Username or email already exists"
        
        # Create new user
        password_hash = generate_password_hash(password)
        cursor.execute(
            "INSERT INTO users (username, email, password_hash) VALUES (?, ?, ?)",
            (username, email, password_hash)
        )
        conn.commit()
        conn.close()
        return True, "User created successfully"
    except Exception as e:
        return False, str(e)

def verify_user(username, password):
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT id, password_hash FROM users WHERE username = ?", (username,))
        user = cursor.fetchone()
        conn.close()
        
        if user and check_password_hash(user['password_hash'], password):
            return True, user['id']
        return False, None
    except Exception as e:
        print(f"Error verifying user: {e}")
        return False, None

def create_session(user_id):
    import uuid
    session_id = str(uuid.uuid4())
    
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("INSERT INTO sessions (user_id, session_id) VALUES (?, ?)", (user_id, session_id))
    conn.commit()
    conn.close()
    
    return session_id

def verify_session(session_id):
    if not session_id:
        return None
    
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT user_id FROM sessions WHERE session_id = ?", (session_id,))
    result = cursor.fetchone()
    conn.close()
    
    return result['user_id'] if result else None

def get_user_ratings(user_id):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(
        "SELECT journal_name, rating, created_at FROM ratings WHERE user_id = ? ORDER BY created_at DESC",
        (user_id,)
    )
    ratings = cursor.fetchall()
    conn.close()
    
    return ratings

def save_rating_to_db(user_id, journal, rating):
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Check if rating already exists
    cursor.execute(
        "SELECT id FROM ratings WHERE user_id = ? AND journal_name = ?",
        (user_id, journal)
    )
    existing = cursor.fetchone()
    
    if existing:
        # Update existing rating
        cursor.execute(
            "UPDATE ratings SET rating = ? WHERE id = ?",
            (rating, existing['id'])
        )
    else:
        # Insert new rating
        cursor.execute(
            "INSERT INTO ratings (user_id, journal_name, rating) VALUES (?, ?, ?)",
            (user_id, journal, rating)
        )
    
    conn.commit()
    conn.close()

# Initialize global variables and models
# Journal ratings storage
journal_ratings = defaultdict(list)  # {journal_name: [ratings]}

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

# Function to load existing ratings from file
def load_ratings():
    global journal_ratings
    try:
        if os.path.exists('journal_ratings.json'):
            with open('journal_ratings.json', 'r') as f:
                ratings_data = json.load(f)
                journal_ratings = defaultdict(list, ratings_data)
            print(f"Loaded ratings for {len(journal_ratings)} journals")
    except Exception as e:
        print(f"Error loading ratings: {e}")
        journal_ratings = defaultdict(list)

# Function to save ratings to file
def save_ratings():
    try:
        with open('journal_ratings.json', 'w') as f:
            json.dump(journal_ratings, f)
    except Exception as e:
        print(f"Error saving ratings: {e}")

# Function to boost scores based on ratings
def apply_rating_boost(journal_matches, rating_weight=0.1):
    """Apply a boost to journal scores based on user ratings"""
    boosted_matches = []
    
    for journal, score, url in journal_matches:
        # Get average rating for this journal (if any)
        ratings = journal_ratings.get(journal, [])
        if ratings:
            avg_rating = sum(ratings) / len(ratings)
            # Normalize rating to 0-1 scale (from 1-5 scale)
            normalized_rating = (avg_rating - 1) / 4
            # Apply weighted boost
            boosted_score = score + (normalized_rating * rating_weight)
        else:
            boosted_score = score
            
        boosted_matches.append((journal, boosted_score, url))
    
    # Sort by boosted score
    boosted_matches.sort(key=lambda x: x[1], reverse=True)
    return boosted_matches

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
def extract_topics(text, nlp, top_n=10):
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

# Initialize database
init_db()

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form.get('username')
        email = request.form.get('email')
        password = request.form.get('password')
        confirm_password = request.form.get('confirm_password')
        
        # Basic validation
        if not username or not email or not password:
            flash('All fields are required', 'danger')
            return render_template('register.html')
        
        if password != confirm_password:
            flash('Passwords do not match', 'danger')
            return render_template('register.html')
        
        # Create user
        success, message = create_user(username, email, password)
        
        if success:
            flash('Registration successful! Please log in.', 'success')
            return redirect(url_for('login'))
        else:
            flash(f'Registration failed: {message}', 'danger')
            return render_template('register.html')
    
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        
        # Validate credentials
        success, user_id = verify_user(username, password)
        
        if success:
            # Create a User object
            conn = get_db_connection()
            cursor = conn.cursor()
            cursor.execute("SELECT id, username, email FROM users WHERE id = ?", (user_id,))
            user_data = cursor.fetchone()
            conn.close()
            
            user = User(user_data['id'], user_data['username'], user_data['email'])
            
            # Login user with Flask-Login
            login_user(user)
            
            # Also create session for legacy support
            session_id = create_session(user_id)
            session['session_id'] = session_id
            session['username'] = username
            
            flash('Login successful!', 'success')
            return redirect(url_for('index'))
        else:
            flash('Invalid username or password', 'danger')
    
    return render_template('login.html')

@app.route('/logout')
def logout():
    # Logout using Flask-Login
    logout_user()
    
    # Clear session data
    session.clear()
    
    flash('You have been logged out', 'info')
    return redirect(url_for('index'))

@app.route('/profile')
@login_required
def profile():
    # Get user information from Flask-Login
    user = current_user
    
    # Get user's ratings
    ratings = get_user_ratings(user.id)
    paper_ratings = get_user_paper_ratings(user.id)
    
    return render_template('profile.html', user=user, ratings=ratings, paper_ratings=paper_ratings)

@app.route('/api/rate-journal', methods=['POST'])
@login_required
def rate_journal():
    try:
        data = request.json
        journal = data.get('journal')
        rating = data.get('rating')
        
        if not journal or not rating or not isinstance(rating, int) or rating < 1 or rating > 5:
            return jsonify({'error': 'Invalid rating data'}), 400
            
        # Store the rating associated with the user
        save_rating_to_db(current_user.id, journal, rating)
        
        # Also store in the global ratings dictionary for backward compatibility
        journal_ratings[journal].append(rating)
        save_ratings()
        
        return jsonify({'success': True, 'message': 'Rating saved'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/recommend', methods=['POST'])
def recommend_journals():
    try:
        # Get data from request
        data = request.json
        title = data.get('title', '')
        abstract = data.get('abstract', '')
        include_ratings = data.get('includeRatings', False)
        
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
        
        # Apply rating boost if requested
        if include_ratings:
            journal_matches = apply_rating_boost(journal_matches)
        
        user_ratings = {}
        user_paper_rating = 0
        
        # Get user's ratings if authenticated
        if current_user.is_authenticated:
            # Get journal ratings
            conn = get_db_connection()
            cursor = conn.cursor()
            cursor.execute(
                "SELECT journal_name, rating FROM ratings WHERE user_id = ?",
                (current_user.id,)
            )
            for row in cursor.fetchall():
                user_ratings[row['journal_name']] = row['rating']
            
            # Get paper rating if exists
            cursor.execute(
                "SELECT rating FROM paper_ratings WHERE user_id = ? AND paper_title = ?",
                (current_user.id, title)
            )
            paper_rating = cursor.fetchone()
            if paper_rating:
                user_paper_rating = paper_rating['rating']
            
            conn.close()
        
        # Format recommendations
        recommendations = [
            {
                'journal': journal,
                'score': float(score),
                'match_percentage': f"{float(score) * 100:.1f}%",
                'url': url or '#',
                'user_rating': user_ratings.get(journal, 0)
            }
            for journal, score, url in journal_matches
        ]
        
        return jsonify({
            'recommendations': recommendations,
            'paper': {
                'title': title,
                'abstract': abstract,
                'topics': paper_data["topics"],
                'user_rating': user_paper_rating
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
            "/api/health": "GET - Check API health status",
            "/api/rate-journal": "POST - Save user ratings for journal recommendations",
            "/api/rate-paper": "POST - Save user ratings for papers",
            "/api/get-paper-ratings": "GET - Get all user's paper ratings",
            "/api/get-paper-rating": "GET - Get rating for a specific paper",
            "/api/get-journal-rating": "GET - Get rating for a specific journal"
        }
    })

@app.route('/rated-papers', methods=['GET'])
@login_required
def rated_papers():
    paper_ratings = get_user_paper_ratings(current_user.id)
    return render_template('rated_papers.html', paper_ratings=paper_ratings)

if __name__ == '__main__':
    load_ratings()
    nlp, model = initialize_models()  # Initialize models at startup
    app.run(debug=True, host='0.0.0.0', port=5000)