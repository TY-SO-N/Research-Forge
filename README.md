# 🔬 Research-Forge: AI-Powered Academic Journal Recommender

![Python](https://img.shields.io/badge/Python-3.9+-blue?style=for-the-badge&logo=python)
![Flask](https://img.shields.io/badge/Flask-REST_API-000000?style=for-the-badge&logo=flask)
![SQLite](https://img.shields.io/badge/SQLite-Database-003B57?style=for-the-badge&logo=sqlite)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Machine_Learning-F7931E?style=for-the-badge&logo=scikit-learn)
![Sentence-Transformers](https://img.shields.io/badge/HuggingFace-Sentence_Transformers-ffcc00?style=for-the-badge&logo=huggingface)

**Research-Forge** is an intelligent academic publishing assistant designed to bridge the gap between researchers and their ideal publication venues. By leveraging state-of-the-art NLP models (AllenAI's SPECTER) and a feedback-driven recommendation pipeline, this platform helps scientists find the most relevant journals for their abstracts in milliseconds.

---

## 🎯 The Problem & The Solution

**The Problem:** Researchers often spend weeks trying to find the appropriate journal for their niche work. Searching for keywords on publisher websites is tedious, inaccurate, and doesn't account for semantic meaning.

**The Solution:** Research-Forge ingests a user's abstract, generates dense vector embeddings using specialized scientific NLP models, and calculates the cosine similarity against thousands of journals fetched via the OpenAlex API. Over time, user ratings dynamically adjust and boost recommendation scores.

---

## ✨ Engineering Highlights

### 1. 🧠 Semantic Search Pipeline (NLP)
Instead of relying on basic keyword matching, this platform utilizes deep semantic understanding:
- **Text Preprocessing:** Leverages `spaCy` and `NLTK` for advanced tokenization, stop-word removal, and scientific term preservation.
- **Dense Vector Embeddings:** Uses the `allenai/specter` model from the `sentence-transformers` library. SPECTER is specifically pre-trained on scientific citations, making it incredibly accurate for academic text.
- **Cosine Similarity Ranking:** Utilizes `scikit-learn` to compute the cosine similarity between the researcher's abstract embedding and the journal description embeddings in $O(1)$ batch operations.

### 2. 🔄 Feedback-Driven Reinforcement Loop
Recommendations aren't static. Research-Forge implements a continuous feedback loop:
- **User Ratings:** Authenticated users can rate the relevance of suggested journals on a 1-5 scale.
- **Score Boosting:** The ranking algorithm applies a weighted boost to journal scores based on historical user ratings, lifting the relevance of highly-rated journals by up to ~17% for subsequent queries.

### 3. 🔐 Secure & Stateful Architecture
- **Flask REST Architecture:** A robust backend built with Flask, utilizing `Flask-Login` for session management and `werkzeug.security` for password hashing.
- **Database Schema:** A normalized `SQLite` database tracks user accounts, active sessions, and historical journal/paper ratings.
- **Live Data Integration:** Interfaces directly with the **OpenAlex REST API** to dynamically fetch up-to-date metadata for thousands of academic venues.

---

## 🛠️ Tech Stack

- **Backend:** Python, Flask, Flask-CORS, Flask-Login
- **Machine Learning / NLP:** `sentence-transformers` (PyTorch), `scikit-learn`, `spaCy` (`en_core_web_sm`), `NLTK`
- **Database:** SQLite (`journal_recommender.db`)
- **External APIs:** OpenAlex API

---

## 🚀 Local Development Setup

Follow these instructions to run the NLP recommendation engine locally.

### Prerequisites
- Python 3.9+
- Pip (Python Package Manager)

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/TY-SO-N/Research-Forge.git
   cd Research-Forge
   ```

2. **Create a virtual environment (Recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Download NLP Models:**
   The application will automatically download the NLTK stop words and `allenai/specter` models on first run. To ensure `spaCy` is ready, run:
   ```bash
   python -m spacy download en_core_web_sm
   ```

5. **Start the Flask Server:**
   ```bash
   python app.py
   ```
   The application will start on `http://127.0.0.1:5000`.

---

## 📁 Repository Structure

- `app.py`: The core Flask application containing route handlers, NLP preprocessing, OpenAlex API integration, and the Cosine Similarity ranking engine.
- `check_db.py`: A utility script for debugging and validating the SQLite database schema.
- `journal_recommender.db`: The SQLite database storing users, sessions, and rating matrices.
- `templates/`: Jinja2 HTML templates for the frontend UI.
- `Research_Forge.ipynb`: Jupyter Notebook containing experimental data science workflows and model evaluations before they were integrated into the Flask backend.

---

## 🤝 Contributing
Contributions are always welcome. Whether it's adding new NLP models, improving the frontend UI, or optimizing the database schema, feel free to open a Pull Request!
