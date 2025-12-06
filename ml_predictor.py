import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter

class ExamPredictor:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(stop_words='english', min_df=1, ngram_range=(1, 2))
        
    def preprocess_text(self, text):
        """Clean and normalize text for vectorization"""
        text = str(text).lower()
        # Remove numbers and special chars, keep only letters and spaces
        text = re.sub(r'[^a-z\s]', ' ', text)
        # Remove multiple spaces
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def predict_questions(self, syllabus_topics, past_questions_text, num_predictions=20):
        """
        Predict questions based on similarity between syllabus topics and past question papers.
        
        Args:
            syllabus_topics (list): List of topic strings from syllabus
            past_questions_text (str): Concatenated text of all past question papers
            num_predictions (int): Number of top predictions to return
            
        Returns:
            list: List of dicts with 'topic', 'score', 'probability', 'question_type'
        """
        if not syllabus_topics:
            return []
            
        # Clean syllabus topics
        clean_topics = [self.preprocess_text(t) for t in syllabus_topics]
        
        # Split past questions into chunks (approx questions) to find correlation
        # We assume past_questions_text is a blob. We verify against the blob.
        # Ideally, we would split past questions into individual questions, but we can score against the whole corpus
        # A better approach for relevance:
        # Score(Topic) = CosineSimilarity(Vector(Topic), Vector(PastQuestions))
        
        # Prepare corpus: First document is past questions, rest are syllabus topics
        corpus = [self.preprocess_text(past_questions_text)] + clean_topics
        
        try:
            tfidf_matrix = self.vectorizer.fit_transform(corpus)
        except ValueError:
            # Handle empty vocabulary case
            return []
            
        # Calculate cosine similarity between each topic (rows 1:) and the past questions (row 0)
        # tfidf_matrix[0] is (1, n_features)
        # tfidf_matrix[1:] is (n_topics, n_features)
        
        # We want similarity of each topic TO the paper
        cosine_sim = cosine_similarity(tfidf_matrix[1:], tfidf_matrix[0:1]) # Result is (n_topics, 1)
        
        predictions = []
        scores = cosine_sim.flatten()
        
        for i, score in enumerate(scores):
            raw_topic = syllabus_topics[i]
            
            # Normalize score to a probability 0-100
            # Cosine sim is -1 to 1, but for text usually 0 to 1
            # We scale it: 0.1 similarity is actually quite good for short topic vs long doc
            # Let's calibrate: sim > 0.05 is relevant.
            
            # Simple calibration curve
            probability = min(score * 400, 99.9) # 0.25 sim -> 100% prob
            
            if probability > 10.0: # Filter low relevance
                predictions.append({
                    'topic': raw_topic,
                    'score': float(score),
                    'probability': round(probability, 1),
                    'question_type': 'Descriptive' if probability > 60 else 'Short Answer'
                })
                
        # Sort by probability descending
        predictions.sort(key=lambda x: x['probability'], reverse=True)
        
        return predictions[:num_predictions]

    def get_difficulty_distribution(self, text):
        """
        Estimate difficulty based on vocabulary complexity.
        Uses word length and presence of technical terms as a proxy.
        """
        words = self.preprocess_text(text).split()
        if not words:
            return {'Easy': 33, 'Medium': 33, 'Hard': 34}
            
        avg_len = sum(len(w) for w in words) / len(words)
        unique_ratio = len(set(words)) / len(words)
        
        # Heuristic rules
        # High unique ratio + long words = Hard
        difficulty_score = (avg_len * 5) + (unique_ratio * 40)
        
        # Normalize to distribution
        # Base: 40/40/20
        # If score > 60: Shift to Hard
        # If score < 40: Shift to Easy
        
        dist = {'Easy': 30, 'Medium': 40, 'Hard': 30}
        
        if difficulty_score > 65:
            dist = {'Easy': 20, 'Medium': 30, 'Hard': 50}
        elif difficulty_score < 45:
            dist = {'Easy': 50, 'Medium': 30, 'Hard': 20}
            
        return dist
