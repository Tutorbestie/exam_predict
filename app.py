from flask import Flask, render_template, request, jsonify, session
from flask_cors import CORS
from flask_session import Session
from werkzeug.utils import secure_filename
import os
import PyPDF2
import docx
import json
from datetime import datetime
import numpy as np
import re
from collections import Counter
import hashlib
import uuid
from pathlib import Path
import sqlite3
import logging
from typing import Dict, List, Tuple, Optional
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import hashlib
import hmac
import time
import threading
from queue import Queue

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Session configuration
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'dev-key-change-in-production')
app.config['SESSION_TYPE'] = 'filesystem'
app.config['PERMANENT_SESSION_LIFETIME'] = 3600  # 1 hour
Session(app)

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'pdf', 'docx', 'txt'}
MAX_FILE_SIZE = 16 * 1024 * 1024  # 16MB max file size
DATABASE = 'exampredict.db'
MODEL_PATH = 'models/exam_predictor_model.pkl'

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE

# Create necessary folders
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs('analysis', exist_ok=True)
os.makedirs('predictions', exist_ok=True)
os.makedirs('models', exist_ok=True)

# Database initialization
def init_db():
    """Initialize the SQLite database with required tables"""
    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()
    
    # Users table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT UNIQUE NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            last_activity TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # Uploads table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS uploads (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            filename TEXT NOT NULL,
            original_filename TEXT NOT NULL,
            file_type TEXT NOT NULL,
            upload_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            file_hash TEXT,
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
    ''')
    
    # Analysis results table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS analyses (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            syllabus_id INTEGER,
            results_json TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users (id),
            FOREIGN KEY (syllabus_id) REFERENCES uploads (id)
        )
    ''')
    
    # Predictions table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            analysis_id INTEGER,
            prediction_results TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users (id),
            FOREIGN KEY (analysis_id) REFERENCES analyses (id)
        )
    ''')
    
    conn.commit()
    conn.close()

def get_user_id():
    """Get or create user ID from session"""
    if 'user_id' not in session:
        # Create a new user record
        conn = sqlite3.connect(DATABASE)
        cursor = conn.cursor()
        
        # Use session ID as unique identifier
        session_id = session.sid if hasattr(session, 'sid') else str(uuid.uuid4())
        
        try:
            cursor.execute('INSERT INTO users (session_id) VALUES (?)', (session_id,))
            user_id = cursor.lastrowid
            conn.commit()
            session['user_id'] = user_id
        except sqlite3.IntegrityError:
            # User already exists (race condition)
            cursor.execute('SELECT id FROM users WHERE session_id = ?', (session_id,))
            user_id = cursor.fetchone()[0]
            session['user_id'] = user_id
        finally:
            conn.close()
    
    return session['user_id']

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def validate_and_save_file(file, file_type='general'):
    """Securely saves uploaded file with validation"""
    if file.filename == '':
        raise ValueError("No file selected")
    
    if not allowed_file(file.filename):
        raise ValueError("Invalid file type")
    
    # Create a unique filename to prevent collisions and path traversal
    unique_filename = f"{file_type}_{uuid.uuid4().hex[:12]}_{secure_filename(file.filename)}"
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
    
    # Double-check file extension after sanitization
    if not allowed_file(filepath):
        raise ValueError("Invalid file type after sanitization")
    
    file.save(filepath)
    
    # Compute file hash for integrity checking
    with open(filepath, 'rb') as f:
        file_hash = hashlib.sha256(f.read()).hexdigest()
    
    # Record upload in database
    user_id = get_user_id()
    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()
    cursor.execute(
        'INSERT INTO uploads (user_id, filename, original_filename, file_type, file_hash) VALUES (?, ?, ?, ?, ?)',
        (user_id, unique_filename, file.filename, file_type, file_hash)
    )
    upload_id = cursor.lastrowid
    conn.commit()
    conn.close()
    
    return unique_filename, upload_id

class AdvancedExamPredictor:
    """
    Advanced ML model for predicting exam questions based on syllabus and historical papers.
    Includes features for exam type detection, difficulty assessment, and pattern recognition.
    """
    
    def __init__(self):
        self.exam_metadata = {}
        self.vectorizer = None
        self.is_trained = False
        
    def extract_exam_metadata(self, text: str) -> Dict:
        """Extract metadata about the exam from the text"""
        metadata = {
            'exam_type': self._detect_exam_type(text),
            'conducting_authority': self._detect_conducting_authority(text),
            'estimated_difficulty': self._estimate_difficulty_level(text),
            'subject_area': self._detect_subject_area(text),
            'question_types': self._detect_question_types(text),
            'exam_level': self._detect_exam_level(text),
            'pattern_consistency': self._assess_pattern_consistency(text)
        }
        return metadata
    
    def _detect_exam_type(self, text: str) -> str:
        """Detect the type of exam (objective, descriptive, mixed)"""
        text_lower = text.lower()
        
        # Keywords for objective type exams
        objective_keywords = [
            'mcq', 'multiple choice', 'objective', 'select the correct', 
            'choose the right', 'one mark', 'single option', 'omr', 'aptitude'
        ]
        
        # Keywords for descriptive exams
        descriptive_keywords = [
            'long answer', 'descriptive', 'detailed', 'explain', 'discuss',
            'write notes', 'essay', 'short note', 'elaborate', 'derivation',
            'proof', 'solve', 'calculate', 'numerical'
        ]
        
        objective_count = sum(1 for kw in objective_keywords if kw in text_lower)
        descriptive_count = sum(1 for kw in descriptive_keywords if kw in text_lower)
        
        if objective_count > descriptive_count:
            return "objective"
        elif descriptive_count > objective_count:
            return "descriptive"
        else:
            return "mixed"
    
    def _detect_conducting_authority(self, text: str) -> str:
        """Detect the conducting authority of the exam"""
        text_upper = text.upper()
        
        authorities = {
            'UPSC': ['upsc', 'civil services', 'ias', 'irs', 'ifs'],
            'SSC': ['ssc', 'staff selection commission'],
            'GATE': ['gate', 'graduate aptitude test'],
            'IES': ['ies', 'engineering services', 'ese'],
            'State PSC': ['psc', 'public service commission', 'state psc'],
            'University': ['university', 'college', 'semester', 'btech', 'mtech', 'b.e.', 'm.e.'],
            'Banking': ['ibps', 'sbi', 'rrb', 'bank po', 'clerk'],
            'Railway': ['railway', 'rrb', 'group d', 'ntpc', 'jee main']
        }
        
        for authority, keywords in authorities.items():
            if any(keyword in text_lower for keyword in keywords):
                return authority
                
        return "Other"
    
    def _estimate_difficulty_level(self, text: str) -> str:
        """Estimate the difficulty level of the exam"""
        # Count technical terms and advanced concepts
        technical_terms = [
            'theorem', 'proof', 'derive', 'calculate', 'analyze', 'evaluate',
            'synthesis', 'application', 'advanced', 'complex', 'algorithm',
            'formula', 'equation', 'method', 'principle', 'conceptual',
            'critical thinking', 'evaluation', 'interpretation'
        ]
        
        text_lower = text.lower()
        technical_count = sum(1 for term in technical_terms if term in text_lower)
        word_count = len(text.split())
        
        if word_count == 0:
            return "medium"
            
        tech_density = technical_count / word_count
        
        if tech_density > 0.05:
            return "Advanced"
        elif tech_density > 0.02:
            return "Intermediate"
        else:
            return "Beginner"
    
    def _detect_subject_area(self, text: str) -> str:
        """Detect the subject area of the exam"""
        subject_keywords = {
            'Engineering': [
                'engineering', 'mechanical', 'electrical', 'civil', 'computer science',
                'electronics', 'chemical', 'aerospace', 'automobile', 'biomedical',
                'instrumentation', 'production', 'industrial'
            ],
            'Medical': [
                'medicine', 'medical', 'physiology', 'anatomy', 'pathology', 'pharmacology',
                'biochemistry', 'microbiology', 'forensic', 'community medicine'
            ],
            'Management': [
                'mba', 'management', 'business', 'marketing', 'finance', 'hr', 'operations',
                'strategy', 'entrepreneurship', 'accounting', 'economics'
            ],
            'Law': ['law', 'legal', 'jurisprudence', 'constitutional', 'civil law', 'criminal law'],
            'Arts': ['literature', 'history', 'philosophy', 'sociology', 'psychology', 'economics'],
            'Science': ['physics', 'chemistry', 'mathematics', 'biology', 'zoology', 'botany'],
            'Competitive': ['aptitude', 'reasoning', 'quantitative', 'verbal', 'non-verbal']
        }
        
        text_lower = text.lower()
        
        for subject, keywords in subject_keywords.items():
            if any(keyword in text_lower for keyword in keywords):
                return subject
                
        return "General"
    
    def _detect_question_types(self, text: str) -> List[str]:
        """Detect the types of questions present in the paper"""
        text_lower = text.lower()
        types = set()
        
        if any(kw in text_lower for kw in ['mcq', 'objective', 'multiple choice', 'aptitude']):
            types.add('MCQ')
        if any(kw in text_lower for kw in ['short answer', 'brief', '2 marks', '5 marks', 'answer in brief']):
            types.add('Short Answer')
        if any(kw in text_lower for kw in ['long answer', 'essay', '10 marks', '15 marks', 'detailed', 'notes on']):
            types.add('Long Answer')
        if any(kw in text_lower for kw in ['numerical', 'calculation', 'problem', 'solve', 'find value', 'equation']):
            types.add('Numerical')
        if any(kw in text_lower for kw in ['true/false', 't/f', 'yes/no']):
            types.add('True/False')
        if any(kw in text_lower for kw in ['fill in blanks', 'fill in the blanks', 'blanks']):
            types.add('Fill in Blanks')
        if any(kw in text_lower for kw in ['match the following', 'matching']):
            types.add('Matching')
        
        return list(types) if types else ['Mixed']
    
    def _detect_exam_level(self, text: str) -> str:
        """Detect the level of the exam (high school, college, competitive)"""
        text_lower = text.lower()
        
        high_school_indicators = [
            'high school', 'secondary', '10th', '12th', 'plus two', 'intermediate',
            'ssc', 'hsc', 'matric', 'higher secondary'
        ]
        
        college_indicators = [
            'btech', 'mtech', 'bsc', 'msc', 'b.e.', 'm.e.', 'undergraduate', 'postgraduate',
            'semester', 'college', 'university', 'degree', 'diploma'
        ]
        
        competitive_indicators = [
            'competitive', 'gate', 'ies', 'ias', 'upsc', 'ssc', 'bank', 'railway', 'psu',
            'cat', 'jee', 'neet', 'bitsat', 'comedk'
        ]
        
        hs_count = sum(1 for indicator in high_school_indicators if indicator in text_lower)
        coll_count = sum(1 for indicator in college_indicators if indicator in text_lower)
        comp_count = sum(1 for indicator in competitive_indicators if indicator in text_lower)
        
        if comp_count >= hs_count and comp_count >= coll_count:
            return "Competitive"
        elif coll_count >= hs_count:
            return "College Level"
        else:
            return "High School"
    
    def _assess_pattern_consistency(self, text: str) -> float:
        """Assess how consistent the question pattern is"""
        # Look for repeated patterns in question formatting
        question_patterns = re.findall(r'(?:Q|Question)\s*\d+[\.\):]?', text, re.IGNORECASE)
        total_chars = len(text)
        
        if total_chars == 0:
            return 0.0
            
        # Calculate density of question markers
        pattern_density = len(question_patterns) / (total_chars / 1000)  # Per thousand chars
        
        # More consistent if pattern appears regularly
        return min(pattern_density * 10, 1.0)  # Normalize to 0-1 scale
    
    def predict_questions(self, syllabus_topics: List[str], past_questions_text: str, 
                         metadata: Optional[Dict] = None, num_predictions: int = 20) -> List[Dict]:
        """
        Predict questions based on syllabus topics and past questions.
        Enhanced to consider exam type, difficulty, and other metadata.
        """
        if not syllabus_topics:
            return []
        
        # Use the TF-IDF approach from ml_predictor but enhanced
        def preprocess_text(text):
            import re
            text = str(text).lower()
            text = re.sub(r'[^a-z0-9\s]', ' ', text)
            text = re.sub(r'\s+', ' ', text).strip()
            return text
        
        clean_topics = [preprocess_text(t) for t in syllabus_topics]
        
        # If we have past questions, use them to find similar topics
        if past_questions_text:
            corpus = [preprocess_text(past_questions_text)] + clean_topics
        else:
            # If no past questions, just use topics
            corpus = clean_topics
        
        try:
            if self.vectorizer is None:
                self.vectorizer = TfidfVectorizer(
                    stop_words='english',
                    ngram_range=(1, 2),
                    max_features=5000
                )
            
            tfidf_matrix = self.vectorizer.fit_transform(corpus)
            
            if past_questions_text:
                # Calculate similarity between each topic and the past questions
                cosine_sim = cosine_similarity(tfidf_matrix[1:], tfidf_matrix[0:1])  # All topics vs questions
                scores = cosine_sim.flatten()
            else:
                # If no past questions, assign equal scores
                scores = [0.5] * len(clean_topics)
        
        except ValueError:
            # Handle empty vocabulary case
            scores = [0.1] * len(clean_topics)
        
        # Create predictions
        predictions = []
        for i, topic in enumerate(syllabus_topics):
            if i < len(scores):
                score = scores[i]
                # Normalize score to probability (0-100)
                probability = min(score * 100, 99.9)
            else:
                probability = 50.0  # Default if we couldn't compute a score
            
            # Determine question type based on metadata
            question_type = "Mixed"
            if metadata and metadata.get('question_types'):
                q_types = metadata['question_types']
                if 'MCQ' in q_types:
                    question_type = "MCQ (1 mark)"
                elif 'Numerical' in q_types:
                    question_type = "Numerical Problem"
                elif 'Long Answer' in q_types:
                    question_type = "Long Answer"
                elif 'Short Answer' in q_types:
                    question_type = "Short Answer"
            
            difficulty = metadata.get('estimated_difficulty', 'Medium') if metadata else 'Medium'
            
            predictions.append({
                'topic': topic,
                'score': float(score) if i < len(scores) else 0.5,
                'probability': round(probability, 1),
                'question_type': question_type,
                'difficulty': difficulty,
                'exam_type': metadata.get('exam_type', 'Mixed') if metadata else 'Mixed',
                'conducting_authority': metadata.get('conducting_authority', 'Unknown') if metadata else 'Unknown'
            })
        
        # Sort by probability
        predictions.sort(key=lambda x: x['probability'], reverse=True)
        return predictions[:num_predictions]

# Initialize the predictor
predictor = AdvancedExamPredictor()

# serve from templates folder
@app.route('/')
def index():
    return render_template('index.html')

def get_analysis_data(user_id):
    """Get analysis data for the current user from DB"""
    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()
    
    # Get latest analysis for this user
    cursor.execute('''
        SELECT results_json FROM analyses 
        WHERE user_id = ? 
        ORDER BY created_at DESC LIMIT 1
    ''', (user_id,))
    
    result = cursor.fetchone()
    conn.close()
    
    if result:
        return json.loads(result[0])
    else:
        return {
            'syllabus': None,
            'question_papers': [],
            'topics': {},
            'patterns': {}
        }

def update_analysis_data(user_id, data):
    """Update analysis data for the current user in DB"""
    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()
    
    # Check if we already have an analysis entry
    cursor.execute('SELECT id FROM analyses WHERE user_id = ? ORDER BY created_at DESC LIMIT 1', (user_id,))
    existing = cursor.fetchone()
    
    results_json = json.dumps(data)
    
    if existing:
        # Update existing
        cursor.execute('UPDATE analyses SET results_json = ? WHERE id = ?', (results_json, existing[0]))
    else:
        # Create new
        cursor.execute('INSERT INTO analyses (user_id, results_json) VALUES (?, ?)', (user_id, results_json))
    
    conn.commit()
    conn.close()

def extract_text_from_pdf(file_path):
    """Extract text from PDF file"""
    text = ""
    try:
        with open(file_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            for page in pdf_reader.pages:
                page_text = page.extract_text()
                # Skip instruction pages (usually first or last page with specific keywords)
                lower_text = page_text.lower()
                instruction_keywords = ['instructions to candidates', 'time allowed', 'maximum marks', 'total marks', 'printed pages', 'roll no']
                match_count = sum(1 for kw in instruction_keywords if kw in lower_text)
                
                if match_count >= 2 and len(lower_text) < 1000:
                    # Likely an instruction page
                    continue
                    
                text += page_text
    except Exception as e:
        logger.error(f"Error extracting PDF: {e}")
    return clean_text(text)

def extract_text_from_docx(file_path):
    """Extract text from DOCX file"""
    text = ""
    try:
        doc = docx.Document(file_path)
        for paragraph in doc.paragraphs:
            text += paragraph.text + "\n"
    except Exception as e:
        logger.error(f"Error extracting DOCX: {e}")
    return clean_text(text)

def extract_text_from_txt(file_path):
    """Extract text from TXT file"""
    text = ""
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            text = file.read()
    except Exception as e:
        logger.error(f"Error extracting TXT: {e}")
    return clean_text(text)

def extract_text(file_path):
    """Extract text based on file extension"""
    ext = file_path.rsplit('.', 1)[1].lower()
    if ext == 'pdf':
        return extract_text_from_pdf(file_path)
    elif ext == 'docx':
        return extract_text_from_docx(file_path)
    elif ext == 'txt':
        return extract_text_from_txt(file_path)
    return ""

def clean_text(text):
    """Clean extracted text by removing administrative lines"""
    cleaned_lines = []
    lines = text.split('\n')
    
    # Phrases that indicate administrative/instructional content
    IGNORE_PHRASES = [
        'time allowed', 'maximum marks', 'total marks', 'printed pages', 
        'roll no', 'candidate name', 'invigilator', 'semester', 
        'examination', 'university', 'college', 'course code',
        'reg no', 'date of exam', 'page', 'instructions',
        'omr', 'duration', 'hours', 'minutes', 'calculator', 'mobile',
        'electronic', 'programmable', 'communicating', 'device',
        'hall ticket', 'signature', 'blue', 'black', 'ink', 'pen', 'pencil',
        'do not', 'attempt any', 'all questions carry', 'equal marks',
        'section', 'part', 'compulsory', 'optional',
        'rough work', 'end of paper', 'blank page', 'question paper code'
    ]
    
    for line in lines:
        line_lower = line.lower().strip()
        
        # Skip empty lines
        if not line_lower:
            continue
            
        # Skip lines with administrative keywords
        if any(phrase in line_lower for phrase in IGNORE_PHRASES):
            continue
            
        # Skip short lines that are just numbers or single words
        if len(line_lower) < 4 and not line_lower.isalpha():
            continue
            
        cleaned_lines.append(line)
        
    return '\n'.join(cleaned_lines)

def is_valid_topic(text):
    """Check if the text looks like a valid academic topic"""
    text = str(text).strip()
    text_lower = text.lower()
    
    # Too short or too long
    if len(text) < 3 or len(text) > 150:
        return False
        
    # Purely numbers or symbols
    if not re.search(r'[a-zA-Z]', text):
        return False
        
    # Starts with a verb typical of instructions (but allow Question-like verbs)
    instruction_verbs = ['do', 'use', 'write', 'attempt', 'answer', 'fill', 'tick', 'mark', 'read', 'note']
    # Question verbs we WANT to keep
    question_verbs = ['explain', 'define', 'describe', 'what', 'why', 'how', 'compare', 'discuss', 'calculate', 'find']
    
    first_word = text_lower.split()[0]
    if first_word in instruction_verbs and first_word not in question_verbs:
        # Allow "Write" if it describes a technical output
        if first_word == 'write' and any(kw in text_lower for kw in ['program', 'code', 'function', 'algorithm', 'note', 'script', 'query', 'syntax']):
            pass
        # Double check it's not "Write a note on..." which is a question
        elif 'note on' in text_lower:
            pass
        else:
            return False
            
    # Contains banned words/phrases (redundant with clean_text but good for topic granularity)
    BANNED_KA_TOKENS = ['marks', 'hours', 'minutes', 'page', 'section', 'part', 'unit', 'module', 'chapter', 'question paper', 'serial no']
    if any(token in text_lower for token in BANNED_KA_TOKENS):
        return False
        
    return True

def extract_topics_from_syllabus(text):
    """Extract actual topics from syllabus by looking for structured content"""
    topics_found = []
    text_lines = text.split('\n')
    
    # Look for numbered/bulleted lists (common in syllabi)
    for line in text_lines:
        line = line.strip()
        # Match patterns like: "1. Topic Name", "• Topic", "- Topic", "(a) Topic"
        if re.match(r'^[\d\.•\-\*\(\)a-z\)]+[\.\)]\s*([A-Z][^\n]{5,100})', line):
            match = re.match(r'^[\d\.•\-\*\(\)a-z\)]+[\.\)]\s*(.+)', line)
            if match:
                topic = match.group(1).strip()
                topic = re.sub(r'^\d+[\.\)]\s*', '', topic)
                topic = re.sub(r'^[•\-\*]\s*', '', topic)
                
                # Filter out structural headers
                if re.match(r'^(?:Module|Unit|Chapter|Section|Part)\s*\d+', topic, re.IGNORECASE):
                    continue
                    
                if is_valid_topic(topic):
                    topics_found.append(topic)
        # Match lines that start with capital letters and contain subject-like terms
        elif re.match(r'^[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*', line) and len(line) < 150:
            # Check if it looks like a topic (not too long, not too short, contains letters)
            if 10 < len(line) < 150 and line[0].isupper() and is_valid_topic(line):
                topics_found.append(line)

    # Also extract from question papers - look for question patterns
    question_patterns = []
    # Match questions like "Q1. What is...", "Question 1:", etc.
    for line in text_lines:
        line = line.strip()
        if re.match(r'^(?:Q|Question)\s*\d+[\.\):]?\s*(.+)', line, re.IGNORECASE):
            match = re.match(r'^(?:Q|Question)\s*\d+[\.\):]?\s*(.+)', line, re.IGNORECASE)
            if match:
                q_text = match.group(1).strip()
                if len(q_text) > 10 and is_valid_topic(q_text):
                    question_patterns.append(q_text[:150])

    return topics_found, question_patterns

def extract_keywords_from_text(text):
    """Extract important keywords/phrases from text as topics"""
    # Remove common words
    stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'can', 'this', 'that', 'these', 'those', 'what', 'which', 'who', 'when', 'where', 'why', 'how', 'question', 'answer', 'following', 'given', 'find', 'calculate', 'determine', 'solve', 'explain', 'describe', 'discuss', 'short', 'note', 'write', 'mark', 'marks'}

    # Blocklist for administrative terms
    BLOCKLIST = {'university', 'examination', 'semester', 'degree', 'btech', 'mtech', 'bsc', 'msc', 'diploma', 'syllabus', 'paper', 'code', 'reg', 'roll', 'name', 'date', 'time', 'max', 'min', 'total', 'page', 'printed', 'valid', 'invalid', 'section', 'part', 'module', 'unit', 'chapter'}

    # Extract meaningful phrases (3-6 words) that are likely topics
    # Look for capitalized phrases
    capitalized_phrases = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3}\b', text)
    phrase_freq = Counter([p.lower() for p in capitalized_phrases if len(p.split()) >= 2])

    # Extract words
    words = re.findall(r'\b[a-z]{4,}\b', text.lower())
    filtered_words = [w for w in words if w not in stop_words and w not in BLOCKLIST and len(w) > 4]
    word_freq = Counter(filtered_words)

    # Combine
    all_keywords = {}
    for phrase, count in phrase_freq.most_common(30):
        if count > 1:
            all_keywords[phrase] = count
    for word, count in word_freq.most_common(30):
        if count > 3:  # Only if appears multiple times
            all_keywords[word] = count

    return all_keywords

def clean_topic_name(topic):
    """Clean and validate topic name"""
    if not topic:
        return None

    # Remove common artifacts
    topic = str(topic).strip()
    topic = re.sub(r'^\d+[\.\)]\s*', '', topic)

    # Skip if it's just a structural header
    if re.match(r'^(?:Module|Unit|Chapter|Section|Part)\s*\d+$', topic, re.IGNORECASE):
        return None

    # Skip if too short or too long or invalid
    if not is_valid_topic(topic):
        return None

    return topic

def identify_subjects(text):
    """Identify subjects and topics from text - extract actual topics from syllabus"""
    # First try to extract structured topics from syllabus
    syllabus_topics, question_patterns = extract_topics_from_syllabus(text)

    # Subject keywords for categorization
    # Updated Subject keywords - strictly non-overlapping
    subject_keywords = {
        'Mathematics': ['algebra', 'trigonometry', 'differential', 'integral', 'matrix', 'matrices', 'eigen', 'vector', 'theorem', 'laplace', 'fourier', 'calculus', 'probability', 'statistics', 'complex', 'numerical'],
        'Reasoning': ['seating arrangement', 'blood relation', 'coding decoding', 'syllogism', 'puzzles', 'data sufficiency', 'logical', 'series', 'direction', 'ranking'],
        'English': ['synonym', 'antonym', 'comprehension', 'vocabulary', 'grammar', 'essay', 'precis', 'idioms', 'phrases', 'correction'],
        'General Knowledge': ['current affairs', 'history', 'polity', 'economy', 'geography', 'constitution', 'culture', 'sports', 'science'],
        'Quantitative Aptitude': ['profit', 'loss', 'interest', 'time', 'work', 'speed', 'distance', 'ratio', 'proportion', 'average', 'percentage', 'mensuration'],

        # Engineering - Specific Technical Terms
        'Electrical Engineering': ['kirchhoff', 'thevenin', 'norton', 'superposition', 'transformer', 'induction', 'synchronous', 'generator', 'transmission', 'switchgear', 'power', 'circuit', 'voltage', 'current', 'motor'],
        'Civil Engineering': ['concrete', 'cement', 'soil', 'fluid', 'surveying', 'structural', 'reinforced', 'beam', 'column', 'foundation', 'hydrology', 'highway', 'transportation', 'environmental'],
        'Mechanical Engineering': ['thermodynamics', 'rankine', 'otto', 'diesel', 'refrigeration', 'fluid', 'manufacturing', 'welding', 'casting', 'gears', 'stress', 'strain', 'kinematics'],
        'Computer Science': ['algorithm', 'data structure', 'operating system', 'database', 'sql', 'compiler', 'network', 'protocol', 'stack', 'queue', 'linked list', 'tree', 'graph', 'automata', 'cloud', 'security'],
        'Electronics Engineering': ['semiconductor', 'transistor', 'op-amp', 'oscillator', 'microprocessor', 'digital', 'embedded', 'vlsi', 'analog', 'signal', 'communication', 'antenna', 'modulation']
    }

    text_lower = text.lower()
    subject_counts = {}
    topic_mapping = {}

    # Categorize extracted syllabus topics
    for topic in syllabus_topics[:50]:  # Limit to 50 topics
        topic_lower = topic.lower()
        # Find which subject this topic belongs to
        matched_subject = None
        max_matches = 0

        for subject, keywords in subject_keywords.items():
            # Require stronger matches for classification
            matches = sum(1 for kw in keywords if kw in topic_lower)
            if matches > 0:
                # Weighted matching: Technical subjects need fewer keywords than General ones if the keywords are unique
                # Weighted matching: Technical subjects need fewer keywords than General ones if the keywords are unique
                if subject in ['Mathematics', 'Reasoning', 'General Knowledge', 'English', 'Quantitative Aptitude']:
                     if matches > max_matches: # Require robust match
                        max_matches = matches
                        matched_subject = subject
                else:
                    # Technical subjects match easier with specific keywords
                    if matches >= 1:
                        max_matches = matches + 1 # Boost preference
                        matched_subject = subject

        # If no match, assign to "Detected Topics" instead of General
        if not matched_subject:
            # Check context from earlier - if we established a subject, use it? For now, flexible fallback.
            matched_subject = 'Detected Topics'

        if matched_subject not in topic_mapping:
            topic_mapping[matched_subject] = []
            subject_counts[matched_subject] = 0

        # Add topic if not already there
        existing_topics = [t['topic'] for t in topic_mapping[matched_subject]]
        if topic not in existing_topics:
            topic_mapping[matched_subject].append({'topic': topic, 'count': 1})
            subject_counts[matched_subject] += 1

    # If no structured topics found, fall back to keyword extraction
    if not topic_mapping:
        extracted_keywords = extract_keywords_from_text(text)
        if extracted_keywords:
            # Categorize keywords
            for kw, cnt in list(extracted_keywords.items())[:30]:
                matched_subject = None
                for subject, keywords in subject_keywords.items():
                    if any(k in kw for k in keywords):
                        matched_subject = subject
                        break

                if not matched_subject:
                    matched_subject = 'Detected Topics'

                if matched_subject not in topic_mapping:
                    topic_mapping[matched_subject] = []
                    subject_counts[matched_subject] = 0

                topic_mapping[matched_subject].append({'topic': kw.title(), 'count': cnt})
                subject_counts[matched_subject] += cnt

    return subject_counts, topic_mapping, question_patterns

def format_question(topic, count):
    """Format a topic into a realistic question"""
    topic = str(topic).strip()

    # Check if it already looks like a question
    if '?' in topic or topic.lower().startswith(('what', 'define', 'explain', 'describe')):
        return topic

    start_phrases = [
        "Explain the concept of {} in detail.",
        "What is {}? Discuss its verification properties.",
        "Write a short note on {}.",
        "Describe the working principle of {}.",
        "Compare and contrast {} with its alternatives.",
        "Discuss the importance of {}."
    ]

    # Deterministic but varied choice based on topic hash
    import hashlib
    hash_val = int(hashlib.md5(topic.encode()).hexdigest(), 16)
    phrase = start_phrases[hash_val % len(start_phrases)]

    return phrase.format(topic)

def analyze_question_patterns(question_papers):
    """Analyze patterns across multiple question papers - uses actual data"""
    all_topics = {}
    total_questions = 0
    difficulty_distribution = {'Easy': 0, 'Medium': 0, 'Hard': 0}

    for paper in question_papers:
        text = paper.get('text', '')
        if not text:
            continue

        # Extract topics from this paper
        subjects, topics, questions = identify_subjects(text)

        # Collect extracted questions if any
        # (This could be stored globally or per paper, but for now we just use topics)

        # Count questions in the paper
        question_count = len(re.findall(r'(?:Q|Question)\s*\d+', text, re.IGNORECASE))
        if question_count == 0:
            # Try alternative patterns
            question_count = len(re.findall(r'\d+[\.\)]\s+[A-Z]', text)) or len(text.split('?')) - 1
        total_questions += max(question_count, 1)

        # Aggregate topics
        for subject, subject_topics in topics.items():
            if subject not in all_topics:
                all_topics[subject] = {}
            for topic_data in subject_topics:
                if isinstance(topic_data, dict):
                    topic = topic_data.get('topic', '')
                    count = topic_data.get('count', 1)
                    if topic:
                        if topic not in all_topics[subject]:
                            all_topics[subject][topic] = 0
                        all_topics[subject][topic] += count

        # Estimate difficulty based on question length and complexity
        # Longer questions with technical terms = harder
        avg_question_length = len(text) / max(question_count, 1) if question_count > 0 else 0
        technical_terms = len(re.findall(r'\b(?:calculate|determine|solve|prove|derive|analyze|design)\b', text, re.IGNORECASE))

        if technical_terms > 5 or avg_question_length > 200:
            difficulty_distribution['Hard'] += question_count // 2
            difficulty_distribution['Medium'] += question_count // 3
            difficulty_distribution['Easy'] += question_count - (question_count // 2) - (question_count // 3)
        elif technical_terms > 2 or avg_question_length > 100:
            difficulty_distribution['Medium'] += question_count // 2
            difficulty_distribution['Easy'] += question_count // 2
            difficulty_distribution['Hard'] += question_count - (question_count // 2) - (question_count // 2)
        else:
            difficulty_distribution['Easy'] += question_count // 2
            difficulty_distribution['Medium'] += question_count // 3
            difficulty_distribution['Hard'] += question_count - (question_count // 2) - (question_count // 3)

    # Normalize difficulty distribution if we have total questions
    if total_questions > 0:
        total_dist = sum(difficulty_distribution.values())
        if total_dist == 0:
            # Default distribution
            difficulty_distribution = {
                'Easy': int(total_questions * 0.4),
                'Medium': int(total_questions * 0.4),
                'Hard': int(total_questions * 0.2)
            }
        else:
            # Scale to match total questions
            scale = total_questions / total_dist
            difficulty_distribution = {
                'Easy': int(difficulty_distribution['Easy'] * scale),
                'Medium': int(difficulty_distribution['Medium'] * scale),
                'Hard': int(difficulty_distribution['Hard'] * scale)
            }

    return all_topics, difficulty_distribution

def calculate_prediction_probability(topic_count, total_papers, syllabus_match):
    """Calculate probability of a topic appearing in the exam"""
    frequency_score = (topic_count / total_papers) * 100
    syllabus_score = syllabus_match * 100

    # Weighted average (60% frequency, 40% syllabus match)
    probability = (frequency_score * 0.6 + syllabus_score * 0.4)
    return min(probability, 99)  # Cap at 99%

def generate_predictions(syllabus_topics, question_paper_topics, num_papers):
    """Generate predicted question topics using ML-based similarity"""

    # 1. Flatten syllabus topics into a list of strings
    flat_syllabus_topics = []
    if syllabus_topics:
        for subject, topic_list in syllabus_topics.items():
            if isinstance(topic_list, list):
                # Extract topic strings
                for t in topic_list:
                    t_name = t.get('topic', '') if isinstance(t, dict) else str(t)
                    if t_name:
                        flat_syllabus_topics.append(t_name)

    # 2. Get full text of question papers
    user_id = get_user_id()
    analysis_data = get_analysis_data(user_id)
    
    qp_text = ""
    if analysis_data.get('question_papers'):
        qp_text = " ".join([qp.get('text', '') for qp in analysis_data['question_papers']])

    # 3. Extract metadata for enhanced predictions
    metadata = predictor.extract_exam_metadata(qp_text) if qp_text else {}

    # 4. Use ML Predictor
    ml_predictions = predictor.predict_questions(flat_syllabus_topics, qp_text, metadata, num_predictions=30)

    # 4. Format Predictions
    final_predictions = []

    if ml_predictions:
        for item in ml_predictions:
            # We need to find the subject for this topic
            # This is a bit inefficient reverse lookup but robust
            subject = 'General'
            topic_name = item['topic']

            # Find subject
            if syllabus_topics:
                for subj, t_list in syllabus_topics.items():
                    # Flatten t_list to strings
                    if not t_list: continue
                    simple_list = [t.get('topic', '') if isinstance(t, dict) else str(t) for t in t_list]
                    if topic_name in simple_list:
                        subject = subj
                        break

            final_predictions.append({
                'subject': subject,
                'topic': topic_name.title(),
                'question': format_question(topic_name, 1), # Count is 1 for now
                'frequency': 1, # ML doesn't count exact frequency
                'probability': item['probability'],
                'question_type': item['question_type'],
                'difficulty': item['difficulty'],
                'exam_type': item['exam_type'],
                'conducting_authority': item['conducting_authority'],
                'score': item['score'] # Debug info
            })
    else:
        # Fallback to simple matching if ML failed or no data
        logger.warning("Using heuristic fallback for predictions")
        
        # Determine strict or loose matching based on whether we have a syllabus
        use_syllabus_fallback = bool(syllabus_topics)

        if not question_paper_topics or len(question_paper_topics) == 0:
             if use_syllabus_fallback:
                 # Just list syllabus topics
                 for subject, topic_list in syllabus_topics.items():
                     if isinstance(topic_list, list):
                         for topic_item in topic_list[:20]:
                             t_name = topic_item.get('topic', '') if isinstance(topic_item, dict) else str(topic_item)
                             if t_name:
                                 final_predictions.append({
                                     'subject': subject,
                                     'topic': t_name.title(),
                                     'question': format_question(t_name, 1),
                                     'frequency': 1,
                                     'probability': 85.0,
                                     'question_type': 'MCQ (1 mark)',
                                     'difficulty': 'Medium'
                                 })

        # Let's restore the heuristic loop properly
        for subject in question_paper_topics:
             if not question_paper_topics[subject]: continue

             topics_dict = {}
             if isinstance(question_paper_topics[subject], dict):
                 topics_dict = question_paper_topics[subject]
             elif isinstance(question_paper_topics[subject], list):
                 for item in question_paper_topics[subject]:
                     if isinstance(item, dict):
                         topics_dict[item.get('topic', '')] = item.get('count', 1)
                     else:
                         topics_dict[str(item)] = 1

             for topic, count in topics_dict.items():
                 if not topic: continue
                 clean_topic = clean_topic_name(topic)
                 if not clean_topic: continue

                 final_predictions.append({
                     'subject': subject,
                     'topic': clean_topic.title(),
                     'question': format_question(clean_topic, count),
                     'frequency': count,
                     'probability': 80.0, # Default high confidence for found topics
                     'question_type': 'Descriptive',
                     'difficulty': 'Medium'
                 })

    return final_predictions[:30]

@app.route('/api/upload/syllabus', methods=['POST'])
def upload_syllabus():
    """Upload and process syllabus file"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400

        file = request.files['file']

        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400

        try:
            filename, upload_id = validate_and_save_file(file, 'syllabus')
        except ValueError as e:
            return jsonify({'error': str(e)}), 400

        # Extract text and analyze
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        text = extract_text(filepath)
        subjects, topics, questions = identify_subjects(text)

        # Get current analysis data and update syllabus
        user_id = get_user_id()
        analysis_data = get_analysis_data(user_id)
        analysis_data['syllabus'] = {
            'filename': filename,
            'text': text,
            'subjects': subjects,
            'topics': topics
        }
        update_analysis_data(user_id, analysis_data)

        return jsonify({
            'success': True,
            'filename': filename,
            'subjects': subjects,
            'message': 'Syllabus uploaded and analyzed successfully'
        }), 200
    except Exception as e:
        logger.error(f"Error in upload_syllabus: {str(e)}", exc_info=True)
        return jsonify({'error': 'Internal server error during syllabus upload'}), 500

@app.route('/api/upload/question-papers', methods=['POST'])
def upload_question_papers():
    """Upload and process question paper files"""
    try:
        if 'files' not in request.files:
            return jsonify({'error': 'No files provided'}), 400

        files = request.files.getlist('files')

        if not files or files[0].filename == '':
            return jsonify({'error': 'No files selected'}), 400

        uploaded_files = []

        for file in files:
            try:
                filename, upload_id = validate_and_save_file(file, 'qp')
            except ValueError as e:
                return jsonify({'error': str(e)}), 400

            # Extract text and analyze
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            text = extract_text(filepath)
            subjects, topics, questions = identify_subjects(text)

            paper_data = {
                'filename': filename,
                'text': text,
                'subjects': subjects,
                'topics': topics
            }

            # Add to question papers in analysis data
            user_id = get_user_id()
            analysis_data = get_analysis_data(user_id)
            analysis_data['question_papers'].append(paper_data)
            update_analysis_data(user_id, analysis_data)
            
            uploaded_files.append(filename)

        return jsonify({
            'success': True,
            'files': uploaded_files,
            'count': len(uploaded_files),
            'message': f'{len(uploaded_files)} question paper(s) uploaded and analyzed successfully'
        }), 200
    except Exception as e:
        logger.error(f"Error in upload_question_papers: {str(e)}", exc_info=True)
        return jsonify({'error': 'Internal server error during question paper upload'}), 500

@app.route('/api/analyze', methods=['POST'])
def analyze_data():
    """Perform comprehensive analysis of uploaded data - works with syllabus OR question papers OR both"""
    try:
        user_id = get_user_id()
        analysis_data = get_analysis_data(user_id)
        
        has_syllabus = analysis_data['syllabus'] is not None
        has_question_papers = len(analysis_data['question_papers']) > 0

        if not has_syllabus and not has_question_papers:
            return jsonify({'error': 'Please upload at least syllabus or question papers'}), 400

        all_topics = {}
        difficulty_dist = {'Easy': 0, 'Medium': 0, 'Hard': 0}

        # Analyze question papers if available
        if has_question_papers:
            all_topics, difficulty_dist = analyze_question_patterns(analysis_data['question_papers'])
        else:
            # If only syllabus, extract topics from syllabus
            if has_syllabus:
                syllabus_text = analysis_data['syllabus'].get('text', '')
                if syllabus_text:
                    subjects, topics, questions = identify_subjects(syllabus_text)
                    # Convert to all_topics format
                    for subject, topic_list in topics.items():
                        all_topics[subject] = {}
                        for topic_item in topic_list:
                            topic_name = topic_item['topic']
                            all_topics[subject][topic_name] = topic_item['count']
                    # Generate random difficulty distribution -> Default to even if no data
                    difficulty_dist = {
                        'Easy': 34,
                        'Medium': 33,
                        'Hard': 33
                    }

        # Calculate syllabus coverage (if syllabus exists)
        coverage = {}
        if has_syllabus:
            syllabus_subjects = analysis_data['syllabus'].get('subjects', {})
            syllabus_topics = analysis_data['syllabus'].get('topics', {})

            if syllabus_subjects:
                for subject in syllabus_subjects.keys():
                    if subject in all_topics:
                        covered = len(all_topics[subject])
                        total = len(syllabus_topics.get(subject, []))
                        if total > 0:
                            coverage[subject] = min(round((covered / max(total, 1)) * 100, 1), 100.0)
                        else:
                            coverage[subject] = 0.0
                    else:
                        coverage[subject] = 0.0
            else:
                # If no predefined subjects, create coverage from all_topics
                for subject in all_topics.keys():
                    coverage[subject] = 0.0  # Cannot calculate coverage without syllabus matched topics
        else:
            # If no syllabus, create coverage from question paper topics
            for subject in all_topics.keys():
                coverage[subject] = 0.0  # No syllabus to compare against

        # Get top topics by frequency
        topic_frequency = []
        for subject, topics in all_topics.items():
            if isinstance(topics, dict):
                for topic, count in topics.items():
                    topic_frequency.append({
                        'subject': subject,
                        'topic': topic.title() if topic else 'Unknown',
                        'count': count
                    })

        topic_frequency.sort(key=lambda x: x['count'], reverse=True)

        # Ensure we have some topics
        if not topic_frequency and (has_syllabus or has_question_papers):
            # Fallback: extract from text directly
            text_to_analyze = ""
            if has_syllabus:
                text_to_analyze += analysis_data['syllabus'].get('text', '')
            if has_question_papers:
                for paper in analysis_data['question_papers']:
                    text_to_analyze += paper.get('text', '')

            if text_to_analyze:
                keywords = extract_keywords_from_text(text_to_analyze)
                for kw, count in list(keywords.items())[:10]:
                    topic_frequency.append({
                        'subject': 'General',
                        'topic': kw.title(),
                        'count': count
                    })
                # Add to all_topics
                all_topics['General'] = dict(list(keywords.items())[:20])

        # Update analysis data
        analysis_data['topics'] = all_topics
        analysis_data['patterns'] = {
            'coverage': coverage,
            'topic_frequency': topic_frequency[:10],
            'difficulty_distribution': difficulty_dist
        }
        update_analysis_data(user_id, analysis_data)

        return jsonify({
            'success': True,
            'analysis': {
                'syllabus_coverage': coverage,
                'topic_frequency': topic_frequency[:10] if topic_frequency else [],
                'difficulty_distribution': difficulty_dist,
                'total_papers_analyzed': len(analysis_data['question_papers']) if has_question_papers else 0,
                'has_syllabus': has_syllabus,
                'has_question_papers': has_question_papers
            }
        }), 200
    except Exception as e:
        logger.error(f"Error in analyze_data: {str(e)}", exc_info=True)
        return jsonify({'error': 'Internal server error during analysis'}), 500

@app.route('/api/predict', methods=['POST'])
def predict_questions():
    """Generate predicted question paper - works with syllabus OR question papers OR both"""
    try:
        user_id = get_user_id()
        analysis_data = get_analysis_data(user_id)
        
        has_syllabus = analysis_data['syllabus'] is not None
        has_question_papers = len(analysis_data['question_papers']) > 0

        if not has_syllabus and not has_question_papers:
            return jsonify({'error': 'Please upload at least syllabus or question papers and analyze first'}), 400

        if not analysis_data.get('topics'):
            return jsonify({'error': 'Please run analysis first to identify topics'}), 400

        # Get topics - use syllabus topics if available, otherwise use analyzed topics
        syllabus_topics = {}
        if has_syllabus:
            syllabus_topics = analysis_data['syllabus'].get('topics', {})

        question_paper_topics = analysis_data.get('topics', {})
        num_papers = len(analysis_data['question_papers']) if has_question_papers else 1

        # If no topics found, extract from available data
        if not question_paper_topics:
            text_to_analyze = ""
            if has_syllabus:
                text_to_analyze += analysis_data['syllabus'].get('text', '')
            if has_question_papers:
                for paper in analysis_data['question_papers']:
                    text_to_analyze += paper.get('text', '')

            if text_to_analyze:
                keywords = extract_keywords_from_text(text_to_analyze)
                # Only use if we actually found keywords
                if keywords:
                    question_paper_topics = {'General': dict(list(keywords.items())[:20])}
                    analysis_data['topics'] = question_paper_topics
                    update_analysis_data(user_id, analysis_data)

        # Debug info
        logger.info(f"Generating predictions - Syllabus: {has_syllabus}, QP: {has_question_papers}")
        logger.info(f"Syllabus topics: {list(syllabus_topics.keys())}")
        logger.info(f"Question paper topics: {list(question_paper_topics.keys())}")

        predictions = generate_predictions(syllabus_topics, question_paper_topics, max(num_papers, 1))

        # If still no predictions, create some from available topics
        if not predictions and question_paper_topics:
            predictions = []
            for subject, topics in question_paper_topics.items():
                if isinstance(topics, dict):
                    for topic, count in list(topics.items())[:10]:
                        predictions.append({
                            'subject': subject,
                            'topic': topic.title() if topic else 'General Topic',
                            'question': format_question(topic, count),
                            'frequency': count,
                            'probability': min(round(60 + (count * 2), 1), 99),
                            'question_type': 'MCQ (1 mark)',
                            'difficulty': 'Medium'
                        })
                elif isinstance(topics, list):
                    for topic_item in topics[:10]:
                        if isinstance(topic_item, dict):
                            topic_name = topic_item.get('topic', 'Unknown')
                            count = topic_item.get('count', 1)
                            predictions.append({
                                'subject': subject,
                                'topic': topic_name.title(),
                                'question': format_question(topic_name, count),
                                'frequency': count,
                                'probability': min(round(60 + (count * 2), 1), 99),
                                'question_type': 'MCQ (1 mark)',
                                'difficulty': 'Medium'
                            })

            predictions.sort(key=lambda x: x.get('probability', 0), reverse=True)
            predictions = predictions[:20]

        logger.info(f"Generated {len(predictions)} predictions")

        # Save predictions to database
        conn = sqlite3.connect(DATABASE)
        cursor = conn.cursor()
        cursor.execute(
            'INSERT INTO predictions (user_id, prediction_results) VALUES (?, ?)',
            (user_id, json.dumps(predictions))
        )
        prediction_id = cursor.lastrowid
        conn.commit()
        conn.close()

        return jsonify({
            'success': True,
            'predictions': predictions,
            'total_predictions': len(predictions),
            'papers_analyzed': num_papers,
            'message': f'Generated {len(predictions)} predictions'
        }), 200
    except Exception as e:
        logger.error(f"Error in predict_questions: {str(e)}", exc_info=True)
        return jsonify({'error': 'Internal server error during prediction'}), 500

@app.route('/api/download-prediction', methods=['GET'])
def download_prediction():
    """Download predicted question paper as JSON"""
    try:
        user_id = get_user_id()
        
        # Get latest predictions for this user
        conn = sqlite3.connect(DATABASE)
        cursor = conn.cursor()
        cursor.execute('''
            SELECT prediction_results FROM predictions 
            WHERE user_id = ? 
            ORDER BY created_at DESC LIMIT 1
        ''', (user_id,))
        
        result = cursor.fetchone()
        conn.close()
        
        if not result:
            return jsonify({'error': 'No predictions available'}), 400
        
        predictions = json.loads(result[0])
        
        return jsonify({
            'predictions': predictions
        }), 200
    except Exception as e:
        logger.error(f"Error in download_prediction: {str(e)}", exc_info=True)
        return jsonify({'error': 'Internal server error during prediction download'}), 500

@app.route('/api/status', methods=['GET'])
def get_status():
    """Get current status of uploaded data"""
    try:
        user_id = get_user_id()
        analysis_data = get_analysis_data(user_id)
        
        return jsonify({
            'syllabus_uploaded': analysis_data['syllabus'] is not None,
            'question_papers_count': len(analysis_data['question_papers']),
            'analysis_completed': 'patterns' in analysis_data and len(analysis_data['patterns']) > 0
        }), 200
    except Exception as e:
        logger.error(f"Error in get_status: {str(e)}", exc_info=True)
        return jsonify({'error': 'Internal server error getting status'}), 500

@app.route('/api/reset', methods=['POST'])
def reset_data():
    """Reset all uploaded data for current user"""
    try:
        user_id = get_user_id()
        
        # Reset analysis data for this user
        update_analysis_data(user_id, {
            'syllabus': None,
            'question_papers': [],
            'topics': {},
            'patterns': {}
        })
        
        # Clear session data
        session.pop('user_id', None)

        return jsonify({
            'success': True,
            'message': 'All data has been reset'
        }), 200
    except Exception as e:
        logger.error(f"Error in reset_data: {str(e)}", exc_info=True)
        return jsonify({'error': 'Internal server error during reset'}), 500

if __name__ == '__main__':
    # Initialize the database
    init_db()
    # Check if running on Render or similar platform
    import os
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)


# Error handlers
@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404


@app.errorhandler(500)
def internal_error(error):
    logger.error(f"Server Error: {error}", exc_info=True)
    return jsonify({'error': 'Internal server error'}), 500


@app.errorhandler(Exception)
def handle_exception(e):
    # Log the error
    logger.error(f"Unhandled exception: {str(e)}", exc_info=True)
    # Return JSON error response
    return jsonify({'error': 'An unexpected error occurred'}), 500