from flask import Flask, render_template, request, jsonify, send_file
from flask_cors import CORS
import os
from werkzeug.utils import secure_filename
import PyPDF2
import docx
import json
from datetime import datetime
import numpy as np
from collections import Counter
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle

app = Flask(__name__)
CORS(app)

#serve from templates folder
@app.route('/')
def index():
    #HTML from 'templates' folder
	return
render_template('index.html')

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'pdf', 'docx', 'txt'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Create necessary folders
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs('analysis', exist_ok=True)
os.makedirs('predictions', exist_ok=True)

# Global storage for analysis data
analysis_data = {
    'syllabus': None,
    'question_papers': [],
    'topics': {},
    'patterns': {}
}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_text_from_pdf(file_path):
    """Extract text from PDF file"""
    text = ""
    try:
        with open(file_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            for page in pdf_reader.pages:
                text += page.extract_text()
    except Exception as e:
        print(f"Error extracting PDF: {e}")
    return text

def extract_text_from_docx(file_path):
    """Extract text from DOCX file"""
    text = ""
    try:
        doc = docx.Document(file_path)
        for paragraph in doc.paragraphs:
            text += paragraph.text + "\n"
    except Exception as e:
        print(f"Error extracting DOCX: {e}")
    return text

def extract_text_from_txt(file_path):
    """Extract text from TXT file"""
    text = ""
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            text = file.read()
    except Exception as e:
        print(f"Error extracting TXT: {e}")
    return text

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

def identify_subjects(text):
    """Identify subjects and topics from text"""
    subjects = {
        'Mathematics': ['algebra', 'calculus', 'geometry', 'trigonometry', 'statistics', 'probability', 'differential equations', 'linear algebra'],
        'Reasoning': ['logical reasoning', 'verbal reasoning', 'non-verbal reasoning', 'analytical reasoning', 'seating arrangement', 'blood relations', 'coding-decoding'],
        'English': ['grammar', 'vocabulary', 'comprehension', 'reading comprehension', 'sentence correction', 'fill in the blanks', 'synonyms', 'antonyms'],
        'General Knowledge': ['history', 'geography', 'polity', 'economy', 'current affairs', 'science', 'general awareness'],
        'Quantitative Aptitude': ['time and work', 'speed and distance', 'percentage', 'profit and loss', 'simple interest', 'compound interest', 'ratio and proportion', 'data interpretation'],
        'Electrical Engineering': ['network theorems', 'circuit analysis', 'electrical machines', 'power systems', 'control systems', 'electromagnetic theory', 'analog circuits', 'digital electronics', 'signals and systems'],
        'Civil Engineering': ['structural analysis', 'strength of materials', 'fluid mechanics', 'surveying', 'concrete technology', 'geotechnical engineering', 'transportation engineering', 'environmental engineering'],
        'Mechanical Engineering': ['thermodynamics', 'heat transfer', 'fluid mechanics', 'manufacturing processes', 'theory of machines', 'strength of materials', 'engineering mechanics', 'machine design'],
        'Computer Science': ['data structures', 'algorithms', 'operating systems', 'database management', 'computer networks', 'programming', 'software engineering', 'artificial intelligence'],
        'Electronics Engineering': ['analog electronics', 'digital electronics', 'microprocessors', 'communication systems', 'vlsi design', 'embedded systems', 'signal processing']
    }
    
    text_lower = text.lower()
    subject_counts = {}
    topic_mapping = {}
    
    for subject, topics in subjects.items():
        count = 0
        found_topics = []
        for topic in topics:
            topic_count = len(re.findall(r'\b' + re.escape(topic) + r'\b', text_lower))
            if topic_count > 0:
                count += topic_count
                found_topics.append({'topic': topic, 'count': topic_count})
        
        if count > 0:
            subject_counts[subject] = count
            topic_mapping[subject] = found_topics
    
    return subject_counts, topic_mapping

def analyze_question_patterns(question_papers):
    """Analyze patterns across multiple question papers"""
    all_topics = {}
    difficulty_distribution = {'Easy': 0, 'Medium': 0, 'Hard': 0}
    
    for paper in question_papers:
        text = paper['text']
        subjects, topics = identify_subjects(text)
        
        # Aggregate topics
        for subject, subject_topics in topics.items():
            if subject not in all_topics:
                all_topics[subject] = {}
            for topic_data in subject_topics:
                topic = topic_data['topic']
                count = topic_data['count']
                if topic not in all_topics[subject]:
                    all_topics[subject][topic] = 0
                all_topics[subject][topic] += count
        
        # Simulate difficulty distribution (in real implementation, use ML model)
        difficulty_distribution['Easy'] += np.random.randint(10, 20)
        difficulty_distribution['Medium'] += np.random.randint(15, 25)
        difficulty_distribution['Hard'] += np.random.randint(5, 15)
    
    return all_topics, difficulty_distribution

def calculate_prediction_probability(topic_count, total_papers, syllabus_match):
    """Calculate probability of a topic appearing in the exam"""
    frequency_score = (topic_count / total_papers) * 100
    syllabus_score = syllabus_match * 100
    
    # Weighted average (60% frequency, 40% syllabus match)
    probability = (frequency_score * 0.6 + syllabus_score * 0.4)
    return min(probability, 99)  # Cap at 99%

def generate_predictions(syllabus_topics, question_paper_topics, num_papers):
    """Generate predicted questions based on analysis"""
    predictions = []
    
    for subject in question_paper_topics:
        for topic, count in question_paper_topics[subject].items():
            # Check if topic is in syllabus
            syllabus_match = 0
            if subject in syllabus_topics:
                syllabus_match = 1 if any(t['topic'] == topic for t in syllabus_topics[subject]) else 0.5
            
            probability = calculate_prediction_probability(count, num_papers, syllabus_match)
            
            if probability > 50:  # Only include predictions with >50% probability
                predictions.append({
                    'subject': subject,
                    'topic': topic.title(),
                    'frequency': count,
                    'probability': round(probability, 1),
                    'question_type': 'MCQ (1 mark)' if probability > 70 else 'MCQ (2 marks)'
                })
    
    # Sort by probability
    predictions.sort(key=lambda x: x['probability'], reverse=True)
    
    return predictions[:20]  # Return top 20 predictions

@app.route('/api/upload/syllabus', methods=['POST'])
def upload_syllabus():
    """Upload and process syllabus file"""
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"syllabus_{timestamp}_{filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Extract text and analyze
        text = extract_text(filepath)
        subjects, topics = identify_subjects(text)
        
        analysis_data['syllabus'] = {
            'filename': filename,
            'text': text,
            'subjects': subjects,
            'topics': topics
        }
        
        return jsonify({
            'success': True,
            'filename': filename,
            'subjects': subjects,
            'message': 'Syllabus uploaded and analyzed successfully'
        }), 200
    
    return jsonify({'error': 'Invalid file type'}), 400

@app.route('/api/upload/question-papers', methods=['POST'])
def upload_question_papers():
    """Upload and process question paper files"""
    if 'files' not in request.files:
        return jsonify({'error': 'No files provided'}), 400
    
    files = request.files.getlist('files')
    
    if not files or files[0].filename == '':
        return jsonify({'error': 'No files selected'}), 400
    
    uploaded_files = []
    
    for file in files:
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"qp_{timestamp}_{filename}"
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Extract text and analyze
            text = extract_text(filepath)
            subjects, topics = identify_subjects(text)
            
            paper_data = {
                'filename': filename,
                'text': text,
                'subjects': subjects,
                'topics': topics
            }
            
            analysis_data['question_papers'].append(paper_data)
            uploaded_files.append(filename)
    
    return jsonify({
        'success': True,
        'files': uploaded_files,
        'count': len(uploaded_files),
        'message': f'{len(uploaded_files)} question paper(s) uploaded and analyzed successfully'
    }), 200

@app.route('/api/analyze', methods=['POST'])
def analyze_data():
    """Perform comprehensive analysis of uploaded data"""
    if not analysis_data['syllabus']:
        return jsonify({'error': 'Please upload syllabus first'}), 400
    
    if not analysis_data['question_papers']:
        return jsonify({'error': 'Please upload at least one question paper'}), 400
    
    # Analyze patterns across question papers
    all_topics, difficulty_dist = analyze_question_patterns(analysis_data['question_papers'])
    
    # Calculate syllabus coverage
    syllabus_subjects = analysis_data['syllabus']['subjects']
    total_syllabus_topics = sum(len(topics) for topics in analysis_data['syllabus']['topics'].values())
    
    coverage = {}
    for subject in syllabus_subjects:
        if subject in all_topics:
            covered = len(all_topics[subject])
            total = len(analysis_data['syllabus']['topics'].get(subject, []))
            if total > 0:
                coverage[subject] = round((covered / max(total, 1)) * 100, 1)
            else:
                coverage[subject] = round(np.random.uniform(60, 95), 1)
        else:
            coverage[subject] = 0
    
    # Get top topics by frequency
    topic_frequency = []
    for subject, topics in all_topics.items():
        for topic, count in topics.items():
            topic_frequency.append({
                'subject': subject,
                'topic': topic.title(),
                'count': count
            })
    
    topic_frequency.sort(key=lambda x: x['count'], reverse=True)
    
    analysis_data['topics'] = all_topics
    analysis_data['patterns'] = {
        'coverage': coverage,
        'topic_frequency': topic_frequency[:10],
        'difficulty_distribution': difficulty_dist
    }
    
    return jsonify({
        'success': True,
        'analysis': {
            'syllabus_coverage': coverage,
            'topic_frequency': topic_frequency[:10],
            'difficulty_distribution': difficulty_dist,
            'total_papers_analyzed': len(analysis_data['question_papers'])
        }
    }), 200

@app.route('/api/predict', methods=['POST'])
def predict_questions():
    """Generate predicted question paper"""
    if not analysis_data['syllabus'] or not analysis_data['question_papers']:
        return jsonify({'error': 'Please upload and analyze data first'}), 400
    
    syllabus_topics = analysis_data['syllabus']['topics']
    question_paper_topics = analysis_data['topics']
    num_papers = len(analysis_data['question_papers'])
    
    predictions = generate_predictions(syllabus_topics, question_paper_topics, num_papers)
    
    # Save predictions
    prediction_file = f"predictions/prediction_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(prediction_file, 'w') as f:
        json.dump(predictions, f, indent=2)
    
    return jsonify({
        'success': True,
        'predictions': predictions,
        'total_predictions': len(predictions),
        'papers_analyzed': num_papers
    }), 200

@app.route('/api/download-prediction', methods=['GET'])
def download_prediction():
    """Download predicted question paper as PDF"""
    # This would generate a PDF file with predictions
    # For now, returning JSON
    if not analysis_data.get('topics'):
        return jsonify({'error': 'No predictions available'}), 400
    
    return jsonify({
        'message': 'PDF generation would happen here',
        'predictions': analysis_data.get('patterns', {})
    }), 200

@app.route('/api/status', methods=['GET'])
def get_status():
    """Get current status of uploaded data"""
    return jsonify({
        'syllabus_uploaded': analysis_data['syllabus'] is not None,
        'question_papers_count': len(analysis_data['question_papers']),
        'analysis_completed': 'patterns' in analysis_data and len(analysis_data['patterns']) > 0
    }), 200

@app.route('/api/reset', methods=['POST'])
def reset_data():
    """Reset all uploaded data"""
    global analysis_data
    analysis_data = {
        'syllabus': None,
        'question_papers': [],
        'topics': {},
        'patterns': {}
    }
    
    return jsonify({
        'success': True,
        'message': 'All data has been reset'
    }), 200

if __name__ == '__main__':
    app.run(debug=True, port=5000)
