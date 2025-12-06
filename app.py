from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import os
from werkzeug.utils import secure_filename
import PyPDF2
import docx
import json
from datetime import datetime
import numpy as np
import re
from collections import Counter

app = Flask(__name__)
CORS(app)

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'pdf', 'docx', 'txt'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['ALLOWED_EXTENSIONS'] = {'pdf', 'docx', 'txt'}
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Create necessary folders
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs('analysis', exist_ok=True)
os.makedirs('predictions', exist_ok=True)

#serve from templates folder
@app.route('/')
def index():
    #HTML from 'templates' folder
    return render_template('index.html')

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
        print(f"Error extracting PDF: {e}")
    return clean_text(text)

def extract_text_from_docx(file_path):
    """Extract text from DOCX file"""
    text = ""
    try:
        doc = docx.Document(file_path)
        for paragraph in doc.paragraphs:
            text += paragraph.text + "\n"
    except Exception as e:
        print(f"Error extracting DOCX: {e}")
    return clean_text(text)

def extract_text_from_txt(file_path):
    """Extract text from TXT file"""
    text = ""
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            text = file.read()
    except Exception as e:
        print(f"Error extracting TXT: {e}")
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
        'reg no', 'date of exam', 'page', 'instructions'
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
                    
                if len(topic) > 5 and len(topic) < 150:
                    topics_found.append(topic)
        # Match lines that start with capital letters and contain subject-like terms
        elif re.match(r'^[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*', line) and len(line) < 150:
            # Check if it looks like a topic (not too long, not too short, contains letters)
            if 10 < len(line) < 150 and line[0].isupper():
                topics_found.append(line)
    
    # Also extract from question papers - look for question patterns
    question_patterns = []
    # Match questions like "Q1. What is...", "Question 1:", etc.
    for line in text_lines:
        line = line.strip()
        if re.match(r'^(?:Q|Question)\s*\d+[\.\):]?\s*(.+)', line, re.IGNORECASE):
            match = re.match(r'^(?:Q|Question)\s*\d+[\.\):]?\s*(.+)', line, re.IGNORECASE)
            if match and len(match.group(1)) > 10:
                question_patterns.append(match.group(1)[:100])
    
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
        
    # Skip if too short or too long
    if len(topic) < 3 or len(topic) > 150:
        return None
        
    return topic

def identify_subjects(text):
    """Identify subjects and topics from text - extract actual topics from syllabus"""
    # First try to extract structured topics from syllabus
    syllabus_topics, question_patterns = extract_topics_from_syllabus(text)
    
    # Subject keywords for categorization
    # Updated Subject keywords - strictly non-overlapping
    subject_keywords = {
        'Mathematics': ['algebra', 'trigonometry', 'differential', 'integral', 'matrix', 'matrices', 'eigen', 'vector', 'theorem', 'laplace', 'fourier', 'calculus'],
        'Reasoning': ['seating arrangement', 'blood relation', 'coding decoding', 'syllogism', 'puzzles', 'data sufficiency'],
        'English': ['synonym', 'antonym', 'comprehension', 'vocabulary', 'grammar', 'essay', 'precis'],
        'General Knowledge': ['current affairs', 'history of india', 'indian polity', 'indian economy', 'geography of india'],
        'Quantitative Aptitude': ['profit and loss', 'simple interest', 'compound interest', 'time and work', 'speed and distance'],
        
        # Engineering - Specific Technical Terms
        'Electrical Engineering': ['kirchhoff', 'thevenin', 'norton', 'superposition', 'transformer', 'induction motor', 'synchronous', 'generator', 'transmission line', 'switchgear', 'power system', 'circuit breaker'],
        'Civil Engineering': ['concrete', 'cement', 'soil mechanics', 'fluid mechanics', 'surveying', 'structural analysis', 'reinforced', 'beam', 'column', 'foundation', 'hydrology'],
        'Mechanical Engineering': ['thermodynamics', 'rankine', 'otto cycle', 'diesel cycle', 'refrigeration', 'fluid dynamics', 'manufacturing', 'welding', 'casting', 'gears'],
        'Computer Science': ['algorithm', 'data structure', 'operating system', 'database', 'sql', 'compiler', 'network', 'protocol', 'stack', 'queue', 'linked list'],
        'Electronics Engineering': ['semiconductor', 'transistor', 'op-amp', 'oscillator', 'microprocessor', 'digital logic', 'embedded system', 'vlsi', 'analog circuit']
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
    if '?' in topic or topic.lower().startswith(('what', 'define', 'explain', 'describe', 'compare')):
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
    
    # If no topics found, extract keywords from all papers
    if not all_topics:
        all_text = ' '.join([paper.get('text', '') for paper in question_papers])
        if all_text:
            keywords = extract_keywords_from_text(all_text)
            all_topics['General'] = dict(list(keywords.items())[:30])
    
    return all_topics, difficulty_distribution

def calculate_prediction_probability(topic_count, total_papers, syllabus_match):
    """Calculate probability of a topic appearing in the exam"""
    frequency_score = (topic_count / total_papers) * 100
    syllabus_score = syllabus_match * 100
    
    # Weighted average (60% frequency, 40% syllabus match)
    probability = (frequency_score * 0.6 + syllabus_score * 0.4)
    return min(probability, 99)  # Cap at 99%

def generate_predictions(syllabus_topics, question_paper_topics, num_papers):
    """Generate predicted question topics based on analysis - focuses on matching syllabus"""
    predictions = []
    
    # Priority: If syllabus exists, prioritize topics that are in syllabus
    syllabus_topic_list = {}
    if syllabus_topics:
        for subject, topic_list in syllabus_topics.items():
            if isinstance(topic_list, list):
                syllabus_topic_list[subject] = [t.get('topic', '').lower() if isinstance(t, dict) else str(t).lower() for t in topic_list]
            else:
                syllabus_topic_list[subject] = []
    
    if not question_paper_topics or len(question_paper_topics) == 0:
        # If no QP topics but syllabus exists, use syllabus topics
        if syllabus_topics:
            for subject, topic_list in syllabus_topics.items():
                if isinstance(topic_list, list):
                    for topic_item in topic_list[:20]:
                        topic_name = topic_item.get('topic', '') if isinstance(topic_item, dict) else str(topic_item)
                        if topic_name:
                            predictions.append({
                                'subject': subject,
                                'topic': topic_name.title(),
                                'question': format_question(topic_name, 1),
                                'frequency': topic_item.get('count', 1) if isinstance(topic_item, dict) else 1,
                                'probability': 85.0,  # High probability for syllabus topics
                                'question_type': 'MCQ (1 mark)'
                            })
        return predictions[:20]
    
    # Generate predictions from question paper topics, prioritizing syllabus matches
    for subject in question_paper_topics:
        if not question_paper_topics[subject] or len(question_paper_topics[subject]) == 0:
            continue
        
        # Get syllabus topics for this subject
        syllabus_subject_topics = syllabus_topic_list.get(subject, [])
        
        if isinstance(question_paper_topics[subject], dict):
            topics_dict = question_paper_topics[subject]
        elif isinstance(question_paper_topics[subject], list):
            # Convert list to dict
            topics_dict = {}
            for item in question_paper_topics[subject]:
                if isinstance(item, dict):
                    topic_name = item.get('topic', '')
                    count = item.get('count', 1)
                    topics_dict[topic_name] = count
                else:
                    topics_dict[str(item)] = 1
        else:
            continue
            
        for topic, count in topics_dict.items():
            if not topic:
                continue
                
            # Check if topic is in syllabus
            topic_lower = str(topic).lower()
            syllabus_match = 0.0
            
            # Check exact match
            if topic_lower in syllabus_subject_topics:
                syllabus_match = 1.0
            # Check partial match
            elif syllabus_subject_topics:
                for syl_topic in syllabus_subject_topics:
                    if topic_lower in syl_topic or syl_topic in topic_lower:
                        syllabus_match = 0.8
                        break
            
            # Calculate probability
            if syllabus_match > 0:
                # High probability if in syllabus
                probability = 70.0 + (syllabus_match * 20) + min(count * 2, 10)
            else:
                # Lower probability if not in syllabus but appeared in QP
                probability = 50.0 + min(count * 3, 20)
            
            probability = min(round(probability, 1), 99.0)
            
            # Include all topics, not just high probability ones
            clean_topic = clean_topic_name(topic)
            if not clean_topic:
                continue
                
            predictions.append({
                'subject': subject,
                'topic': clean_topic.title(),
                'question': format_question(clean_topic, count),
                'frequency': count,
                'probability': probability,
                'question_type': 'Descriptive' if probability > 70 else 'Short Answer'
            })
    
    # Sort by probability (syllabus matches first)
    predictions.sort(key=lambda x: (x['probability'], x['frequency']), reverse=True)
    
    # Return top 30 predictions
    return predictions[:30]

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
        subjects, topics, questions = identify_subjects(text)
        
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
            subjects, topics, questions = identify_subjects(text)
            
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
    """Perform comprehensive analysis of uploaded data - works with syllabus OR question papers OR both"""
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
            'topic_frequency': topic_frequency[:10] if topic_frequency else [],
            'difficulty_distribution': difficulty_dist,
            'total_papers_analyzed': len(analysis_data['question_papers']) if has_question_papers else 0,
            'has_syllabus': has_syllabus,
            'has_question_papers': has_question_papers
        }
    }), 200

@app.route('/api/predict', methods=['POST'])
def predict_questions():
    """Generate predicted question paper - works with syllabus OR question papers OR both"""
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
    
    # Debug info
    print(f"Generating predictions - Syllabus: {has_syllabus}, QP: {has_question_papers}")
    print(f"Syllabus topics: {list(syllabus_topics.keys())}")
    print(f"Question paper topics: {list(question_paper_topics.keys())}")
    
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
                        'question_type': 'MCQ (1 mark)'
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
                            'question_type': 'MCQ (1 mark)'
                        })
        
        predictions.sort(key=lambda x: x.get('probability', 0), reverse=True)
        predictions = predictions[:20]
    
    print(f"Generated {len(predictions)} predictions")
    
    # Save predictions
    prediction_file = f"predictions/prediction_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    try:
        with open(prediction_file, 'w') as f:
            json.dump(predictions, f, indent=2)
    except Exception as e:
        print(f"Error saving prediction file: {e}")
    
    return jsonify({
        'success': True,
        'predictions': predictions,
        'total_predictions': len(predictions),
        'papers_analyzed': num_papers,
        'message': f'Generated {len(predictions)} predictions'
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
