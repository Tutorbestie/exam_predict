import app
from app import analyze_question_patterns, identify_subjects, extract_text, is_valid_topic
import sys
import traceback

print("Starting debug script...")

# 1. Test basic text cleaning/extraction functions
try:
    print("Testing is_valid_topic...")
    # New behavior: should be TRUE
    assert is_valid_topic("Write a program to sort") == True 
    assert is_valid_topic("Explain the theory") == True
    print("is_valid_topic check passed (confirming FIX).")
except Exception as e:
    print(f"is_valid_topic failed: {e}")
    traceback.print_exc()

# Test alphanumeric cleaning
try:
    print("Testing ML alphanumeric support...")
    from ml_predictor import ExamPredictor
    ep = ExamPredictor()
    text = "Comparison of 3G and 4G technology"
    cleaned = ep.preprocess_text(text)
    print(f"Cleaned text: '{cleaned}'")
    assert "3g" in cleaned and "4g" in cleaned
    print("Preprocess check passed.")
except Exception as e:
    print(f"ML Preprocess failed: {e}")
    traceback.print_exc()

# 2. Test Analysis Logic (Mocking data)
try:
    print("Testing analyze_question_patterns...")
    mock_papers = [{'text': "Q1. What is Thermodynamics? Explain the laws of thermodynamics. (10 marks)"}]
    topics, difficulty = analyze_question_patterns(mock_papers)
    print(f"Analysis result: Topics={list(topics.keys())}, Difficulty={difficulty}")
except Exception as e:
    print("CRITICAL: analyze_question_patterns crashed!")
    traceback.print_exc()

# 3. Test ML Predictor directly
try:
    print("Testing ML Predictor...")
    from ml_predictor import ExamPredictor
    ep = ExamPredictor()
    dist = ep.get_difficulty_distribution("some random text with technical terms like algorithm and thermodynamics")
    print(f"ML Difficulty: {dist}")
except Exception as e:
    print("CRITICAL: ML Predictor crashed!")
    traceback.print_exc()

print("Debug script finished.")
