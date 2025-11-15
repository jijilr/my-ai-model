"""Shared utilities for the intent classification project"""

import json
import pickle
import sys
from pathlib import Path


def load_json_data(filepath):
    """Load JSON data with error handling"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Error: File '{filepath}' not found.")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in '{filepath}': {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Error loading '{filepath}': {e}")
        sys.exit(1)


def save_json_data(data, filepath):
    """Save data as JSON with error handling"""
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        return True
    except Exception as e:
        print(f"Error saving '{filepath}': {e}")
        return False


def load_model_components(model_path, vectorizer_path):
    """Load model and vectorizer with error handling"""
    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        with open(vectorizer_path, 'rb') as f:
            vectorizer = pickle.load(f)
        return model, vectorizer
    except FileNotFoundError as e:
        print(f"Error: Model file not found. Please train the model first.")
        print(f"Missing file: {e.filename}")
        sys.exit(1)
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)


def save_model_components(model, vectorizer, model_path, vectorizer_path):
    """Save model and vectorizer with error handling"""
    try:
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        with open(vectorizer_path, 'wb') as f:
            pickle.dump(vectorizer, f)
        return True
    except Exception as e:
        print(f"Error saving model: {e}")
        return False


def validate_data_structure(data, required_fields=['prompt', 'intent', 'operands', 'result']):
    """Validate that data has required structure"""
    if not isinstance(data, list):
        return False, "Data must be a list"
    
    for i, item in enumerate(data):
        if not isinstance(item, dict):
            return False, f"Item {i} is not a dictionary"
        
        missing = [field for field in required_fields if field not in item]
        if missing:
            return False, f"Item {i} missing fields: {missing}"
    
    return True, "Valid"


def check_data_leakage(train_data, test_data):
    """Check for overlapping samples between train and test"""
    train_prompts = set(item['prompt'] for item in train_data)
    test_prompts = set(item['prompt'] for item in test_data)
    overlap = train_prompts.intersection(test_prompts)
    
    return list(overlap)
