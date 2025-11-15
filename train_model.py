import json
import argparse
import sys
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score
import pickle


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


def check_data_leakage(train_prompts, test_prompts):
    """Check for data leakage between train and test sets"""
    train_set = set(train_prompts)
    test_set = set(test_prompts)
    overlap = train_set.intersection(test_set)
    
    if overlap:
        print(f"\n⚠ WARNING: {len(overlap)} duplicate samples found between train and test!")
        print("Sample duplicates:")
        for i, sample in enumerate(list(overlap)[:3]):
            print(f"  {i+1}. {sample[:60]}...")
        return True
    return False


def save_model(model, vectorizer, model_path, vectorizer_path):
    """Save model and vectorizer with error handling"""
    try:
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        with open(vectorizer_path, 'wb') as f:
            pickle.dump(vectorizer, f)
        print(f"\n✓ Model saved as '{model_path}'")
        print(f"✓ Vectorizer saved as '{vectorizer_path}'")
        return True
    except Exception as e:
        print(f"Error saving model: {e}")
        return False


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Train intent classification model')
    parser.add_argument('--train', default='train.json', help='Training data file')
    parser.add_argument('--test', default='test.json', help='Test data file')
    parser.add_argument('--model', default='intent_model.pkl', help='Output model file')
    parser.add_argument('--vectorizer', default='vectorizer.pkl', help='Output vectorizer file')
    parser.add_argument('--cv-folds', type=int, default=5, help='Cross-validation folds')
    args = parser.parse_args()
    
    # Load training data
    print("Loading training data...")
    train_data = load_json_data(args.train)
    
    # Load test data
    print("Loading test data...")
    test_data = load_json_data(args.test)

    # Extract features and labels
    try:
        train_prompts = [item['prompt'] for item in train_data]
        train_labels = [item['intent'] for item in train_data]
        
        test_prompts = [item['prompt'] for item in test_data]
        test_labels = [item['intent'] for item in test_data]
    except KeyError as e:
        print(f"Error: Missing required field {e} in data")
        sys.exit(1)
    
    print(f"\nTraining samples: {len(train_prompts)}")
    print(f"Test samples: {len(test_prompts)}")
    
    # Check for data leakage
    check_data_leakage(train_prompts, test_prompts)

    # Vectorize text using TF-IDF
    print("\nVectorizing text...")
    vectorizer = TfidfVectorizer(
        max_features=1000,
        ngram_range=(1, 3),  # Use unigrams, bigrams, and trigrams
        lowercase=True
    )
    
    X_train = vectorizer.fit_transform(train_prompts)
    X_test = vectorizer.transform(test_prompts)
    
    # Train model
    print("Training model...")
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train, train_labels)
    
    # Cross-validation
    print(f"\nPerforming {args.cv_folds}-fold cross-validation...")
    cv_scores = cross_val_score(model, X_train, train_labels, cv=args.cv_folds, scoring='accuracy')
    print(f"CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    
    # Predictions on test set only
    print("\nMaking predictions on test set...")
    test_predictions = model.predict(X_test)
    
    # Evaluate
    test_accuracy = accuracy_score(test_labels, test_predictions)

    print(f"\n{'='*50}")
    print(f"RESULTS")
    print(f"{'='*50}")
    print(f"Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
    
    # Warning for suspiciously high accuracy
    if test_accuracy >= 0.99:
        print("\n⚠ WARNING: Accuracy ≥99% may indicate:")
        print("  - Task is too simple")
        print("  - Possible data leakage")
        print("  - Test set not diverse enough")
    
    print(f"\n{'='*50}")
    print(f"DETAILED TEST REPORT")
    print(f"{'='*50}")
    print(classification_report(test_labels, test_predictions))
    
    print(f"\nConfusion Matrix:")
    cm = confusion_matrix(test_labels, test_predictions)
    print(cm)
    
    # Show misclassified examples
    misclassified = [(i, test_prompts[i], test_labels[i], test_predictions[i]) 
                     for i in range(len(test_data)) 
                     if test_labels[i] != test_predictions[i]]
    
    if misclassified:
        print(f"\n{'='*50}")
        print(f"MISCLASSIFIED EXAMPLES ({len(misclassified)} total)")
        print(f"{'='*50}")
        for i, prompt, actual, predicted in misclassified[:10]:
            print(f"✗ '{prompt[:60]}...'")
            print(f"  Actual: {actual}, Predicted: {predicted}\n")
    else:
        print(f"\n✓ No misclassified examples!")
    
    # Show correct sample predictions
    correct = [(i, test_prompts[i], test_labels[i]) 
               for i in range(len(test_data)) 
               if test_labels[i] == test_predictions[i]]
    
    if correct:
        print(f"\n{'='*50}")
        print(f"SAMPLE CORRECT PREDICTIONS")
        print(f"{'='*50}")
        for i, prompt, label in correct[:5]:
            print(f"✓ '{prompt[:60]}...'")
            print(f"  Intent: {label}\n")
    
    # Save model and vectorizer
    print("Saving model...")
    save_model(model, vectorizer, args.model, args.vectorizer)
    
    print("\nDone!")


if __name__ == '__main__':
    main()
