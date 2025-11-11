import json
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pickle

# Load training data
print("Loading training data...")
with open('train.json', 'r', encoding='utf-8') as f:
    train_data = json.load(f)

# Load test data
print("Loading test data...")
with open('test.json', 'r', encoding='utf-8') as f:
    test_data = json.load(f)

# Extract features and labels
train_prompts = [item['prompt'] for item in train_data]
train_labels = [item['intent'] for item in train_data]

test_prompts = [item['prompt'] for item in test_data]
test_labels = [item['intent'] for item in test_data]

print(f"\nTraining samples: {len(train_prompts)}")
print(f"Test samples: {len(test_prompts)}")

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

# Predictions
print("\nMaking predictions...")
train_predictions = model.predict(X_train)
test_predictions = model.predict(X_test)

# Evaluate
train_accuracy = accuracy_score(train_labels, train_predictions)
test_accuracy = accuracy_score(test_labels, test_predictions)

print(f"\n{'='*50}")
print(f"RESULTS")
print(f"{'='*50}")
print(f"Training Accuracy: {train_accuracy:.4f} ({train_accuracy*100:.2f}%)")
print(f"Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")

print(f"\n{'='*50}")
print(f"DETAILED TEST REPORT")
print(f"{'='*50}")
print(classification_report(test_labels, test_predictions))

print(f"Confusion Matrix:")
cm = confusion_matrix(test_labels, test_predictions)
print(cm)

# Show some example predictions
print(f"\n{'='*50}")
print(f"SAMPLE PREDICTIONS")
print(f"{'='*50}")
for i in range(min(10, len(test_data))):
    actual = test_labels[i]
    predicted = test_predictions[i]
    status = "✓" if actual == predicted else "✗"
    print(f"{status} '{test_prompts[i][:60]}...'")
    print(f"  Actual: {actual}, Predicted: {predicted}\n")

# Save model and vectorizer
print("Saving model...")
with open('intent_model.pkl', 'wb') as f:
    pickle.dump(model, f)
    
with open('vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)

print("\n✓ Model saved as 'intent_model.pkl'")
print("✓ Vectorizer saved as 'vectorizer.pkl'")
print("\nDone!")
