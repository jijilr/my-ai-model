import pickle
import json

# Load model and vectorizer
print("Loading model...")
with open('intent_model.pkl', 'rb') as f:
    model = pickle.load(f)
    
with open('vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

print("Model loaded successfully!\n")

def predict_intent(prompt):
    """Predict the intent (add/subtract) for a given prompt"""
    X = vectorizer.transform([prompt])
    prediction = model.predict(X)[0]
    probability = model.predict_proba(X)[0]
    
    return {
        'intent': prediction,
        'confidence': max(probability) * 100
    }

# Interactive mode
print("="*50)
print("MATH INTENT CLASSIFIER")
print("="*50)
print("Enter math word problems to classify as 'add' or 'subtract'")
print("Type 'quit' to exit\n")

while True:
    user_input = input("Enter problem: ").strip()
    
    if user_input.lower() in ['quit', 'exit', 'q']:
        print("\nGoodbye!")
        break
    
    if not user_input:
        continue
    
    result = predict_intent(user_input)
    print(f"  → Intent: {result['intent']}")
    print(f"  → Confidence: {result['confidence']:.2f}%\n")
