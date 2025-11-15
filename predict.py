import pickle
import argparse
import sys
from pathlib import Path


def load_model(model_path, vectorizer_path):
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


def predict_intent(model, vectorizer, prompt):
    """Predict the intent (add/subtract) for a given prompt"""
    if not prompt or not prompt.strip():
        return None
    
    try:
        X = vectorizer.transform([prompt])
        prediction = model.predict(X)[0]
        probability = model.predict_proba(X)[0]
        
        return {
            'intent': prediction,
            'confidence': max(probability) * 100
        }
    except Exception as e:
        print(f"Error making prediction: {e}")
        return None


def batch_predict(model, vectorizer, prompts):
    """Predict intents for multiple prompts"""
    results = []
    for prompt in prompts:
        result = predict_intent(model, vectorizer, prompt)
        if result:
            results.append({**result, 'prompt': prompt})
    return results


def interactive_mode(model, vectorizer):
    """Run interactive prediction mode"""
    print("="*50)
    print("MATH INTENT CLASSIFIER")
    print("="*50)
    print("Enter math word problems to classify as 'add' or 'subtract'")
    print("Type 'quit' to exit\n")
    
    while True:
        try:
            user_input = input("Enter problem: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("\nGoodbye!")
                break
            
            if not user_input:
                continue
            
            result = predict_intent(model, vectorizer, user_input)
            if result:
                print(f"  → Intent: {result['intent']}")
                print(f"  → Confidence: {result['confidence']:.2f}%\n")
        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except EOFError:
            print("\n\nGoodbye!")
            break


def main():
    parser = argparse.ArgumentParser(description='Predict intent for math word problems')
    parser.add_argument('--model', default='intent_model.pkl', help='Model file path')
    parser.add_argument('--vectorizer', default='vectorizer.pkl', help='Vectorizer file path')
    parser.add_argument('--prompt', type=str, help='Single prompt to classify')
    parser.add_argument('--batch', type=str, help='File with prompts (one per line)')
    args = parser.parse_args()
    
    # Load model and vectorizer
    print("Loading model...")
    model, vectorizer = load_model(args.model, args.vectorizer)
    print("Model loaded successfully!\n")
    
    # Single prompt mode
    if args.prompt:
        result = predict_intent(model, vectorizer, args.prompt)
        if result:
            print(f"Prompt: {args.prompt}")
            print(f"Intent: {result['intent']}")
            print(f"Confidence: {result['confidence']:.2f}%")
        return
    
    # Batch mode
    if args.batch:
        try:
            with open(args.batch, 'r', encoding='utf-8') as f:
                prompts = [line.strip() for line in f if line.strip()]
            
            print(f"Processing {len(prompts)} prompts...\n")
            results = batch_predict(model, vectorizer, prompts)
            
            for r in results:
                print(f"Prompt: {r['prompt']}")
                print(f"  → Intent: {r['intent']} (Confidence: {r['confidence']:.2f}%)\n")
            return
        except FileNotFoundError:
            print(f"Error: Batch file '{args.batch}' not found.")
            sys.exit(1)
    
    # Interactive mode (default)
    interactive_mode(model, vectorizer)


if __name__ == '__main__':
    main()
