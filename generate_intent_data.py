#!/usr/bin/env python3
"""
Generate synthetic training and testing data for intent detection.
Produces verbose sentences about money transactions with add/subtract intents.
"""

import argparse
import json
import random
from typing import List, Dict, Tuple, Any


# Names by faith (no interfaith pairs)
NAMES = {
    'hindu': {
        'male': ['Arjun', 'Rohan', 'Vikram', 'Aryan', 'Karan', 'Dev', 'Sameer', 'Raj', 'Amit', 'Mahesh'],
        'female': ['Priya', 'Lakshmi', 'Radha', 'Sita', 'Anjali', 'Pooja', 'Divya', 'Meera', 'Riya', 'Kavya']
    },
    'muslim': {
        'male': ['Ahmed', 'Faisal', 'Omar', 'Bilal', 'Yusuf', 'Hassan', 'Tariq', 'Nadeem', 'Salman', 'Zain'],
        'female': ['Fatima', 'Ayesha', 'Sara', 'Mariam', 'Layla', 'Noor', 'Hina', 'Sana', 'Zara', 'Rubina']
    },
    'christian': {
        'male': ['John', 'David', 'Michael', 'James', 'Peter', 'Matthew', 'Luke', 'Paul', 'Mark', 'Thomas'],
        'female': ['Mary', 'Sarah', 'Elizabeth', 'Anna', 'Grace', 'Ruth', 'Hannah', 'Rebecca', 'Esther', 'Lydia']
    }
}

# Relationships (same-faith only, no LGBT)
RELATIONSHIPS = {
    'male': ['father', 'friend', 'son', 'teacher', 'brother', 'boyfriend'],
    'female': ['friend', 'girlfriend', 'mother', 'wife', 'sister', 'teacher']
}

# Objects for context
OBJECTS = ['candies', 'cake', 'books']

# Positive start verbs (REQUIRED for every sentence)
POSITIVE_START_VERBS = ['had', 'possessed', 'saved', 'held']

# Transaction verbs
ADD_VERBS = ['gained', 'received', 'added', 'got', 'obtained', 'found', '+']
SUBTRACT_VERBS = ['lost', 'gave away', 'spent', 'donated', 'sacrificed', '-']

# Sentence templates - Enforcing clear flow pattern
SUBTRACTION_TEMPLATES = [
    "{name} {positive_verb} {amt1}$ {positive_context} but then {subtract_verb} {amt2}$ {reason}.",
    "My {rel} {name} {positive_verb} {amt1}$ {positive_context} but then {subtract_verb} {amt2}$ {reason}.",
    "{name}, who works as a {occupation}, {positive_verb} {amt1}$ {positive_context} but then {subtract_verb} {amt2}$ {reason}.",
]

ADDITION_TWO_OPERAND_TEMPLATES = [
    "{name} {positive_verb} {amt1}$ {context} and also {add_verb} {amt2}$ {source}.",
    "My {rel} {name} {positive_verb} {amt1}$ {context} and also {add_verb} {amt2}$ {source}.",
    "{name}, who works as a {occupation}, {positive_verb} {amt1}$ {context} and also {add_verb} {amt2}$ {source}.",
]

SINGLE_OPERAND_TEMPLATES = [
    "{name} {positive_verb} {amt1}$ {context}.",
    "My {rel} {name} {positive_verb} {amt1}$ {context}.",
    "{name}, who works as a {occupation}, {positive_verb} {amt1}$ {context}.",
]

MULTI_OPERAND_TEMPLATES = [
    "{name1} {positive_verb} {amt1}$, {name2} {positive_verb2} {amt2}$, and {name3} {positive_verb3} {amt3}$ {context}.",
]

# Context phrases
POSITIVE_CONTEXTS = [
    "from his teaching job", "in his wallet", "in her savings", "from his work", "from her business", 
    "in cash", "from his father", "from her mother", "after selling {obj}", "from teaching", "from the market"
]

SUBTRACTION_REASONS = [
    "on {obj}", "while shopping", "during the trip", "for repairs", 
    "as a donation", "for {obj}", "to charity"
]

ADDITION_SOURCES = [
    "from a bonus", "from the lottery", "as an extra", "from a side job",
    "by selling {obj}", "from teaching", "from the market"
]

GENERAL_CONTEXTS = [
    "in his pocket", "in her bag", "from work", "from business", 
    "after the sale", "as savings", "from teaching", "from the market"
]

OCCUPATIONS = ['teacher', 'doctor', 'engineer', 'artist', 'farmer', 'student', 'manager', 'driver']


def detect_intent(sentence: str, operands: List[int]) -> str:
    """
    Detect intent from sentence and operands.
    
    Rules:
    - If exactly 2 operands and subtract cue: return "a - b"
    - Otherwise: return "sum: a, b, ..."
    """
    sentence_lower = sentence.lower()
    
    # Check for subtract cues
    has_subtract = any(verb in sentence_lower for verb in SUBTRACT_VERBS)
    
    # Check for add cues
    has_add = any(verb in sentence_lower for verb in ADD_VERBS)
    
    # Intent logic - subtraction only if exactly 2 operands and subtract verb, and no add verbs
    if len(operands) == 2 and has_subtract and not has_add:
        # Ensure first operand > second operand for subtraction
        if operands[0] > operands[1]:
            return f"{operands[0]} - {operands[1]}"
        else:
            # If not properly ordered, treat as addition
            return "sum: " + ", ".join(map(str, operands))
    else:
        return "sum: " + ", ".join(map(str, operands))


def get_random_person() -> Tuple[str, Dict]:
    """Get a random person with name, gender, and relationship."""
    faith = random.choice(['hindu', 'muslim', 'christian'])
    gender = random.choice(['male', 'female'])
    name = random.choice(NAMES[faith][gender])
    rel = random.choice(RELATIONSHIPS[gender])
    
    person = {'name': name, 'gender': gender, 'rel': rel, 'faith': faith}
    return faith, person


def generate_two_operand_sentence() -> Dict[str, Any]:
    """Generate a sentence with exactly 2 operands (90% of data)."""
    # Generate two operands - ensure first is larger for subtraction
    amt1 = random.randint(50, 100)  # Higher range for first operand
    amt2 = random.randint(1, 49)    # Lower range for second operand
    
    # Ensure first operand is always larger than second for subtraction cases
    # For addition cases, we still maintain the constraint but it's less critical
    
    # Decide intent (approximately 50/50 split)
    is_subtract = random.random() < 0.5
    
    operands = [amt1, amt2]
    
    # Get person
    faith, person = get_random_person()
    
    # Always start with positive verb
    positive_verb = random.choice(POSITIVE_START_VERBS)
    
    if is_subtract:
        subtract_verb = random.choice(SUBTRACT_VERBS)
        template = random.choice(SUBTRACTION_TEMPLATES)
        
        # Fill subtraction template
        sentence = template.format(
            name=person['name'],
            rel=person['rel'],
            occupation=random.choice(OCCUPATIONS),
            positive_verb=positive_verb,
            subtract_verb=subtract_verb,
            amt1=operands[0],
            amt2=operands[1],
            obj=random.choice(OBJECTS),
            positive_context=random.choice(POSITIVE_CONTEXTS).format(
                obj=random.choice(OBJECTS)
            ),
            reason=random.choice(SUBTRACTION_REASONS).format(
                obj=random.choice(OBJECTS)
            )
        )
    else:
        add_verb = random.choice(ADD_VERBS)
        template = random.choice(ADDITION_TWO_OPERAND_TEMPLATES)
        
        # Fill addition template
        sentence = template.format(
            name=person['name'],
            rel=person['rel'],
            occupation=random.choice(OCCUPATIONS),
            positive_verb=positive_verb,
            add_verb=add_verb,
            amt1=operands[0],
            amt2=operands[1],
            context=random.choice(GENERAL_CONTEXTS),
            source=random.choice(ADDITION_SOURCES).format(
                obj=random.choice(OBJECTS)
            )
        )
    
    return {'prompt': sentence, 'operands': operands}


def generate_single_operand_sentence() -> Dict[str, Any]:
    """Generate a sentence with exactly 1 operand (3% of data)."""
    operands = [random.randint(1, 100)]
    
    # Get person
    faith, person = get_random_person()
    positive_verb = random.choice(POSITIVE_START_VERBS)
    
    template = random.choice(SINGLE_OPERAND_TEMPLATES)
    
    sentence = template.format(
        name=person['name'],
        rel=person['rel'],
        occupation=random.choice(OCCUPATIONS),
        positive_verb=positive_verb,
        amt1=operands[0],
        context=random.choice(GENERAL_CONTEXTS)
    )
    
    return {'prompt': sentence, 'operands': operands}


def generate_multi_operand_sentence() -> Dict[str, Any]:
    """Generate a sentence with 3-5 operands (7% of data)."""
    num_operands = random.randint(3, 5)
    operands = [random.randint(1, 100) for _ in range(num_operands)]
    
    # Get persons (3 different people for 3+ operand sentences)
    faith1, person1 = get_random_person()
    faith2, person2 = get_random_person()
    faith3, person3 = get_random_person()
    
    # Ensure all people are of the same faith
    # If not, adjust person2 and person3 to match person1's faith
    if faith2 != faith1:
        person2['name'] = random.choice(NAMES[faith1][person2['gender']])
        person2['faith'] = faith1
    
    if faith3 != faith1:
        person3['name'] = random.choice(NAMES[faith1][person3['gender']])
        person3['faith'] = faith1
    
    positive_verb = random.choice(POSITIVE_START_VERBS)
    positive_verb2 = random.choice(POSITIVE_START_VERBS)
    positive_verb3 = random.choice(POSITIVE_START_VERBS)
    
    if num_operands == 3:
        template = random.choice(MULTI_OPERAND_TEMPLATES)
        sentence = template.format(
            name1=person1['name'],
            name2=person2['name'],
            name3=person3['name'],
            positive_verb=positive_verb,
            positive_verb2=positive_verb2,
            positive_verb3=positive_verb3,
            amt1=operands[0],
            amt2=operands[1],
            amt3=operands[2],
            context=random.choice(GENERAL_CONTEXTS)
        )
    else:
        # For 4-5 operands, build a custom sentence
        people_names = [person1['name'], person2['name'], person3['name']]
        # Add more names if needed
        while len(people_names) < num_operands:
            _, extra_person = get_random_person()
            # Ensure same faith
            if extra_person['faith'] != faith1:
                extra_person['name'] = random.choice(NAMES[faith1][extra_person['gender']])
            people_names.append(extra_person['name'])
        
        possession_phrases = []
        for i in range(num_operands):
            verb = random.choice(POSITIVE_START_VERBS)
            possession_phrases.append(f"{people_names[i]} {verb} {operands[i]}$")
        
        sentence = ", ".join(possession_phrases[:-1]) + f", and {possession_phrases[-1]} {random.choice(GENERAL_CONTEXTS)}."
    
    return {'prompt': sentence, 'operands': operands}


def generate_dataset(num_samples: int, include_intent: bool = True) -> List[Dict]:
    """Generate dataset with specified distribution of operand counts.
    
    Args:
        num_samples: Number of samples to generate
        include_intent: If True, include 'intent' and 'operands' fields (for training).
                       If False, only include 'prompt' field (for testing).
    """
    dataset = []
    
    # Calculate counts based on distribution
    two_operand_count = int(num_samples * 0.90)
    single_operand_count = int(num_samples * 0.03)
    multi_operand_count = num_samples - two_operand_count - single_operand_count
    
    # Generate samples
    for _ in range(two_operand_count):
        dataset.append(generate_two_operand_sentence())
    
    for _ in range(single_operand_count):
        dataset.append(generate_single_operand_sentence())
    
    for _ in range(multi_operand_count):
        dataset.append(generate_multi_operand_sentence())
    
    # Shuffle to mix operand types
    random.shuffle(dataset)
    
    # Add intent field for training data, or remove fields for testing data
    if include_intent:
        for sample in dataset:
            sample['intent'] = detect_intent(sample['prompt'], sample['operands'])
    else:
        # Testing data: only keep the prompt
        dataset = [{'prompt': sample['prompt']} for sample in dataset]
    
    return dataset


def validate_dataset(dataset: List[Dict]) -> None:
    """Validate that the dataset follows all rules."""
    errors = []
    
    for i, sample in enumerate(dataset):
        prompt = sample['prompt']
        operands = sample.get('operands', [])
        
        # Check 1: Every sentence must contain a positive start verb
        contains_positive_verb = any(verb in prompt for verb in POSITIVE_START_VERBS)
        if not contains_positive_verb:
            errors.append(f"Sample {i}: Does not contain positive start verb: {prompt[:50]}...")
        
        # Check 2: For subtraction cases, first operand > second operand
        if len(operands) == 2 and 'intent' in sample and '-' in sample['intent'] and 'sum:' not in sample['intent']:
            # Extract the numbers from the intent string
            parts = sample['intent'].split(' - ')
            if len(parts) == 2:
                try:
                    a, b = int(parts[0]), int(parts[1])
                    if a <= b:
                        errors.append(f"Sample {i}: Subtraction but first operand <= second: {a} <= {b}")
                except ValueError:
                    errors.append(f"Sample {i}: Invalid subtraction intent format: {sample['intent']}")
        
        # Check 3: No mixed intent cues (only for 2 operand cases)
        sentence_lower = prompt.lower()
        has_subtract = any(verb in sentence_lower for verb in SUBTRACT_VERBS)
        has_add = any(verb in sentence_lower for verb in ADD_VERBS)
        
        # Only check for mixed cues in 2-operand cases
        if len(operands) == 2:
            if has_subtract and has_add:
                errors.append(f"Sample {i}: Mixed intent cues detected: {prompt[:50]}...")
    
    if errors:
        print(f"VALIDATION ERRORS ({len(errors)} found):")
        for error in errors[:10]:  # Show first 10 errors
            print(f"  - {error}")
        if len(errors) > 10:
            print(f"  ... and {len(errors) - 10} more errors")
    else:
        print("✓ Dataset validation passed!")


def main():
    parser = argparse.ArgumentParser(description='Generate synthetic intent detection training data')
    parser.add_argument('--scale', type=int, default=100, choices=range(1, 101),
                        help='Percentage of data to generate (1-100, default: 100)')
    parser.add_argument('--validate', action='store_true', 
                        help='Run validation on generated data')
    args = parser.parse_args()
    
    # Set seed for reproducibility
    random.seed(42)
    
    # Calculate actual counts based on scale
    training_count = int(100000 * args.scale / 100)
    testing_count = int(20000 * args.scale / 100)
    
    print(f"Generating {training_count} training samples...")
    training_data = generate_dataset(training_count, include_intent=True)
    
    print(f"Generating {testing_count} testing samples...")
    testing_data = generate_dataset(testing_count, include_intent=False)
    
    # Run validation if requested
    if args.validate:
        print("\nValidating training data...")
        validate_dataset(training_data)
    
    # Save to JSON files
    with open('training_data.json', 'w') as f:
        json.dump(training_data, f, indent=2)
    print(f"Saved training_data.json ({len(training_data)} samples)")
    
    with open('testing_data.json', 'w') as f:
        json.dump(testing_data, f, indent=2)
    print(f"Saved testing_data.json ({len(testing_data)} samples)")
    
    # Demo: Show intent detection for first 5 training samples
    print("\n=== Intent Detection Demo (First 5 Training Samples) ===")
    for i, sample in enumerate(training_data[:5], 1):
        print(f"\nSample {i}:")
        print(f"  Prompt: {sample['prompt']}")
        print(f"  Operands: {sample['operands']}")
        print(f"  Intent: {sample['intent']}")


if __name__ == '__main__':
    main()