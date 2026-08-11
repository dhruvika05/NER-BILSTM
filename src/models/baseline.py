from datasets import load_dataset
from collections import Counter, defaultdict

def train_baseline():
    print("Loading dataset for Baseline...")
    dataset = load_dataset("lhoestq/conll2003")
    train_data = dataset['train']
    test_data = dataset['test']
    
    print("Training Baseline (Memorizing words)...")
    # Dictionary to hold word -> list of tags it was seen with
    word_tag_counts = defaultdict(Counter)
    
    for example in train_data:
        for word, tag in zip(example['tokens'], example['ner_tags']):
            word_tag_counts[word.lower()][tag] += 1
            
    # Figure out the most common tag for each word
    memorized_model = {}
    for word, tag_counts in word_tag_counts.items():
        # .most_common(1) returns e.g. [(tag_id, count)]
        best_tag = tag_counts.most_common(1)[0][0]
        memorized_model[word] = best_tag
        
    print(f"Memorized {len(memorized_model)} unique words!")
    
    # Let's test it!
    print("Evaluating Baseline on Test Set...")
    correct = 0
    total = 0
    
    for example in test_data:
        for word, true_tag in zip(example['tokens'], example['ner_tags']):
            # Look up the word. If we don't know it, guess 0 ("O")
            predicted_tag = memorized_model.get(word.lower(), 0)
            
            if predicted_tag == true_tag:
                correct += 1
            total += 1
            
    accuracy = (correct / total) * 100
    print(f"\n=== Baseline Results ===")
    print(f"Total Words Tested : {total}")
    print(f"Baseline Accuracy  : {accuracy:.2f}%")
    print("Note: This accuracy is inflated because it gets all the 'O' tags right easily!")

if __name__ == "__main__":
    train_baseline()