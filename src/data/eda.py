from datasets import load_dataset
from collections import Counter
import matplotlib.pyplot as plt

def run_eda():
    print("Loading dataset for EDA...")
    dataset = load_dataset("lhoestq/conll2003")
    train_data = dataset['train']
    
    # 1. Analyze Sentence Lengths
    sentence_lengths = [len(example['tokens']) for example in train_data]
    max_len = max(sentence_lengths)
    avg_len = sum(sentence_lengths) / len(sentence_lengths)
    
    print("\n=== Sentence Length Statistics ===")
    print(f"Longest sentence : {max_len} words")
    print(f"Average sentence : {avg_len:.2f} words")
    
    # 2. Analyze Label Distribution
    all_tags = []
    for example in train_data:
        all_tags.extend(example['ner_tags'])
        
    tag_counts = Counter(all_tags)
    total_tags = sum(tag_counts.values())
    
    # Standard CoNLL-2003 Mapping
    tag_map = {0: "O", 1: "B-PER", 2: "I-PER", 3: "B-ORG", 4: "I-ORG", 
               5: "B-LOC", 6: "I-LOC", 7: "B-MISC", 8: "I-MISC"}
    
    print("\n=== NER Tag Distribution ===")
    for tag_id, count in sorted(tag_counts.items()):
        percentage = (count / total_tags) * 100
        print(f"{tag_map[tag_id]:<7} : {count:>7} ({percentage:.2f}%)")

if __name__ == "__main__":
    run_eda()