from datasets import load_dataset
from collections import Counter
import json
import os

def build_vocab(min_freq=2):
    print("Loading dataset to build vocabulary...")
    dataset = load_dataset("lhoestq/conll2003")
    train_data = dataset['train']
    
    # 1. Count all words in the training set
    print("Counting words...")
    word_counts = Counter()
    for example in train_data:
        # We convert words to lowercase to reduce vocabulary size. 
        # (e.g., "The" and "the" become the same word)
        words = [word.lower() for word in example['tokens']]
        word_counts.update(words)
        
    print(f"Total unique words found: {len(word_counts)}")
    
    # 2. Create the Vocabulary mappings
    # Start with our special tokens!
    word2idx = {"<PAD>": 0, "<UNK>": 1}
    
    # 3. Add words that appear at least 'min_freq' times
    # We ignore words that only appear once because they might be typos,
    # and it forces the model to learn how to handle the <UNK> token.
    idx = 2
    for word, count in word_counts.items():
        if count >= min_freq:
            word2idx[word] = idx
            idx += 1
            
    print(f"Final Vocabulary size (words appearing >={min_freq} times): {len(word2idx)}")
    
    # Create the reverse mapping (so we can turn IDs back into words later)
    idx2word = {i: w for w, i in word2idx.items()}
    
    # 4. Save the vocabulary to disk so we can use it during inference/deployment
    os.makedirs("data", exist_ok=True)
    with open("data/word2idx.json", "w") as f:
        json.dump(word2idx, f)
        
    print("Vocabulary successfully saved to data/word2idx.json!")
    
    return word2idx, idx2word

if __name__ == "__main__":
    build_vocab()