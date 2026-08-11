from datasets import load_dataset
import os

def download_and_inspect_dataset():
    print("Fetching CoNLL-2003 dataset from HuggingFace...")
    
    # Load the CoNLL-2003 dataset
    # "eriktks/conll2003" is the standard public repository for CoNLL-2003
    dataset = load_dataset("lhoestq/conll2003")
    
    print("\n=== Dataset Successfully Loaded! ===")
    print(f"Train split size      : {len(dataset['train'])} sentences")
    print(f"Validation split size : {len(dataset['validation'])} sentences")
    print(f"Test split size       : {len(dataset['test'])} sentences")
    
    # Inspect a single example from the training set
    sample = dataset['train'][0]
    
    print("\n=== Inspecting Example #0 ===")
    print(f"Tokens (words)   : {sample['tokens']}")
    print(f"POS Tags         : {sample['pos_tags']}")
    print(f"Chunk Tags       : {sample['chunk_tags']}")
    print(f"NER Tags (IDs)   : {sample['ner_tags']}")
    
    # Define the standard CoNLL-2003 NER tags since the Parquet version drops the metadata
    ner_tags_mapping = [
        "O", "B-PER", "I-PER", "B-ORG", "I-ORG", 
        "B-LOC", "I-LOC", "B-MISC", "I-MISC"
    ]
    
    print("\n=== NER Tag ID Mapping ===")
    for id, label in enumerate(ner_tags_mapping):
        print(f"ID {id} -> {label}")
        
    return dataset

if __name__ == "__main__":
    download_and_inspect_dataset()