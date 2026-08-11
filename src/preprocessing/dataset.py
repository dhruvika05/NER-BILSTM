import torch
from torch.utils.data import Dataset
import json

# Define the standard CoNLL-2003 Tag Mapping
TAG2IDX = {
    "O": 0,
    "B-PER": 1, "I-PER": 2,
    "B-ORG": 3, "I-ORG": 4,
    "B-LOC": 5, "I-LOC": 6,
    "B-MISC": 7, "I-MISC": 8
}

IDX2TAG = {v: k for k, v in TAG2IDX.items()}

class NERDataset(Dataset):
    def __init__(self, hf_dataset_split, word2idx, max_len=50):
        """
        Custom PyTorch Dataset for NER.
        
        Args:
            hf_dataset_split: HuggingFace dataset split (e.g. dataset['train'])
            word2idx: Dictionary mapping words -> integer IDs
            max_len: Maximum length to pad/truncate sentences to
        """
        self.data = hf_dataset_split
        self.word2idx = word2idx
        self.max_len = max_len

    def __len__(self):
        # Returns total number of sentences in this split
        return len(self.data)

    def __getitem__(self, idx):
        # 1. Grab a single sentence example
        item = self.data[idx]
        raw_tokens = item['tokens']
        raw_ner_tags = item['ner_tags']

        # 2. Convert words -> Word IDs using our vocab dictionary
        # If a word isn't in our dictionary, use the <UNK> ID (1)
        word_ids = []
        for token in raw_tokens:
            word = token.lower()
            word_id = self.word2idx.get(word, self.word2idx["<UNK>"])
            word_ids.append(word_id)

        # 3. Truncate if the sentence is longer than max_len
        word_ids = word_ids[:self.max_len]
        tag_ids = raw_ner_tags[:self.max_len]

        # 4. Create the Mask (1 for real tokens, 0 for padding)
        mask = [1] * len(word_ids)

        # 5. Apply Padding if sentence is shorter than max_len
        pad_len = self.max_len - len(word_ids)
        if pad_len > 0:
            # Pad word IDs with <PAD> token ID (0)
            word_ids += [self.word2idx["<PAD>"]] * pad_len
            # Pad tag IDs with 0 ("O" tag)
            tag_ids += [TAG2IDX["O"]] * pad_len
            # Pad mask with 0 (indicates padding)
            mask += [0] * pad_len

        # 6. Convert lists into PyTorch Tensors
        return {
            "input_ids": torch.tensor(word_ids, dtype=torch.long),
            "tag_ids": torch.tensor(tag_ids, dtype=torch.long),
            "mask": torch.tensor(mask, dtype=torch.uint8)
        }


# Quick test script inside the same file
if __name__ == "__main__":
    from datasets import load_dataset
    
    # Load vocabulary we created in Step 5
    with open("data/word2idx.json", "r") as f:
        word2idx = json.load(f)
        
    dataset = load_dataset("lhoestq/conll2003")
    train_dataset = NERDataset(dataset['train'], word2idx, max_len=15)
    
    sample = train_dataset[0]
    print("\n=== Sample Processed Tensor Output (Example #0) ===")
    print("input_ids (Word IDs) :", sample['input_ids'])
    print("tag_ids   (Target IDs):", sample['tag_ids'])
    print("mask      (1=real,0=pad):", sample['mask'])
    print("Tensor Shape          :", sample['input_ids'].shape)