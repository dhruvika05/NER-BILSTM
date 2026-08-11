import torch
from torch.utils.data import Dataset, DataLoader
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
        self.data = hf_dataset_split
        self.word2idx = word2idx
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        raw_tokens = item['tokens']
        raw_ner_tags = item['ner_tags']

        word_ids = []
        for token in raw_tokens:
            word = token.lower()
            word_id = self.word2idx.get(word, self.word2idx["<UNK>"])
            word_ids.append(word_id)

        word_ids = word_ids[:self.max_len]
        tag_ids = raw_ner_tags[:self.max_len]

        mask = [1] * len(word_ids)

        pad_len = self.max_len - len(word_ids)
        if pad_len > 0:
            word_ids += [self.word2idx["<PAD>"]] * pad_len
            tag_ids += [-100] * pad_len
            mask += [0] * pad_len

        return {
            "input_ids": torch.tensor(word_ids, dtype=torch.long),
            "tag_ids": torch.tensor(tag_ids, dtype=torch.long),
            "mask": torch.tensor(mask, dtype=torch.uint8)
        }

def get_dataloaders(word2idx, batch_size=32, max_len=50):
    from datasets import load_dataset
    print("Loading dataset for DataLoaders...")
    dataset = load_dataset("lhoestq/conll2003")
    
    train_dataset = NERDataset(dataset['train'], word2idx, max_len=max_len)
    val_dataset = NERDataset(dataset['validation'], word2idx, max_len=max_len)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader

# Only ONE test block at the bottom!
if __name__ == "__main__":
    with open("data/word2idx.json", "r") as f:
        word2idx = json.load(f)
        
    train_loader, val_loader = get_dataloaders(word2idx, batch_size=32, max_len=15)
    
    first_batch = next(iter(train_loader))
    
    print("\n=== First Batch (Tray) Received! ===")
    print(f"Batch Input Shape : {first_batch['input_ids'].shape}")
    print(f"Batch Tag Shape   : {first_batch['tag_ids'].shape}")
    print(f"Batch Mask Shape  : {first_batch['mask'].shape}")