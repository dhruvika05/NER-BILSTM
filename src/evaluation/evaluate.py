import torch
import json
from seqeval.metrics import classification_report

from src.preprocessing.dataset import get_dataloaders, IDX2TAG
from src.models.bilstm import BiLSTM_NER

def evaluate_model():
    print("Loading Vocabulary...")
    with open("data/word2idx.json", "r") as f:
        word2idx = json.load(f)
        
    vocab_size = len(word2idx)
    pad_idx = word2idx["<PAD>"]
    
    # 1. Get the Waiter (We only care about val_loader this time)
    _, val_loader = get_dataloaders(word2idx, batch_size=32, max_len=30)
    
    # 2. Rebuild the empty brain...
    print("Loading Trained Model...")
    model = BiLSTM_NER(
        vocab_size=vocab_size,
        embedding_dim=100,
        hidden_dim=128,
        num_tags=9,
        padding_idx=pad_idx
    )
    
    # ...and fill it with our trained weights!
    model.load_state_dict(torch.load("saved_models/ner_bilstm.pt"))
    
    # LOCK the model for testing
    model.eval()
    
    all_true_tags = []
    all_pred_tags = []
    
    print("Taking the Final Exam (Evaluating on Validation Set)...")
    
    # Turn off the calculus engine to save memory
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch['input_ids']
            tag_ids = batch['tag_ids']
            mask = batch['mask']
            
            # Predict! Shape: [32, 30, 9]
            logits = model(input_ids)
            
            # Find the tag ID with the highest probability
            # Shape becomes: [32, 30]
            predictions = torch.argmax(logits, dim=-1)
            
            # Convert IDs back to human words (B-PER, O, etc.)
            for i in range(input_ids.shape[0]):  # Loop over sentences in batch
                true_seq = []
                pred_seq = []
                for j in range(input_ids.shape[1]):  # Loop over words in sentence
                    # ONLY evaluate real words. Ignore the <PAD> zeros!
                    if mask[i, j] == 1:
                        true_tag_id = tag_ids[i, j].item()
                        pred_tag_id = predictions[i, j].item()
                        
                        true_seq.append(IDX2TAG[true_tag_id])
                        pred_seq.append(IDX2TAG[pred_tag_id])
                        
                all_true_tags.append(true_seq)
                all_pred_tags.append(pred_seq)
                
    # 3. Print the strict grading report
    print("\n=== Final Exam Results (Entity-Level) ===")
    print(classification_report(all_true_tags, all_pred_tags))

if __name__ == "__main__":
    evaluate_model()