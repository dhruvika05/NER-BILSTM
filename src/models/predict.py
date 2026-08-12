import torch
import json
from src.models.bilstm import BiLSTM_NER
from src.preprocessing.dataset import IDX2TAG

def predict_sentence(sentence, model, word2idx):
    # Lock the model for testing
    model.eval()
    
    # Simple tokenization (split by spaces)
    words = sentence.split()
    
    # Convert words to IDs
    word_ids = []
    for w in words:
        word_ids.append(word2idx.get(w.lower(), word2idx["<UNK>"]))
        
    # Convert to PyTorch Tensor and add a fake Batch dimension [1, Seq_Len]
    input_tensor = torch.tensor([word_ids], dtype=torch.long)
    
    # Predict!
    with torch.no_grad():
        logits = model(input_tensor)
        predictions = torch.argmax(logits, dim=-1)
        
    # Convert predicted IDs back to human tags (B-PER, O, etc.)
    pred_tags = [IDX2TAG[tag_id.item()] for tag_id in predictions[0]]
    
    print("\n=== Model Output ===")
    for word, tag in zip(words, pred_tags):
        print(f"{word:>15} : {tag}")

if __name__ == "__main__":
    # 1. Load the Vocabulary
    with open("data/word2idx.json", "r") as f:
        word2idx = json.load(f)
        
    # 2. Rebuild the empty brain
    model = BiLSTM_NER(
        vocab_size=len(word2idx),
        embedding_dim=100,
        hidden_dim=128,
        num_tags=9,
        padding_idx=word2idx["<PAD>"]
    )
    
    # 3. Load our smart weights!
    model.load_state_dict(torch.load("saved_models/ner_bilstm.pt"))
    
    # 4. Test it!
    # Try changing this sentence to whatever you want!
    test_sentence = "Elon Musk flew to London to visit Google headquarters ."
    
    print(f"Input Sentence: {test_sentence}")
    predict_sentence(test_sentence, model, word2idx)