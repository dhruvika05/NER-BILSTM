from fastapi import FastAPI
from pydantic import BaseModel
import torch
import json

from src.models.bilstm import BiLSTM_NER
from src.preprocessing.dataset import IDX2TAG

# 1. Initialize the FastAPI app
app = FastAPI(title="NER API", description="Custom BiLSTM Named Entity Recognizer")

# 2. Global variables
model = None
word2idx = None

class NERRequest(BaseModel):
    text: str

# 3. Load the brain when server starts
@app.on_event("startup")
def load_model():
    global model, word2idx
    print("Starting up: Loading model and vocabulary...")
    
    with open("data/word2idx.json", "r") as f:
        word2idx = json.load(f)
        
    model = BiLSTM_NER(
        vocab_size=len(word2idx),
        embedding_dim=100,
        hidden_dim=128,
        num_tags=9,
        padding_idx=word2idx["<PAD>"]
    )
    model.load_state_dict(torch.load("saved_models/ner_bilstm.pt"))
    model.eval()
    print("Ready!")

# 4. Create the prediction endpoint
@app.post("/predict")
def predict_entities(request: NERRequest):
    words = request.text.split()
    
    word_ids = []
    for w in words:
        word_ids.append(word2idx.get(w.lower(), word2idx["<UNK>"]))
        
    input_tensor = torch.tensor([word_ids], dtype=torch.long)
    
    with torch.no_grad():
        logits = model(input_tensor)
        predictions = torch.argmax(logits, dim=-1)
        
    pred_tags = [IDX2TAG[tag_id.item()] for tag_id in predictions[0]]
    
    results = [{"word": w, "entity": tag} for w, tag in zip(words, pred_tags)]
    return {"original_text": request.text, "entities": results}