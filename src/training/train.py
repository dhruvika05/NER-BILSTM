import torch
import torch.nn as nn
import torch.optim as optim
import json
import os

from src.preprocessing.dataset import get_dataloaders
from src.models.bilstm import BiLSTM_NER

def train_model():
    # 1. Load our saved Vocabulary
    print("Loading Vocabulary...")
    with open("data/word2idx.json", "r") as f:
        word2idx = json.load(f)
        
    vocab_size = len(word2idx)
    
    # 2. Get DataLoaders (The Waiters)
    train_loader, val_loader = get_dataloaders(word2idx, batch_size=32, max_len=30)
    
    # 3. Initialize the Newborn Model
    print("Initializing Model...")
    model = BiLSTM_NER(
        vocab_size=vocab_size,
        embedding_dim=100,
        hidden_dim=128,
        num_tags=9,
        padding_idx=word2idx["<PAD>"]
    )
    
    # 4. The Loss Function and Optimizer
    # ignore_index=0 tells PyTorch: "Do not calculate loss for <PAD> tokens!"
    criterion = nn.CrossEntropyLoss(ignore_index=-100)
    
    # Adam is the most popular optimizer. lr=0.001 is the "Learning Rate" 
    # (how drastically it turns the dials after each mistake)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # 5. The Training Loop!
    epochs = 3  # We will run through the entire dataset 3 times
    print("\n=== Starting Training ===")
    
    for epoch in range(epochs):
        model.train() # Put model in training mode
        total_loss = 0
        
        # Loop through every batch in the training set
        for batch in train_loader:
            input_ids = batch['input_ids']
            tag_ids = batch['tag_ids']
            
            # Step A: Clear old math gradients from the last batch
            optimizer.zero_grad()
            
            # Step B: Forward Pass (Predict)
            # Shape: [Batch, Seq_Len, Num_Tags] -> [32, 30, 9]
            logits = model(input_ids)
            
            # Step C: Calculate Loss
            # PyTorch's Loss function expects a 2D grid, not a 3D cube.
            # .view(-1) mathematically smashes the Batch and Seq_Len dimensions together.
            logits_flat = logits.view(-1, 9)
            tag_ids_flat = tag_ids.view(-1)
            
            loss = criterion(logits_flat, tag_ids_flat)
            
            # Step D: Backpropagation (Calculate how wrong we were)
            loss.backward()
            
            # Step E: Update the dials!
            optimizer.step()
            
            total_loss += loss.item()
            
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{epochs} | Average Training Loss: {avg_loss:.4f}")

    # 6. Save the trained brain!
    os.makedirs("saved_models", exist_ok=True)
    torch.save(model.state_dict(), "saved_models/ner_bilstm.pt")
    print("\n=== Training Complete! ===")
    print("Model weights saved to saved_models/ner_bilstm.pt")

if __name__ == "__main__":
    train_model()