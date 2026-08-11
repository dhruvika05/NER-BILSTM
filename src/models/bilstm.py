import torch
import torch.nn as nn

class BiLSTM_NER(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_tags, padding_idx=0):
        """
        Args:
            vocab_size: Total number of unique words (e.g., ~11,000)
            embedding_dim: Size of the word vectors (e.g., 100)
            hidden_dim: Internal memory size of the LSTM (e.g., 128)
            num_tags: Total number of BIO tags to predict (e.g., 9)
            padding_idx: The ID of our <PAD> token (0), so the Embedding layer ignores it.
        """
        # Always call the parent class initialization
        super(BiLSTM_NER, self).__init__()
        
        # 1. The Embedding Layer: ID -> Vector Cube
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size, 
            embedding_dim=embedding_dim, 
            padding_idx=padding_idx
        )
        
        # 2. The BiLSTM Layer
        # batch_first=True tells PyTorch our data is shaped [Batch, Seq_Len, Features]
        # bidirectional=True doubles the output size automatically!
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=1,             # We use 1 layer for speed, can increase later
            batch_first=True,
            bidirectional=True
        )
        
        # 3. The Linear Output Layer (The Funnel)
        # We multiply hidden_dim * 2 because the LSTM is bidirectional!
        self.fc = nn.Linear(hidden_dim * 2, num_tags)

    def forward(self, x):
        """
        This function defines how data flows through the layers.
        x shape: [Batch_Size, Seq_Len] (e.g., [32, 15])
        """
        # Step 1: Embeddings
        # Shape becomes: [Batch, Seq_Len, Embedding_Dim] -> [32, 15, 100]
        embedded = self.embedding(x)
        
        # Step 2: BiLSTM
        # The LSTM returns the output, plus hidden states we don't need right now (_)
        # Shape becomes: [Batch, Seq_Len, Hidden_Dim * 2] -> [32, 15, 256]
        lstm_out, _ = self.lstm(embedded)
        
        # Step 3: Linear Classifier (Logits)
        # Shape becomes: [Batch, Seq_Len, Num_Tags] -> [32, 15, 9]
        logits = self.fc(lstm_out)
        
        return logits


# Let's test the math and shapes!
if __name__ == "__main__":
    # Create fake dimensions
    BATCH_SIZE = 32
    SEQ_LEN = 15
    VOCAB_SIZE = 5000
    EMBEDDING_DIM = 100
    HIDDEN_DIM = 128
    NUM_TAGS = 9
    
    # 1. Create the model
    model = BiLSTM_NER(VOCAB_SIZE, EMBEDDING_DIM, HIDDEN_DIM, NUM_TAGS)
    print("=== Model Architecture ===")
    print(model)
    
    # 2. Create a fake batch of word IDs (random integers)
    # Shape: [32, 15]
    fake_input = torch.randint(low=1, high=VOCAB_SIZE, size=(BATCH_SIZE, SEQ_LEN))
    
    print(f"\nFake Input Shape: {fake_input.shape} -> [Batch, Seq_Len]")
    
    # 3. Pass the fake input through the model!
    predictions = model(fake_input)
    
    print(f"Predictions Shape : {predictions.shape} -> [Batch, Seq_Len, Num_Tags]")