# Custom Named Entity Recognition (NER) Engine

A Deep Learning sequence-labeling engine built entirely from scratch to extract named entities (People, Locations, Organizations, and Misc) from unstructured text. 

Instead of relying on pre-trained mega-models, this project implements a custom **Bidirectional Long Short-Term Memory (BiLSTM)** neural network using **PyTorch**. The pipeline handles raw data processing, model training, evaluation using entity-level F1-scores, and deployment via a live **FastAPI** web server.

## 🏗️ System Architecture
1. **Client Request:** The user sends a raw text string via an HTTP POST request to the FastAPI `/predict` endpoint.
2. **Text Preprocessing:** The backend tokenizes the text, converts words to integer IDs using the custom `word2idx.json` vocabulary, and maps unknown words to `<UNK>`.
3. **Tensor Conversion:** The integer IDs are converted into a PyTorch Tensor and fed into the model.
4. **BiLSTM Forward Pass:**
   * **Embedding Layer:** Converts word IDs into 100-dimensional dense vectors.
   * **BiLSTM Layer:** Processes the sequence in both directions to capture forward and backward context (128 hidden dimensions).
   * **Linear Classifier:** Maps the LSTM outputs to the 9 possible NER tag probabilities.
5. **Post-processing:** The `argmax` of the output is taken to find the highest probability tag, which is mapped back to human-readable labels (e.g., `B-PER`, `B-LOC`) using `IDX2TAG`.
6. **JSON Response:** The structured entity list is returned to the client.

## 🛠️ Tech Stack
*   **Machine Learning:** PyTorch, Torch NN
*   **Data Processing:** HuggingFace Datasets (CoNLL-2003), seqeval
*   **Backend / API:** FastAPI, Uvicorn, Pydantic
*   **Language:** Python 3

## 📂 Project Structure
```text
ner-bilstm/
├── app.py                           # Main FastAPI server and inference endpoint
├── data/
│   └── word2idx.json                # Custom vocabulary mapping
├── saved_models/
│   └── ner_bilstm.pt                # Trained PyTorch model weights (brain)
├── src/
│   ├── preprocessing/
│   │   └── dataset.py               # Data loading, tokenization, and padding
│   ├── models/
│   │   ├── bilstm.py                # BiLSTM neural network architecture
│   │   └── predict.py               # Local terminal inference script
│   ├── training/
│   │   └── train.py                 # Training loop and backpropagation
│   └── evaluation/
│       └── evaluate.py              # Validation testing using seqeval F1-scores
├── requirements.txt                 # Project dependencies
└── README.md


## ⚙️ Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/yourusername/ner-bilstm.git](https://github.com/yourusername/ner-bilstm.git)
   cd ner-bilstm

2. Create and activate a virtual environment:
python -m venv venv
source venv/bin/activate  # macOS/Linux

3. Install the dependencies:
pip install -r requirements.txt

🧠 Training & Evaluation
If you want to train the model yourself from scratch:

1. Run the Training Loop:
This will process the CoNLL-2003 dataset, train the BiLSTM for 3 epochs, and save the weights to saved_models/ner_bilstm.pt.
python -m src.training.train

2. Evaluate the Model:
Grades the model on unseen validation data to output the Precision, Recall, and F1-scores for each entity type.
python -m src.evaluation.evaluate

🌐 Running the Application & API Documentation
To launch the live inference server:
python -m uvicorn app:app --reload

Once the server says Application startup complete, navigate to the automatic Swagger UI in your browser for full API documentation:
http://127.0.0.1:8000/docs

API Usage Example
Send a POST request to the /predict endpoint with a JSON body:
{
  "text": "Tim Cook is the CEO of Apple in California"
}

Expected Response:
{
  "original_text": "Tim Cook is the CEO of Apple in California",
  "entities": [
    {"word": "Tim", "entity": "B-PER"},
    {"word": "Cook", "entity": "I-PER"},
    {"word": "is", "entity": "O"},
    {"word": "the", "entity": "O"},
    {"word": "CEO", "entity": "O"},
    {"word": "of", "entity": "O"},
    {"word": "Apple", "entity": "O"},
    {"word": "in", "entity": "O"},
    {"word": "California", "entity": "B-LOC"}
  ]
}

## 💡 Real-World Use Cases
This model serves as a foundational data-extraction tool that can be integrated into larger software pipelines:
*   **Enhancing RAG Systems:** Pre-processing unstructured text to extract critical entities (like specific names from policies or locations from reviews) before feeding them into a vector database to improve AI search and retrieval accuracy.
*   **Automated Data Redaction (Security):** Identifying and masking Personally Identifiable Information (PII) such as names (PER) and locations (LOC) in legal, medical, or financial documents before they are shared publicly.
*   **Sports Analytics & Media:** Automatically extracting player names (PER), team organizations (ORG), and match venues (LOC) from live text commentary or sports news articles to build structured databases.
*   **Resume Parsing (HR Tech):** Scanning thousands of unstructured resumes to instantly pull out candidate names, previous employers, and university locations to speed up recruitment pipelines.
*   **Customer Support Routing:** Tagging incoming support tickets or emails based on the organizations or people mentioned to automatically route them to the correct internal department.
*   **Financial Information Extraction:** Automatically pulling out key stakeholders, companies, and geographic locations from massive batches of news articles or quarterly earnings reports.


