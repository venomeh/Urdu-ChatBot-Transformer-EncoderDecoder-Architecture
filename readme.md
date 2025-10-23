# Urdu ChatBot - Transformer Encoder-Decoder Architecture

An Urdu conversational AI chatbot built from scratch using a Transformer Encoder-Decoder architecture. This project demonstrates a deep dive into the core mechanics of transformer models, trained entirely on custom Urdu conversational data.

## 🌟 Overview

This chatbot uses a **Transformer Encoder-Decoder architecture** built and trained from scratch on Urdu conversational data provided in `corpus.txt`. The model learns to understand and generate contextually appropriate responses in Urdu, showcasing the power of attention mechanisms and sequential modeling.

## 🏗️ Architecture

- **Model**: Encoder-Decoder Transformer
- **Built from scratch**: All components (Multi-Head Attention, Positional Encoding, Feed-Forward Networks) implemented manually
- **Framework**: PyTorch
- **Features**:
  - 8 Encoder Layers
  - 8 Decoder Layers
  - 16 Attention Heads
  - 256 Dimensional Embeddings
  - Trained on custom Urdu conversational corpus

## 🚀 Getting Started

### Prerequisites

- Python 3.11+ (recommended)
- PyTorch
- Streamlit

### Installation & Running Locally

Follow these steps to run the chatbot on your local machine:

#### 1. Download Code and Model

```bash
git clone https://github.com/venomeh/Urdu-ChatBot-Transformer-EncoderDecoder-Architecture.git
cd Urdu-ChatBot-Transformer-EncoderDecoder-Architecture
```

#### 2. Install Requirements

```bash
pip install -r requirements.txt
```

#### 3. Run the Application

```bash
streamlit run app.py
```

The app will open in your default browser at `http://localhost:8501`

## 📁 Project Structure

```
├── model.py                          # Core Model
├── app.py                          # Streamlit web application
├── corpus.txt                      # Training data (Urdu conversation)
├── urdu_encoder_decoder.pth       # Trained model weights
├── requirements.txt               # Python dependencies
└── README.md                      # Project documentation
```

## 🎯 Features

- **RTL (Right-to-Left) Support**: Proper display of Urdu text
- **Interactive Web UI**: Clean, user-friendly Streamlit interface
- **Adjustable Parameters**:
  - Temperature (controls randomness)
  - Top-K sampling
  - Top-P (nucleus) sampling
  - Response length control
- **Chat History**: Maintains conversation context
- **Real-time Response Generation**: Powered by the trained transformer model

## 🧠 Model Training

The model was trained using:
- **Data**: Custom Urdu conversational pairs from `corpus.txt`
- **Optimizer**: Adam
- **Loss Function**: Cross-Entropy Loss
- **Text Normalization**: Custom Urdu text normalizer
- **Vocabulary**: Built from training corpus with special tokens (<PAD>, <SOS>, <EOS>, <UNK>, <MASK>)

The training achieved:
- **Perplexity Score**: ~33
- **RoughL**: 0.30
- **chrF**: 0.42
- **BLEU**: 0.28

  
## 💡 Technical Highlights

This project was an **interesting dive into the core of things**:
- Manual implementation of Multi-Head Attention mechanism
- Custom Positional Encoding
- Encoder-Decoder architecture with causal masking
- Advanced sampling strategies (Top-K, Top-P)
- Urdu text processing and normalization

## 📊 Model Parameters

| Parameter | Value |
|-----------|-------|
| Embedding Dimension | 256 |
| Number of Attention Heads | 16 |
| Encoder Layers | 8 |
| Decoder Layers | 8 |
| Feed-Forward Dimension | 1024 |
| Dropout | 0.25 |


## 📝 Usage

1. Launch the application using `streamlit run app.py`
2. Type your message in Urdu in the input box
3. Click "Send / بھیجیں" to get a response
4. Adjust settings in the sidebar to control response generation
5. Clear chat history using the sidebar button

## 🎓 Learning Outcomes

This project provided hands-on experience with:
- Building transformer models from scratch
- Understanding attention mechanisms
- Working with sequential data
- Urdu NLP processing
- Model deployment with Streamlit



## 🙏 Acknowledgments

Built as an exploration into transformer architectures and their application to low-resource languages like Urdu.

---

**Note**: This is an educational project demonstrating transformer architecture implementation from scratch. The model's performance depends on the training data provided in `corpus.txt`.

## Demo link : https://urdu-chatbot-transformer-encoderdecoder-architecture-nasxnlahg.streamlit.app/
