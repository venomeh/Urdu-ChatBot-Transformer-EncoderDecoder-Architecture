import streamlit as st
import torch
import torch.nn as nn
import math
import re
import sys

# Set page config
st.set_page_config(
    page_title="Urdu Chatbot",
    page_icon="💬",
    layout="centered"
)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ==================== COPY MODEL CLASSES ====================

class Vocabulary:
    """Vocabulary class to manage word-to-index mappings"""
    
    def __init__(self):
        self.word2idx = {'<PAD>': 0, '<SOS>': 1, '<EOS>': 2, '<UNK>': 3}
        self.idx2word = {0: '<PAD>', 1: '<SOS>', 2: '<EOS>', 3: '<UNK>'}
        self.n_words = 4
    
    def add_word(self, word):
        """Add a word to vocabulary"""
        if word not in self.word2idx:
            self.word2idx[word] = self.n_words
            self.idx2word[self.n_words] = word
            self.n_words += 1
    
    def add_sentence(self, sentence):
        """Add all words in a sentence to vocabulary"""
        for word in sentence.split():
            self.add_word(word)
    
    def sentence_to_indices(self, sentence):
        """Convert sentence to list of indices"""
        return [self.word2idx.get(word, self.word2idx['<UNK>']) 
                for word in sentence.split()]
    
    def indices_to_sentence(self, indices):
        """Convert list of indices to sentence"""
        words = []
        for idx in indices:
            if idx in [self.word2idx['<EOS>'], self.word2idx['<PAD>']]:
                break
            if idx != self.word2idx['<SOS>']:
                words.append(self.idx2word.get(idx, '<UNK>'))
        return ' '.join(words)


class UrduTextNormalizer:
    """Normalizes and cleans Urdu text"""
    
    def __init__(self):
        self.urdu_pattern = re.compile(r'[\u0600-\u06FF\s]+')
        
    def normalize(self, text):
        """Normalize Urdu text"""
        text = text.strip()
        text = re.sub(r'\s+', ' ', text)
        
        # Normalize Arabic/Urdu characters
        text = text.replace('ك', 'ک')
        text = text.replace('ي', 'ی')
        text = text.replace('ى', 'ی')
        
        # Remove diacritics
        diacritics = ''.join([chr(i) for i in range(0x064B, 0x0653)])
        text = text.translate(str.maketrans('', '', diacritics))
        
        return text.strip()
    
    def clean(self, text):
        """Clean text by removing unwanted characters"""
        text = self.normalize(text)
        text = re.sub(r'[^\u0600-\u06FF\s\?\!\.\،\؛]', '', text)
        return text.strip()


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer"""
    
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        self.d_model = d_model
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                             (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        pe_slice = self.get_buffer('pe')[:, :x.size(1), :]
        return x + pe_slice


class MultiHeadAttention(nn.Module):
    """Multi-head attention mechanism"""
    
    def __init__(self, d_model, n_heads):
        super().__init__()
        assert d_model % n_heads == 0
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
    
    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attention = torch.softmax(scores, dim=-1)
        output = torch.matmul(attention, V)
        
        return output, attention
    
    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        
        Q = self.W_q(query)
        K = self.W_k(key)
        V = self.W_v(value)
        
        Q = Q.view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        
        x, attention = self.scaled_dot_product_attention(Q, K, V, mask)
        
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        output = self.W_o(x)
        
        return output, attention


class FeedForward(nn.Module):
    """Position-wise feed-forward network"""
    
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        return self.linear2(self.dropout(torch.relu(self.linear1(x))))


class EncoderLayer(nn.Module):
    """Single Transformer Encoder Layer"""
    
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        # Self-attention
        attn_output, _ = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # Feed-forward
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        
        return x


class DecoderLayer(nn.Module):
    """Single Transformer Decoder Layer"""
    
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads)
        self.cross_attn = MultiHeadAttention(d_model, n_heads)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, encoder_output, src_mask=None, tgt_mask=None):
        # Self-attention (with causal mask)
        attn_output, _ = self.self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # Cross-attention (decoder attends to encoder)
        attn_output, _ = self.cross_attn(x, encoder_output, encoder_output, src_mask)
        x = self.norm2(x + self.dropout(attn_output))
        
        # Feed-forward
        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ff_output))
        
        return x


class EncoderDecoderTransformer(nn.Module):
    """Encoder-Decoder Transformer for conversational understanding"""
    
    def __init__(self, vocab_size, d_model=256, n_heads=8, n_encoder_layers=4,
                 n_decoder_layers=4, d_ff=1024, dropout=0.1, max_len=5000):
        super().__init__()
        
        self.d_model = d_model
        
        # Embeddings
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len)
        
        # Encoder
        self.encoder_layers = nn.ModuleList([
            EncoderLayer(d_model, n_heads, d_ff, dropout) 
            for _ in range(n_encoder_layers)
        ])
        
        # Decoder
        self.decoder_layers = nn.ModuleList([
            DecoderLayer(d_model, n_heads, d_ff, dropout) 
            for _ in range(n_decoder_layers)
        ])
        
        # Output projection
        self.fc_out = nn.Linear(d_model, vocab_size)
        self.dropout = nn.Dropout(dropout)
    
    def generate_causal_mask(self, seq_len, device):
        """Generate causal mask for decoder (prevents looking at future tokens)"""
        mask = torch.tril(torch.ones(seq_len, seq_len)).bool()
        return mask.unsqueeze(0).unsqueeze(0).to(device)
    
    def generate_padding_mask(self, x):
        """Generate padding mask"""
        return (x != 0).unsqueeze(1).unsqueeze(2)
    
    def encode(self, src, src_mask=None):
        """Encode source sequence"""
        # Apply embeddings
        x = self.embedding(src) * math.sqrt(self.d_model)
        x = self.pos_encoding(x)
        x = self.dropout(x)
        
        # Pass through encoder layers
        for layer in self.encoder_layers:
            x = layer(x, src_mask)
        
        return x
    
    def decode(self, tgt, encoder_output, src_mask=None, tgt_mask=None):
        """Decode target sequence"""
        # Apply embeddings
        x = self.embedding(tgt) * math.sqrt(self.d_model)
        x = self.pos_encoding(x)
        x = self.dropout(x)
        
        # Pass through decoder layers
        for layer in self.decoder_layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)
        
        return x
    
    def forward(self, src, tgt):
        """Forward pass through encoder-decoder"""
        # Generate masks
        src_mask = self.generate_padding_mask(src)
        tgt_mask = self.generate_padding_mask(tgt)
        
        # Generate causal mask for decoder
        seq_len = tgt.size(1)
        causal_mask = self.generate_causal_mask(seq_len, tgt.device)
        tgt_mask = tgt_mask & causal_mask
        
        # Encode
        encoder_output = self.encode(src, src_mask)
        
        # Decode
        decoder_output = self.decode(tgt, encoder_output, src_mask, tgt_mask)
        
        # Output projection
        output = self.fc_out(decoder_output)
        
        return output


# ==================== GENERATION FUNCTION ====================

def generate_response(model, input_text, vocab, normalizer, device, 
                     max_len=50, temperature=0.5, top_k=50, top_p=0.9):
    """Generate response using the trained encoder-decoder model"""
    model.eval()
    
    # Normalize and convert input
    input_text = normalizer.clean(input_text)
    encoder_indices = [vocab.word2idx['<SOS>']] + \
                      vocab.sentence_to_indices(input_text) + \
                      [vocab.word2idx['<EOS>']]
    
    encoder_input = torch.tensor([encoder_indices]).to(device)
    
    # Encode the input
    with torch.no_grad():
        src_mask = model.generate_padding_mask(encoder_input)
        encoder_output = model.encode(encoder_input, src_mask)
    
    # Start decoding with SOS token
    decoder_input = [vocab.word2idx['<SOS>']]
    
    for _ in range(max_len):
        decoder_tensor = torch.tensor([decoder_input]).to(device)
        
        with torch.no_grad():
            # Decode
            tgt_mask = model.generate_padding_mask(decoder_tensor)
            seq_len = decoder_tensor.size(1)
            causal_mask = model.generate_causal_mask(seq_len, decoder_tensor.device)
            tgt_mask = tgt_mask & causal_mask
            
            decoder_output = model.decode(decoder_tensor, encoder_output, src_mask, tgt_mask)
            output = model.fc_out(decoder_output)
        
        # Get logits for last position
        logits = output[0, -1] / temperature
        
        # Mask out special tokens from being generated
        logits[vocab.word2idx['<PAD>']] = -float('Inf')
        logits[vocab.word2idx['<SOS>']] = -float('Inf')
        logits[vocab.word2idx['<UNK>']] = -float('Inf')
        
        # Apply top-k filtering
        if top_k > 0:
            indices_to_remove = logits < torch.topk(logits, min(top_k, logits.size(-1)))[0][..., -1, None]
            logits[indices_to_remove] = -float('Inf')
        
        # Apply top-p (nucleus) filtering
        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
            
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            
            indices_to_remove = sorted_indices[sorted_indices_to_remove]
            logits[indices_to_remove] = -float('Inf')
        
        # Sample from distribution
        probs = torch.softmax(logits, dim=-1)
        
        if probs.sum() == 0:
            break
            
        next_token = torch.multinomial(probs, num_samples=1).item()
        
        decoder_input.append(next_token)
        
        if next_token == vocab.word2idx['<EOS>']:
            break
    
    # Convert to sentence (skip SOS)
    response_indices = decoder_input[1:]
    response = vocab.indices_to_sentence(response_indices)
    
    return response if response else "معذرت، میں سمجھ نہیں سکا"


# ==================== LOAD MODEL ====================

@st.cache_resource
def load_model():
    """Load the trained model"""
    try:
        # Add classes to __main__ module so pickle can find them
        import __main__
        __main__.Vocabulary = Vocabulary
        __main__.UrduTextNormalizer = UrduTextNormalizer
        
        checkpoint = torch.load('urdu_encoder_decoder.pth', map_location=device, weights_only=False)
        
        vocab = checkpoint['vocab']
        normalizer = checkpoint['normalizer']
        
        model = EncoderDecoderTransformer(
            vocab_size=vocab.n_words,
            d_model=256,
            n_heads=16,
            n_encoder_layers=8,
            n_decoder_layers=8,
            d_ff=1024,
            dropout=0.25
        ).to(device)
        
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        return model, vocab, normalizer
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None, None


# ==================== STREAMLIT APP ====================

def main():
    # Custom CSS for better UI
    st.markdown("""
        <style>
        .main {
            background-color: #f5f5f5;
        }
        .stTextInput > div > div > input {
            font-size: 18px;
            direction: rtl;
            text-align: right;
        }
        .response-box {
            background-color: #e3f2fd;
            padding: 20px;
            border-radius: 10px;
            margin: 10px 0;
            direction: rtl;
            text-align: right;
            font-size: 20px;
            border-left: 5px solid #2196f3;
        }
        .user-input-box {
            background-color: #fff3e0;
            padding: 20px;
            border-radius: 10px;
            margin: 10px 0;
            direction: rtl;
            text-align: right;
            font-size: 20px;
            border-left: 5px solid #ff9800;
        }
        h1 {
            text-align: center;
            color: #1976d2;
        }
        .stButton > button {
            width: 100%;
            background-color: #1976d2;
            color: white;
            font-size: 18px;
            padding: 10px;
            border-radius: 5px;
        }
        .stButton > button:hover {
            background-color: #1565c0;
        }
        </style>
    """, unsafe_allow_html=True)
    
    # Title
    st.title("💬 اردو چیٹ بوٹ")
    st.markdown("### Urdu Conversational AI")
    
    # Load model
    with st.spinner('Loading model...'):
        model, vocab, normalizer = load_model()
    
    if model is None:
        st.error("Failed to load model. Please ensure 'urdu_encoder_decoder.pth' exists in the same directory.")
        return
    
    st.success("✅ Model loaded successfully!")
    
    # Initialize chat history
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    # Sidebar settings
    st.sidebar.title("⚙️ Settings")
    temperature = st.sidebar.slider("Temperature", 0.1, 1.5, 0.7, 0.1)
    top_k = st.sidebar.slider("Top-K", 10, 100, 50, 10)
    top_p = st.sidebar.slider("Top-P", 0.5, 1.0, 0.9, 0.05)
    max_len = st.sidebar.slider("Max Length", 20, 100, 50, 10)
    
    if st.sidebar.button("🗑️ Clear Chat History"):
        st.session_state.chat_history = []
        st.rerun()
    
    # Main input area
    st.markdown("---")
    
    user_input = st.text_input(
        "آپ کا سوال / Your Message:",
        placeholder="یہاں اردو میں لکھیں... Type your message in Urdu...",
        key="user_input"
    )
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        send_button = st.button("📤 Send / بھیجیں", use_container_width=True)
    
    # Process input
    if send_button and user_input:
        with st.spinner('Generating response...'):
            response = generate_response(
                model, 
                user_input, 
                vocab, 
                normalizer, 
                device,
                max_len=max_len,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p
            )
        
        # Add to chat history
        st.session_state.chat_history.append({
            'user': user_input,
            'bot': response
        })
    
    # Display chat history
    if st.session_state.chat_history:
        st.markdown("---")
        st.markdown("### 💭 Chat History")
        
        for i, chat in enumerate(reversed(st.session_state.chat_history)):
            st.markdown(f"""
                <div class="user-input-box">
                    <strong>👤 آپ:</strong> {chat['user']}
                </div>
            """, unsafe_allow_html=True)
            
            st.markdown(f"""
                <div class="response-box">
                    <strong>🤖 بوٹ:</strong> {chat['bot']}
                </div>
            """, unsafe_allow_html=True)
            
            if i < len(st.session_state.chat_history) - 1:
                st.markdown("<hr style='margin: 5px 0;'>", unsafe_allow_html=True)
    
    # Footer
    st.markdown("---")
    st.markdown(
        "<p style='text-align: center; color: gray;'>Powered by Transformer Model | "
        f"Device: {device}</p>",
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()
