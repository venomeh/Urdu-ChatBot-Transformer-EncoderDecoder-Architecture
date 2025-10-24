import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import math
import re
import random
import pickle
import os
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# ==================== TEXT PREPROCESSING ====================

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
    
    def split_into_utterances(self, text):
        """Split text into utterances"""
        # Split by common sentence endings
        utterances = re.split(r'[\.!\?؟۔]+', text)
        return [u.strip() for u in utterances if u.strip()]


# ==================== VOCABULARY ====================

class Vocabulary:
    """Build vocabulary from text"""
    
    def __init__(self):
        self.word2idx = {
            '<PAD>': 0, 
            '<SOS>': 1, 
            '<EOS>': 2, 
            '<UNK>': 3
        }
        self.idx2word = {
            0: '<PAD>', 
            1: '<SOS>', 
            2: '<EOS>', 
            3: '<UNK>'
        }
        self.word_count = Counter()
        self.n_words = 4
        
    def add_sentence(self, sentence):
        """Add words from sentence to vocabulary"""
        for word in sentence.split():
            self.add_word(word)
    
    def add_word(self, word):
        """Add a word to vocabulary"""
        if word not in self.word2idx:
            self.word2idx[word] = self.n_words
            self.idx2word[self.n_words] = word
            self.n_words += 1
        self.word_count[word] += 1
    
    def sentence_to_indices(self, sentence, max_len=None):
        """Convert sentence to indices"""
        indices = [self.word2idx.get(word, self.word2idx['<UNK>']) 
                   for word in sentence.split()]
        if max_len:
            indices = indices[:max_len]
        return indices
    
    def indices_to_sentence(self, indices):
        """Convert indices to sentence"""
        words = []
        for idx in indices:
            if idx not in [0, 1, 2]:  # Skip PAD, SOS, EOS
                word = self.idx2word.get(idx, None)
                if word and word != '<UNK>':  # Only add valid, non-UNK words
                    words.append(word)
        return ' '.join(words)


# ==================== ENCODER-DECODER DATASET ====================

class EncoderDecoderDataset(Dataset):
    """Dataset for encoder-decoder training with context-response pairs"""
    
    def __init__(self, transcriptions, vocab, max_len=100, context_size=2):
        self.vocab = vocab
        self.max_len = max_len
        self.context_size = context_size
        self.pairs = []
        
        # Create context-response pairs
        for trans in transcriptions:
            utterances = trans.split('۔')  # Split by Urdu period
            utterances = [u.strip() for u in utterances if u.strip()]
            
            for i in range(1, len(utterances)):
                # Get context (previous utterances)
                start = max(0, i - context_size)
                context = ' '.join(utterances[start:i])
                response = utterances[i]
                
                if context and response:
                    self.pairs.append((context, response))
    
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        context, response = self.pairs[idx]
        
        # Encoder input (context)
        encoder_indices = [self.vocab.word2idx['<SOS>']] + \
                          self.vocab.sentence_to_indices(context, self.max_len-2) + \
                          [self.vocab.word2idx['<EOS>']]
        
        # Decoder input (response with SOS)
        decoder_input_indices = [self.vocab.word2idx['<SOS>']] + \
                                self.vocab.sentence_to_indices(response, self.max_len-2)
        
        # Decoder target (response with EOS)
        decoder_target_indices = self.vocab.sentence_to_indices(response, self.max_len-1) + \
                                 [self.vocab.word2idx['<EOS>']]
        
        return (torch.tensor(encoder_indices),
                torch.tensor(decoder_input_indices),
                torch.tensor(decoder_target_indices))


def collate_fn_encoder_decoder(batch):
    """Custom collate function for encoder-decoder training"""
    encoder_seqs, decoder_input_seqs, decoder_target_seqs = zip(*batch)
    
    # Find max lengths
    max_enc_len = max(len(x) for x in encoder_seqs)
    max_dec_len = max(len(x) for x in decoder_input_seqs)
    max_tgt_len = max(len(x) for x in decoder_target_seqs)
    
    # Pad sequences
    encoder_padded = torch.zeros(len(batch), max_enc_len, dtype=torch.long)
    decoder_input_padded = torch.zeros(len(batch), max_dec_len, dtype=torch.long)
    decoder_target_padded = torch.zeros(len(batch), max_tgt_len, dtype=torch.long)
    
    for i, (enc, dec_in, dec_tgt) in enumerate(zip(encoder_seqs, decoder_input_seqs, decoder_target_seqs)):
        encoder_padded[i, :len(enc)] = enc
        decoder_input_padded[i, :len(dec_in)] = dec_in
        decoder_target_padded[i, :len(dec_tgt)] = dec_tgt
    
    return encoder_padded, decoder_input_padded, decoder_target_padded


# ==================== TRANSFORMER COMPONENTS ====================

class PositionalEncoding(nn.Module):
    """Positional encoding for transformer"""
    
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        
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


# ==================== TRAINING ====================

def train_epoch(model, dataloader, optimizer, criterion, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    
    for batch_idx, (encoder_input, decoder_input, decoder_target) in enumerate(dataloader):
        encoder_input = encoder_input.to(device)
        decoder_input = decoder_input.to(device)
        decoder_target = decoder_target.to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        output = model(encoder_input, decoder_input)
        
        # Calculate loss
        loss = criterion(output.reshape(-1, output.size(-1)), 
                        decoder_target.reshape(-1))
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        total_loss += loss.item()
        
        if (batch_idx + 1) % 10 == 0:
            print(f"  Batch {batch_idx+1}/{len(dataloader)}, Loss: {loss.item():.4f}")
    
    return total_loss / len(dataloader)


def validate_epoch(model, dataloader, criterion, device):
    """Validate for one epoch"""
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for encoder_input, decoder_input, decoder_target in dataloader:
            encoder_input = encoder_input.to(device)
            decoder_input = decoder_input.to(device)
            decoder_target = decoder_target.to(device)
            
            # Forward pass
            output = model(encoder_input, decoder_input)
            
            # Calculate loss
            loss = criterion(output.reshape(-1, output.size(-1)), 
                            decoder_target.reshape(-1))
            
            total_loss += loss.item()
    
    return total_loss / len(dataloader)


# ==================== INFERENCE ====================

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
        logits[vocab.word2idx['<UNK>']] = -float('Inf')  # Prevent UNK generation
        
        # Apply top-k filtering
        if top_k > 0:
            indices_to_remove = logits < torch.topk(logits, min(top_k, logits.size(-1)))[0][..., -1, None]
            logits[indices_to_remove] = -float('Inf')
        
        # Apply top-p (nucleus) filtering
        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
            
            # Remove tokens with cumulative probability above the threshold
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            
            indices_to_remove = sorted_indices[sorted_indices_to_remove]
            logits[indices_to_remove] = -float('Inf')
        
        # Sample from distribution
        probs = torch.softmax(logits, dim=-1)
        
        # Check if all probabilities are zero (shouldn't happen, but safety check)
        if probs.sum() == 0:
            break
            
        next_token = torch.multinomial(probs, num_samples=1).item()
        
        decoder_input.append(next_token)
        
        # Stop if EOS token
        if next_token == vocab.word2idx['<EOS>']:
            break
    
    # Convert to sentence (skip SOS)
    response_indices = decoder_input[1:]
    response = vocab.indices_to_sentence(response_indices)
    
    return response if response else "معذرت، میں سمجھ نہیں سکا"  # Fallback response





def main():
    print("=== Urdu Encoder-Decoder Transformer Chatbot ===\n")
    
    transcription_data_usual = [
        "سلام کیسے ہیں آپ۔ میں بالکل ٹھیک ہوں شکریہ۔ آج موسم بہت اچھا ہے۔ جی ہاں بہت خوشگوار موسم ہے",
        "آپ کا نام کیا ہے۔ میرا نام احمد ہے۔ آپ کہاں سے ہیں۔ میں کراچی سے ہوں",
        "آج آپ کیا کر رہے ہیں۔ میں کام کر رہا ہوں۔ کیا آپ کو مدد چاہیے۔ جی نہیں شکریہ",
        "مجھے بھوک لگی ہے۔ تو کھانا کھا لیں۔ آپ کے ساتھ چلوں۔ جی ضرور چلیں",
        "یہ کتاب کیسی ہے۔ یہ بہت اچھی کتاب ہے۔ مجھے بھی پڑھنی ہے۔ آپ لے جا سکتے ہیں",
        "آج کیا دن ہے۔ آج جمعہ ہے۔ کل چھٹی ہے۔ جی ہاں کل سنیچر ہے",
        "میں تھک گیا ہوں۔ تو آرام کر لیں۔ شکریہ آپ کی مہربانی۔ کوئی بات نہیں",
        "یہ شہر کیسا ہے۔ یہ بہت خوبصورت شہر ہے۔ لوگ کیسے ہیں۔ لوگ بہت اچھے ہیں",
        "آپ کیا کھائیں گے۔ میں بریانی کھاؤں گا۔ میں بھی یہی لوں گا۔ ٹھیک ہے",
        "مجھے سردی لگ رہی ہے۔ گرم کپڑے پہن لیں۔ جی میں پہن لیتا ہوں۔ اچھا ہے",
        "آپ کا پسندیدہ رنگ کیا ہے۔ مجھے نیلا رنگ پسند ہے۔ مجھے سبز رنگ اچھا لگتا ہے۔ دونوں اچھے ہیں",
        "فلم کیسی تھی۔ فلم بہت اچھی تھی۔ میں بھی دیکھوں گا۔ ضرور دیکھیں",
        "آپ کتنے بجے آئیں گے۔ میں پانچ بجے آؤں گا۔ ٹھیک ہے میں انتظار کروں گا۔ شکریہ",
        "کیا آپ کو چائے پسند ہے۔ جی ہاں مجھے بہت پسند ہے۔ میں بناتا ہوں۔ بہت شکریہ",
        "یہ جگہ کہاں ہے۔ یہ شہر کے وسط میں ہے۔ وہاں کیسے جائیں۔ بس سے جا سکتے ہیں",
        "سلام، کیا حال ہے؟ وعلیکم السلام، اللہ کا شکر ہے۔ آپ سنائیں۔",
    "آپ کا نام جان سکتا ہوں؟ جی میرا نام علی ہے۔ اور آپ کا؟",
    "آج موسم کافی خوشگوار ہے۔ جی ہاں، کل کی بارش کے بعد موسم بدل گیا ہے۔",
    "شکریہ آپ کی مدد کا۔ کوئی بات نہیں، خوشی ہوئی۔",
    "معافی چاہتا ہوں، مجھے دیر ہو گئی۔ خیر ہے، ہم نے ابھی شروع کیا ہے۔",
    "یہ بس کہاں جائے گی؟ یہ بس کلاک ٹاور تک جائے گی۔",
    "انڈے کیا بھاؤ ہیں؟ انڈے تین سو روپے درجن ہیں۔",
    "میرے سر میں صبح سے درد ہے۔ آپ نے کوئی گولی لی؟",
    "چلیں چائے پینے چلتے ہیں۔ ہاں ضرور، مجھے بھی چائے کی طلب ہو رہی تھی۔",
    "آپ کا دن کیسا گزرا؟ بہت مصروف دن تھا، ابھی دفتر سے آیا ہوں۔",
    "رات کے کھانے میں کیا پکائیں؟ آج آلو گوشت بنا لیں۔",
    "کیا آپ نے کل کا میچ دیکھا؟ ہاں، کیا زبردست مقابلہ تھا۔ پاکستان جیت گیا۔",
    "مجھے نیند آ رہی ہے۔ تو سو جاؤ، صبح جلدی اٹھنا ہے۔",
    "یہ کتاب کس کی ہے؟ یہ میری ہے، آپ پڑھنا چاہیں تو لے سکتے ہیں۔",
    "آپ کا فون نمبر مل سکتا ہے؟ جی لکھ لیں، صفر تین سو۔۔۔",
    "وائی فائی کا پاس ورڈ کیا ہے؟ پاس ورڈ ہے۔",
    "اگلی چھٹیاں کب ہیں؟ اگلے مہینے عید کی چھٹیاں ہوں گی۔",
    "میں یہ کام کیسے کروں؟ رکو، میں تمہیں سمجھاتا ہوں۔",
    "آپ کو کونسا رنگ پسند ہے؟ مجھے کالا رنگ بہت پسند ہے۔",
    "یہاں قریب کوئی پارک ہے؟ جی، اس سڑک کے آخر میں ایک بڑا پارک ہے۔",
    "آپ کیا پینا پسند کریں گے؟ میرے لیے ایک گلاس پانی کافی ہے۔",
    "مجھے بھوک نہیں ہے۔ تھوڑا سا کھا لو، ورنہ طبیعت خراب ہو جائے گی۔",
    "یہ راستہ بند ہے۔ اب کیا کریں؟ ہمیں دوسرے راستے سے جانا پڑے گا۔",
    "آپ بہت اچھا بولتے ہیں۔ شکریہ، یہ آپ کی ذرہ نوازی ہے۔",
    "اس کی قیمت کچھ کم کریں۔ ٹھیک ہے، آپ کے لیے پچاس روپے کم کر لوں گا۔",
    "میں آپ سے اتفاق کرتا ہوں۔ شکریہ، یہ جان کر خوشی ہوئی۔",
    "کیا آپ میری ایک تصویر کھینچ سکتے ہیں؟ جی ضرور، فون دیجیے۔",
    "مجھے یہ جگہ بالکل پسند نہیں۔ کیوں، کیا مسئلہ ہے؟",
    "کیا میں اندر آ سکتا ہوں؟ جی، تشریف لائیے۔",
    "آپ کہاں رہتے ہیں؟ میں گلبرگ میں رہتا ہوں۔",
    "دفتر سے کب فارغ ہوں گے؟ بس ایک گھنٹے میں نکلتا ہوں۔",
    "یہ دوائی کب لینی ہے؟ یہ صبح شام کھانے کے بعد لینی ہے۔",
    "آج بہت ٹریفک جام ہے۔ ہاں، لگتا ہے کوئی حادثہ ہوا ہے۔",
    "اپنا خیال رکھنا۔ آپ بھی۔ اللہ حافظ۔",
    "کیا آپ کو مدد چاہیے؟ جی مہربانی، یہ سامان اٹھا دیں۔",
    "یہ شرٹ مجھ پر کیسی لگ رہی ہے؟ بہت اچھی لگ رہی ہے، آپ پر جچ رہی ہے۔",
    "آج میری طبیعت ٹھیک نہیں۔ ڈاکٹر کے پاس چلیں؟",
    "آپ کو گانے سننا پسند ہے؟ جی، میں موسیقی کا شوقین ہوں۔",
    "اس مسئلے کا کیا حل ہے؟ ہمیں مل کر سوچنا پڑے گا۔",
    "فکر نہ کریں، سب ٹھیک ہو جائے گا۔ آپ کی تسلی کا شکریہ۔",
    "آپ کی پسندیدہ فلم کونسی ہے؟ مجھے 'مولا جٹ' بہت پسند آئی۔",
    "کیا آپ اخبار پڑھتے ہیں؟ جی، میں روز صبح پڑھتا ہوں۔",
    "بجلی پھر چلی گئی۔ ہاں، گرمیوں میں لوڈ شیڈنگ بہت ہوتی ہے۔",
    "یہ کس نے کیا؟ مجھے نہیں معلوم، میں تو ابھی آیا ہوں۔",
    "آپ بہت محنتی ہیں۔ کامیابی کے لیے محنت ضروری ہے۔",
    "مجھے کچھ پیسے ادھار چاہئیں۔ کتنے چاہئیں؟",
    "آپ کا بہت انتظار کیا۔ معذرت، راستے میں پھنس گیا تھا۔",
    "کیا آپ سوشل میڈیا استعمال کرتے ہیں؟ جی ہاں، میں فیس بک اور انسٹاگرام استعمال کرتا ہوں۔",
    "یہ کھانا بہت مزیدار ہے۔ شکریہ، یہ میں نے خود بنایا ہے۔",
    "کیا ہم دوست بن سکتے ہیں؟ جی کیوں نہیں، مجھے خوشی ہوگی۔",
    "صبح کتنے بجے اٹھتے ہو؟ میں فجر کی نماز کے لیے اٹھتا ہوں۔",
    "آج کیا تاریخ ہے؟ آج سترہ اکتوبر ہے۔",
    "کیا آپ نے اپنا کام ختم کر لیا؟ نہیں، ابھی تھوڑا باقی ہے۔",
    "یہاں بہت شور ہے۔ چلو کسی پرسکون جگہ چلتے ہیں۔",
    "آپ کو سب سے اچھی جگہ کونسی لگتی ہے؟ مجھے شمالی علاقہ جات بہت پسند ہیں۔",
    "میں آپ کو بعد میں فون کرتا ہوں۔ ٹھیک ہے، انتظار کروں گا۔",
    "یہ آپ کا فیصلہ ہے۔ جی، میں نے بہت سوچ سمجھ کر یہ فیصلہ کیا ہے۔",
    "امتحان کی تیاری کیسی ہے؟ بس چل رہی ہے، دعا کرنا۔",
    "آپ بہت بدل گئے ہو۔ وقت انسان کو بدل دیتا ہے۔",
    "مجھے تم پر فخر ہے۔ آپ کا شکریہ، یہ سب آپ کی دعا ہے۔",
    "یہاں سگریٹ پینا منع ہے۔ اوہ، میں نے بورڈ نہیں دیکھا۔",
    "آپ کی آواز بہت پیاری ہے۔ مہربانی۔",
    "کیا آپ کو کچھ اور چاہیے؟ نہیں، بس یہی کافی ہے۔",
    "یہ میرا قصور نہیں ہے۔ تو پھر کس کی غلطی ہے؟",
    "وقت تیزی سے گزر رہا ہے۔ ہاں، پتہ ہی نہیں چلتا۔",
    "آپ کی فیملی کیسی ہے؟ الحمدللہ سب ٹھیک ہیں۔",
    "مجھے ایک نیا فون خریدنا ہے۔ کونسا ماڈل لینا ہے؟",
    "آپ بہت خوش قسمت ہیں۔ یہ سب اللہ کا کرم ہے۔",
    "اس بات کو چھوڑو۔ ٹھیک ہے، کوئی اور بات کرتے ہیں۔",
    "کیا آپ نے کبھی ہوائی سفر کیا ہے؟ جی ہاں، میں دو بار کر چکا ہوں۔",
    "مجھے آپ کا مشورہ چاہیے۔ جی ضرور، بتائیں۔",
    "یہ ایک اچھا موقع ہے۔ ہمیں اس سے فائدہ اٹھانا چاہیے۔",
    "آپ اتنے پریشان کیوں ہیں؟ بس کام کا تھوڑا دباؤ ہے۔",
    "میں یہ کبھی نہیں بھولوں گا۔ مجھے امید ہے کہ آپ یاد رکھیں گے۔",
    "کیا یہ محفوظ ہے؟ جی بالکل، فکر کی کوئی بات نہیں۔",
    "آپ کی ہمت کیسے ہوئی؟ اپنی زبان سنبھال کر بات کریں۔",
    "مجھے کچھ سمجھ نہیں آ رہا۔ میں دوبارہ وضاحت کرتا ہوں۔",
    "آپ کو کس نے بتایا؟ مجھے میرے دوست نے بتایا۔",
    "کیا یہ سچ ہے؟ جی، یہ سو فیصد سچ ہے۔",
    "آپ بہت مذاحیہ ہیں۔ شکریہ، ہنستے رہنا چاہیے۔",
    "مجھے آرام کی ضرورت ہے۔ ٹھیک ہے، آپ آرام کریں۔",
    "یہ بہت غیر منصفانہ ہے۔ میں جانتا ہوں، لیکن ہم کچھ نہیں کر سکتے۔",
    "کیا میں آپ پر بھروسہ کر سکتا ہوں؟ جی بالکل۔",
    "آپ کا تجربہ کیسا رہا؟ بہت اچھا، میں نے بہت کچھ سیکھا۔",
    "یہ بہت آسان ہے۔ ہاں، مجھے بھی ایسا ہی لگتا ہے۔",
    "آپ کو سردی لگ رہی ہے؟ جی، موسم اچانک ٹھنڈا ہو گیا ہے۔",
    "کیا میں آپ کے ساتھ چل سکتا ہوں؟ جی، خوش آمدید۔",
    "اسے اردو میں کیا کہتے ہیں؟ اسے 'کرسی' کہتے ہیں۔",
    "آپ کی لکھائی بہت خوبصورت ہے۔ شکریہ، میں نے خطاطی سیکھی ہے۔",
    "کیا آپ کو تیراکی آتی ہے؟ جی ہاں، میں ایک اچھا تیراک ہوں۔",
    "یہاں سے ریلوے اسٹیشن کتنی دور ہے؟ بس دس منٹ کا پیدل راستہ ہے۔",
    "آپ نے بہت اچھا کام کیا۔ یہ سب آپ کی رہنمائی کی وجہ سے ممکن ہوا۔",
    "کیا آپ کو کوئی اعتراض ہے؟ نہیں، مجھے کوئی اعتراض نہیں۔",
    "یہ ایک بری عادت ہے۔ میں اسے چھوڑنے کی کوشش کر رہا ہوں۔",
    "آپ کا پسندیدہ شاعر کون ہے؟ مجھے علامہ اقبال کی شاعری پسند ہے۔",
    "یہ بہت مہنگا ہے۔ اس سے سستا کچھ دکھائیں۔",
    "کیا آپ کو یقین ہے؟ جی مجھے اپنے آپ پر پورا یقین ہے۔",
    "آپ کی شادی کب ہوئی؟ میری شادی پچھلے سال ہوئی تھی۔",
    "آپ کے بچے ہیں؟ جی، میرا ایک بیٹا ہے۔",
    "اللہ آپ کو لمبی زندگی دے۔ آمین، شکریہ۔",
    "یہ رنگ مجھ پر اچھا لگے گا؟ ہاں، یہ آپ پر بہت کھلے گا۔",
    "میں آپ سے ملنا چاہتا ہوں۔ ٹھیک ہے، کل شام کو ملتے ہیں۔",
    "آپ کی پسندیدہ سبزی کونسی ہے؟ مجھے بھنڈی بہت پسند ہے۔",
    "کلاس کب شروع ہوگی؟ کلاس نو بجے شروع ہوگی۔",
    "مجھے دیر ہو رہی ہے۔ ٹھیک ہے، جلدی چلیں۔",
    "آپ کا گھر بہت پیارا ہے۔ شکریہ، تشریف لانے کا۔",
    "کبھی ہمارے ہاں بھی تشریف لائیں۔ ضرور، میں جلد ہی آؤں گا۔",
    "کیا آپ نے نماز پڑھ لی؟ جی، میں نے پڑھ لی ہے۔",
    "یہ ایک یادگار دن ہے۔ ہاں، میں اسے ہمیشہ یاد رکھوں گا۔",
    "آپ کی ہمت کی داد دیتا ہوں۔ یہ آپ کا بڑا پن ہے۔",
    "آپ بہت باصلاحیت ہیں۔ بس اللہ کا کرم ہے۔",
    "یہ ممکن نہیں ہے۔ کوشش کرنے میں کیا حرج ہے۔",
    "آپ کا مقصد کیا ہے؟ میں ایک کامیاب کاروباری بننا چاہتا ہوں۔",
    "کیا آپ کو میری بات سمجھ آئی؟ جی، میں اچھی طرح سمجھ گیا ہوں۔",
    "یہ بہت پیچیدہ ہے۔ اسے آسان بنانے کی کوشش کریں۔",
    "مجھے تم سے یہ امید نہیں تھی۔ مجھے معاف کر دو۔",
    "آپ کا مستقبل کا کیا منصوبہ ہے؟ میں بیرون ملک جا کر مزید تعلیم حاصل کرنا چاہتا ہوں۔",
    "یہ بہت اچھی خبر ہے۔ جی، میں بھی بہت خوش ہوں۔",
    "کیا آپ نے دروازہ بند کیا تھا؟ جی، میں نے تالا لگا دیا تھا۔",
    "مجھے بھوک سے زیادہ پیاس لگی ہے۔ ٹھنڈا پانی پئیں گے؟",
    "آپ کا لہجہ بہت شائستہ ہے۔ شکریہ۔",
    "یہ دوا کڑوی ہے۔ لیکن صحت کے لیے ضروری ہے۔",
    "آپ کے والد کیا کرتے ہیں؟ وہ ایک سرکاری ملازم ہیں۔",
    "کیا آپ میری ضمانت دے سکتے ہیں؟ جی، میں آپ کو اچھی طرح جانتا ہوں۔",
    "یہاں کوئی مسئلہ ہے۔ کیا ہوا، سب خیریت تو ہے؟",
    "آج میں بہت پرجوش ہوں۔ کیا خاص بات ہے؟",
    "آپ نے مجھے مایوس کیا۔ میں اپنی غلطی پر شرمندہ ہوں۔",
    "یہ ایک بہترین خیال ہے۔ ہمیں اس پر کام کرنا چاہیے۔",
    "کیا آپ نے ٹکٹ خرید لیے؟ جی، میں نے آن لائن بکنگ کروا لی ہے۔",
    "سفر کے لیے کیا تیاری ہے؟ میں نے اپنا سامان باندھ لیا ہے۔",
    "آپ کو وہاں کون ملے گا؟ وہاں میرا بھائی مجھے لینے آئے گا۔",
    "یہ ایک طویل سفر ہے۔ ہاں، تقریبا آٹھ گھنٹے لگیں گے۔",
    "کیا آپ کو ڈرائیونگ پسند ہے؟ جی، خاص طور پر لمبی ڈرائیو۔",
    "گاڑی میں پٹرول کتنا ہے؟ ٹینکی بھری ہوئی ہے۔",
    "راستے میں کہیں رکیں گے؟ جی، چائے پینے کے لیے رکیں گے۔",
    "آپ بہت ذمہ دار ہیں۔ شکریہ، مجھے اپنی ذمہ داری کا احساس ہے۔",
    "میں اس سے تنگ آ گیا ہوں۔ صبر کرو، حالات بہتر ہو جائیں گے۔",
    "یہ آپ کی مہربانی ہے۔ اس میں مہربانی کی کیا بات ہے۔",
    "کیا آپ کو کچھ یاد آیا؟ نہیں، مجھے کچھ یاد نہیں آ رہا۔",
    "آپ کی یادداشت بہت اچھی ہے۔ میں چیزیں لکھ لیتا ہوں۔",
    "یہ قانون کے خلاف ہے۔ ہمیں قانون کا احترام کرنا چاہیے۔",
    "کیا آپ نے وکیل سے بات کی؟ جی، میری ان سے کل ملاقات ہے۔",
    "مجھے انصاف چاہیے۔ انصاف ضرور ملے گا۔",
    "آپ بہت ایماندار ہیں۔ ایمانداری سب سے بہترین حکمت عملی ہے۔",
    "یہ ایک راز ہے۔ آپ کا راز میرے پاس محفوظ رہے گا۔",
    "کسی پر بھی آنکھیں بند کرکے بھروسہ نہ کریں۔ آپ ٹھیک کہہ رہے ہیں۔",
    "دنیا بہت چھوٹی ہے۔ ہاں، گھوم پھر کر وہیں آ جاتے ہیں۔",
    "آپ کا پسندیدہ مضمون کونسا تھا؟ مجھے تاریخ پڑھنا اچھا لگتا تھا۔",
    "آپ ایک اچھے طالب علم تھے۔ میں ہمیشہ اول آتا تھا۔",
    "وقت کسی کا انتظار نہیں کرتا۔ ہمیں وقت کی قدر کرنی چاہیے۔",
    "آپ نے مجھے ڈرا دیا۔ معافی چاہتا ہوں، میرا یہ مطلب نہیں تھا۔",
    "یہ ایک خواب جیسا ہے۔ ہاں، مجھے یقین نہیں آ رہا۔",
    "محنت کا پھل میٹھا ہوتا ہے۔ بے شک۔",
    "کیا آپ نے کبھی گاؤں کا دورہ کیا ہے؟ جی، میرا آبائی گاؤں بہت خوبصورت ہے۔",
    "گاؤں کی زندگی پرسکون ہوتی ہے۔ جی، شہر کے شور سے بہت دور۔",
    "وہاں تازہ ہوا ملتی ہے۔ اور خالص خوراک بھی۔",
    "آپ کی انگریزی بہت اچھی ہے۔ میں نے ایک کورس کیا تھا۔",
    "کیا آپ مجھے بھی سکھا سکتے ہیں؟ جی ضرور، شوق سے۔",
    "یہ بہت مشکل زبان ہے۔ نہیں، مشق سے سب آسان ہو جاتا ہے۔",
    "آج رات آسمان صاف ہے۔ ہاں، ستارے واضح نظر آ رہے ہیں۔",
    "کیا آپ کو فلکیات سے دلچسپی ہے؟ جی، مجھے کائنات کے بارے میں پڑھنا اچھا لگتا ہے۔",
    "یہ حیرت انگیز ہے۔ اللہ کی قدرت بے مثال ہے۔",
    "آپ کے بال بہت لمبے ہیں۔ جی، مجھے لمبے بال پسند ہیں۔",
    "آپ کونسا شیمپو استعمال کرتی ہیں؟ میں کوئی خاص شیمپو استعمال نہیں کرتی۔",
    "صحت مند غذا بالوں کے لیے اچھی ہے۔ جی، میں پھل اور سبزیاں زیادہ کھاتی ہوں۔",
    "آپ بہت صحت مند نظر آتی ہیں۔ میں روزانہ ورزش کرتی ہوں۔",
    "کونسی ورزش کرتی ہیں؟ میں صبح یوگا کرتی ہوں۔",
    "یہ ایک اچھی عادت ہے۔ اس سے ذہنی سکون بھی ملتا ہے۔",
    "کیا آپ کو پالتو جانور پسند ہیں؟ جی، مجھے بلیاں بہت پسند ہیں۔",
    "کیا آپ کے گھر میں کوئی بلی ہے؟ جی، اس کا نام مانو ہے۔",
    "جانور وفادار ہوتے ہیں۔ جی، انسانوں سے بھی زیادہ۔",
    "آپ بہت رحم دل ہیں۔ ہمیں تمام مخلوقات پر رحم کرنا چاہیے۔",
]
    transcriptions = transcription_data_usual
    
    
    print(f"Loaded {len(transcriptions)} transcriptions")
    
    # Initialize normalizer and vocabulary
    normalizer = UrduTextNormalizer()
    vocab = Vocabulary()
    
    # Build vocabulary from all transcriptions
    print("\nBuilding vocabulary...")
    for trans in transcriptions:
        cleaned = normalizer.clean(trans)
        vocab.add_sentence(cleaned)
    
    print(f"Vocabulary size: {vocab.n_words}")
    
    # Create dataset
    print("\n=== Creating Encoder-Decoder Dataset ===")
    dataset = EncoderDecoderDataset(
        transcriptions,
        vocab,
        max_len=100,
        context_size=6
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=3,
        shuffle=False,
        collate_fn=collate_fn_encoder_decoder
    )
    
    print(f"Dataset size: {len(dataset)} context-response pairs")
    
    # Initialize model
    print("\nInitializing Encoder-Decoder Transformer...")
    model = EncoderDecoderTransformer(
        vocab_size=vocab.n_words,
        d_model=256,
        n_heads=16,
        n_encoder_layers=8,
        n_decoder_layers=8,
        d_ff=1024,
        dropout=0.25
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    
    # Training setup
    criterion = nn.CrossEntropyLoss(ignore_index=0)  # Ignore padding
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)
    
    # Training loop
    print("\n=== Training ===")
    n_epochs = 40
    
    best_loss = float('inf')
    for epoch in range(n_epochs):
        train_loss = train_epoch(model, dataloader, optimizer, criterion, device)
        scheduler.step(train_loss)
        
        print(f"Epoch {epoch+1}/{n_epochs}, Loss: {train_loss:.4f}, LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Save best model
        if train_loss < best_loss:
            best_loss = train_loss
            torch.save({
                'model_state_dict': model.state_dict(),
                'vocab': vocab,
                'normalizer': normalizer,
                'epoch': epoch,
                'loss': train_loss
            }, 'urdu_encoder_decoder.pth')
            print(f"  → Model saved (best loss: {best_loss:.4f})")
    
    print(f"\nTraining Complete!")
    print(f"Best Loss: {best_loss:.4f}")
    print("Model saved as 'urdu_encoder_decoder.pth'")
    
    # Test the chatbot
    print("\n=== Testing Chatbot ===")
    test_inputs = [
        "سلام",
        "آپ کیسے ہیں",
        "آج موسم کیسا ہے",
        "شکریہ",
        "مجھے بھوک لگی ہے",
        "کیا آپ کو چائے پسند ہے"
    ]
    
    for inp in test_inputs:
        response = generate_response(model, inp, vocab, normalizer, device, 
                                    temperature=0.5, top_k=50, top_p=0.9)
        print(f"Input: {inp}")
        print(f"Response: {response}\n")
    
    # # Interactive mode
    # print("\n=== Interactive Chat (type 'exit' to quit) ===")
    # while True:
    #     user_input = input("\nآپ: ")
    #     if user_input.lower() in ['exit', 'quit', 'خارج', '']:
    #         print("خدا حافظ!")
    #         break
        
    #     response = generate_response(
    #         model, user_input, vocab, normalizer, device,
    #         temperature=0.8, top_k=40, top_p=0.92
    #     )
    #     print(f"بوٹ: {response}")


if __name__ == "__main__":
    main()