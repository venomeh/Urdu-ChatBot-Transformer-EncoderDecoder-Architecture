import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import math
import re
from collections import Counter
import numpy as np
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
import sacrebleu

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
            if idx in [self.word2idx['<EOS>'], self.word2idx['<PAD>']]:
                break
            if idx != self.word2idx['<SOS>']:
                words.append(self.idx2word.get(idx, '<UNK>'))
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
            utterances = trans.split('۔')
            utterances = [u.strip() for u in utterances if u.strip()]
            
            for i in range(len(utterances) - 1):
                context_start = max(0, i - context_size + 1)
                context = ' '.join(utterances[context_start:i+1])
                response = utterances[i + 1]
                
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


# ==================== METRICS ====================

class MetricsCalculator:
    """Calculate BLEU, ROUGE-L, chrF, and Perplexity metrics"""
    
    def __init__(self):
        self.rouge_scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=False)
        self.smoothing = SmoothingFunction()
    
    def calculate_bleu(self, reference, hypothesis):
        """Calculate BLEU score"""
        ref_tokens = reference.split()
        hyp_tokens = hypothesis.split()
        
        if not hyp_tokens or not ref_tokens:
            return 0.0
        
        try:
            score = sentence_bleu(
                [ref_tokens], 
                hyp_tokens,
                smoothing_function=self.smoothing.method1
            )
            return score
        except:
            return 0.0
    
    def calculate_rouge_l(self, reference, hypothesis):
        """Calculate ROUGE-L score"""
        if not hypothesis or not reference:
            return 0.0
        
        try:
            scores = self.rouge_scorer.score(reference, hypothesis)
            return scores['rougeL'].fmeasure
        except:
            return 0.0
    
    def calculate_chrf(self, reference, hypothesis):
        """Calculate chrF score using sacrebleu"""
        if not hypothesis or not reference:
            return 0.0
        
        try:
            score = sacrebleu.sentence_chrf(hypothesis, [reference])
            return score.score / 100.0  # Normalize to 0-1
        except:
            return 0.0
    
    def calculate_perplexity(self, loss):
        """Calculate perplexity from cross-entropy loss"""
        try:
            return math.exp(loss)
        except OverflowError:
            return float('inf')


# ==================== TRAINING & EVALUATION ====================

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
        output = output.view(-1, output.size(-1))
        decoder_target = decoder_target.view(-1)
        loss = criterion(output, decoder_target)
        
        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        total_loss += loss.item()
    
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
            
            output = model(encoder_input, decoder_input)
            
            output = output.view(-1, output.size(-1))
            decoder_target = decoder_target.view(-1)
            loss = criterion(output, decoder_target)
            
            total_loss += loss.item()
    
    return total_loss / len(dataloader)


def evaluate_metrics(model, dataloader, vocab, normalizer, device, metrics_calc, max_samples=100):
    """Evaluate model with BLEU, ROUGE-L, and chrF metrics"""
    model.eval()
    
    bleu_scores = []
    rouge_scores = []
    chrf_scores = []
    
    sample_count = 0
    
    with torch.no_grad():
        for encoder_input, decoder_input, decoder_target in dataloader:
            batch_size = encoder_input.size(0)
            
            for i in range(batch_size):
                if sample_count >= max_samples:
                    break
                
                # Generate response
                enc_input = encoder_input[i:i+1].to(device)
                generated_response = generate_response_from_input(
                    model, enc_input, vocab, device, max_len=50
                )
                
                # Get reference response
                target_indices = decoder_target[i].cpu().numpy()
                reference_response = vocab.indices_to_sentence(target_indices)
                
                if generated_response and reference_response:
                    # Calculate metrics
                    bleu = metrics_calc.calculate_bleu(reference_response, generated_response)
                    rouge = metrics_calc.calculate_rouge_l(reference_response, generated_response)
                    chrf = metrics_calc.calculate_chrf(reference_response, generated_response)
                    
                    bleu_scores.append(bleu)
                    rouge_scores.append(rouge)
                    chrf_scores.append(chrf)
                
                sample_count += 1
            
            if sample_count >= max_samples:
                break
    
    avg_bleu = np.mean(bleu_scores) if bleu_scores else 0.0
    avg_rouge = np.mean(rouge_scores) if rouge_scores else 0.0
    avg_chrf = np.mean(chrf_scores) if chrf_scores else 0.0
    
    return avg_bleu, avg_rouge, avg_chrf


# ==================== INFERENCE ====================

def generate_response_from_input(model, encoder_input, vocab, device, max_len=50):
    """Generate response from encoder input tensor"""
    model.eval()
    
    with torch.no_grad():
        src_mask = model.generate_padding_mask(encoder_input)
        encoder_output = model.encode(encoder_input, src_mask)
    
    decoder_input = [vocab.word2idx['<SOS>']]
    
    for _ in range(max_len):
        decoder_tensor = torch.tensor([decoder_input]).to(device)
        
        with torch.no_grad():
            tgt_mask = model.generate_padding_mask(decoder_tensor)
            seq_len = decoder_tensor.size(1)
            causal_mask = model.generate_causal_mask(seq_len, decoder_tensor.device)
            tgt_mask = tgt_mask & causal_mask
            
            decoder_output = model.decode(decoder_tensor, encoder_output, src_mask, tgt_mask)
            output = model.fc_out(decoder_output)
        
        next_token = output[0, -1].argmax().item()
        decoder_input.append(next_token)
        
        if next_token == vocab.word2idx['<EOS>']:
            break
    
    response_indices = decoder_input[1:]
    response = vocab.indices_to_sentence(response_indices)
    
    return response


def generate_response(model, input_text, vocab, normalizer, device, 
                     max_len=50, temperature=0.7, top_k=50, top_p=0.9):
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
            tgt_mask = model.generate_padding_mask(decoder_tensor)
            seq_len = decoder_tensor.size(1)
            causal_mask = model.generate_causal_mask(seq_len, decoder_tensor.device)
            tgt_mask = tgt_mask & causal_mask
            
            decoder_output = model.decode(decoder_tensor, encoder_output, src_mask, tgt_mask)
            output = model.fc_out(decoder_output)
        
        # Get logits for last position
        logits = output[0, -1] / temperature
        
        # Mask out special tokens
        logits[vocab.word2idx['<PAD>']] = -float('Inf')
        logits[vocab.word2idx['<SOS>']] = -float('Inf')
        logits[vocab.word2idx['<UNK>']] = -float('Inf')
        
        # Apply top-k filtering
        if top_k > 0:
            indices_to_remove = logits < torch.topk(logits, min(top_k, logits.size(-1)))[0][..., -1, None]
            logits[indices_to_remove] = -float('Inf')
        
        # Apply top-p filtering
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
    
    # Convert to sentence
    response_indices = decoder_input[1:]
    response = vocab.indices_to_sentence(response_indices)
    
    return response if response else "معذرت، میں سمجھ نہیں سکا"


# ==================== DATA SPLIT ====================

def split_data(transcriptions, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
    """Split data into train/validation/test sets"""
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1.0"
    
    np.random.seed(42)
    np.random.shuffle(transcriptions)
    
    n = len(transcriptions)
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)
    
    train_data = transcriptions[:train_end]
    val_data = transcriptions[train_end:val_end]
    test_data = transcriptions[val_end:]
    
    return train_data, val_data, test_data


# ==================== MAIN ====================

def main():
    print("=== Urdu Encoder-Decoder Transformer with Metrics ===\n")
    
    # Load data from corpus.txt
    print("Loading data from corpus.txt...")
    try:
        with open('corpus.txt', 'r', encoding='utf-8') as f:
            transcriptions = [line.strip() for line in f if line.strip()]
        print(f"Loaded {len(transcriptions)} transcriptions from corpus.txt")
    except FileNotFoundError:
        print("corpus.txt not found! Using sample data...")
        transcriptions = [
            "سلام کیسے ہیں آپ۔ میں بالکل ٹھیک ہوں شکریہ۔ آج موسم بہت اچھا ہے۔ جی ہاں بہت خوشگوار موسم ہے",
            "آپ کا نام کیا ہے۔ میرا نام احمد ہے۔ آپ کہاں سے ہیں۔ میں کراچی سے ہوں",
            "آج آپ کیا کر رہے ہیں۔ میں کام کر رہا ہوں۔ کیا آپ کو مدد چاہیے۔ جی نہیں شکریہ",
        ]
    
    # Split data: 80% train, 10% validation, 10% test
    train_data, val_data, test_data = split_data(transcriptions, 0.8, 0.1, 0.1)
    print(f"\nData Split:")
    print(f"  Training: {len(train_data)} samples ({len(train_data)/len(transcriptions)*100:.1f}%)")
    print(f"  Validation: {len(val_data)} samples ({len(val_data)/len(transcriptions)*100:.1f}%)")
    print(f"  Test: {len(test_data)} samples ({len(test_data)/len(transcriptions)*100:.1f}%)")
    
    # Initialize normalizer and vocabulary
    normalizer = UrduTextNormalizer()
    vocab = Vocabulary()
    
    # Build vocabulary from training data only
    print("\nBuilding vocabulary from training data...")
    for trans in train_data:
        utterances = normalizer.split_into_utterances(trans)
        for utterance in utterances:
            cleaned = normalizer.clean(utterance)
            vocab.add_sentence(cleaned)
    
    print(f"Vocabulary size: {vocab.n_words}")
    
    # Create datasets
    print("\nCreating datasets...")
    train_dataset = EncoderDecoderDataset(train_data, vocab, max_len=100, context_size=2)
    val_dataset = EncoderDecoderDataset(val_data, vocab, max_len=100, context_size=2)
    test_dataset = EncoderDecoderDataset(test_data, vocab, max_len=100, context_size=2)
    
    print(f"  Training pairs: {len(train_dataset)}")
    print(f"  Validation pairs: {len(val_dataset)}")
    print(f"  Test pairs: {len(test_dataset)}")
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=8,
        shuffle=True,
        collate_fn=collate_fn_encoder_decoder
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=8,
        shuffle=False,
        collate_fn=collate_fn_encoder_decoder
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=8,
        shuffle=False,
        collate_fn=collate_fn_encoder_decoder
    )
    
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
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)
    
    # Initialize metrics calculator
    metrics_calc = MetricsCalculator()
    
    # Training loop
    print("\n" + "="*70)
    print("TRAINING")
    print("="*70)
    
    n_epochs = 50
    best_val_loss = float('inf')
    
    for epoch in range(n_epochs):
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        
        # Validate
        val_loss = validate_epoch(model, val_loader, criterion, device)
        
        # Calculate perplexity
        train_ppl = metrics_calc.calculate_perplexity(train_loss)
        val_ppl = metrics_calc.calculate_perplexity(val_loss)
        
        # Learning rate scheduler
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        
        print(f"Epoch {epoch+1}/{n_epochs}")
        print(f"  Train Loss: {train_loss:.4f} | Train PPL: {train_ppl:.2f}")
        print(f"  Val Loss:   {val_loss:.4f} | Val PPL:   {val_ppl:.2f}")
        print(f"  LR: {current_lr:.6f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'vocab': vocab,
                'normalizer': normalizer,
                'val_loss': val_loss,
            }, 'best_model.pth')
            print(f"  ✓ Best model saved!")
        
        print()
    
    # Load best model
    print("Loading best model for evaluation...")
    checkpoint = torch.load('best_model.pth', map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Evaluate on test set
    print("\n" + "="*70)
    print("EVALUATION ON TEST SET")
    print("="*70)
    
    test_loss = validate_epoch(model, test_loader, criterion, device)
    test_ppl = metrics_calc.calculate_perplexity(test_loss)
    
    print(f"\nTest Loss: {test_loss:.4f}")
    print(f"Test Perplexity: {test_ppl:.2f}")
    
    print("\nCalculating BLEU, ROUGE-L, and chrF scores...")
    test_bleu, test_rouge, test_chrf = evaluate_metrics(
        model, test_loader, vocab, normalizer, device, metrics_calc, max_samples=100
    )
    
    print("\n" + "="*70)
    print("FINAL METRICS")
    print("="*70)
    print(f"Test Loss:       {test_loss:.4f}")
    print(f"Test Perplexity: {test_ppl:.2f}")
    print(f"BLEU Score:      {test_bleu:.4f}")
    print(f"ROUGE-L Score:   {test_rouge:.4f}")
    print(f"chrF Score:      {test_chrf:.4f}")
    print("="*70)
    
    # Save final model
    torch.save({
        'model_state_dict': model.state_dict(),
        'vocab': vocab,
        'normalizer': normalizer,
        'metrics': {
            'test_loss': test_loss,
            'test_ppl': test_ppl,
            'bleu': test_bleu,
            'rouge_l': test_rouge,
            'chrf': test_chrf
        }
    }, 'urdu_encoder_decoder_final.pth')
    print("\nFinal model saved as 'urdu_encoder_decoder_final.pth'")
    
    # Test the chatbot
    print("\n" + "="*70)
    print("TESTING CHATBOT")
    print("="*70)
    
    test_inputs = [
        "سلام",
        "آپ کیسے ہیں",
        "آج موسم کیسا ہے",
        "شکریہ",
        "مجھے بھوک لگی ہے",
        "کیا آپ کو چائے پسند ہے"
    ]
    
    for inp in test_inputs:
        response = generate_response(model, inp, vocab, normalizer, device)
        print(f"آپ: {inp}")
        print(f"بوٹ: {response}")
        print()


if __name__ == "__main__":
    main()
