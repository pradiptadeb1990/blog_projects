"""
Grapheme to Phoneme conversion using the Zero2Former Transformer model.
This example demonstrates how to use the transformer model for converting written text to phonetic transcriptions.
"""

import sys
import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import nltk
from nltk.corpus import cmudict
from typing import List, Tuple, Dict
import random

# Add the parent directory to the path so we can import our transformer
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from zero2former.transformer import Transformer

# Download required NLTK data
nltk.download('cmudict')

class G2PDataset(Dataset):
    """Dataset for Grapheme to Phoneme conversion"""
    def __init__(self, data: List[Tuple[str, str]], 
                 grapheme2idx: Dict[str, int], 
                 phoneme2idx: Dict[str, int]):
        self.data = data
        self.grapheme2idx = grapheme2idx
        self.phoneme2idx = phoneme2idx
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        word, phones = self.data[idx]
        
        # Convert word to indices
        grapheme_indices = [self.grapheme2idx['<sos>']]
        grapheme_indices.extend(self.grapheme2idx[c] for c in word)
        grapheme_indices.append(self.grapheme2idx['<eos>'])
        
        # Convert phones to indices
        phoneme_indices = [self.phoneme2idx['<sos>']]
        phoneme_indices.extend(self.phoneme2idx[p] for p in phones.split())
        phoneme_indices.append(self.phoneme2idx['<eos>'])
        
        return torch.tensor(grapheme_indices), torch.tensor(phoneme_indices)

def collate_fn(batch):
    """Custom collate function to pad sequences in batch"""
    graphemes, phonemes = zip(*batch)
    
    # Pad graphemes
    grapheme_lengths = [len(x) for x in graphemes]
    max_grapheme_len = max(grapheme_lengths)
    padded_graphemes = torch.zeros((len(batch), max_grapheme_len), dtype=torch.long)
    for i, seq in enumerate(graphemes):
        padded_graphemes[i, :len(seq)] = seq
        
    # Pad phonemes
    phoneme_lengths = [len(x) for x in phonemes]
    max_phoneme_len = max(phoneme_lengths)
    padded_phonemes = torch.zeros((len(batch), max_phoneme_len), dtype=torch.long)
    for i, seq in enumerate(phonemes):
        padded_phonemes[i, :len(seq)] = seq
        
    return padded_graphemes, padded_phonemes

def prepare_data(max_samples: int = 20000):
    """Prepare CMUDict data for G2P task"""
    # Load CMUdict
    cmu = cmudict.dict()
    
    # Prepare data
    data = []
    for word, pronunciations in cmu.items():
        for phones in pronunciations:
            data.append((word, ' '.join(phones)))
    
    # Limit dataset size
    data = data[:max_samples]
    
    # Build vocabularies
    grapheme_vocab = set(''.join(word for word, _ in data))
    phoneme_vocab = set(' '.join(phones for _, phones in data).split())
    
    grapheme_vocab = ['<pad>', '<sos>', '<eos>'] + sorted(list(grapheme_vocab))
    phoneme_vocab = ['<pad>', '<sos>', '<eos>'] + sorted(list(phoneme_vocab))
    
    grapheme2idx = {char: idx for idx, char in enumerate(grapheme_vocab)}
    phoneme2idx = {phone: idx for idx, phone in enumerate(phoneme_vocab)}
    
    idx2grapheme = {idx: char for char, idx in grapheme2idx.items()}
    idx2phoneme = {idx: phone for phone, idx in phoneme2idx.items()}
    
    return (data, grapheme2idx, phoneme2idx, 
            idx2grapheme, idx2phoneme,
            len(grapheme_vocab), len(phoneme_vocab))

def evaluate(model, val_loader, criterion, device):
    """Evaluate the model on validation data"""
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for src, tgt in val_loader:
            src, tgt = src.to(device), tgt.to(device)
            
            # Create source and target masks
            src_mask = model.create_padding_mask(src)
            tgt_mask = model.create_padding_mask(tgt[:, :-1])
            
            output = model(src, tgt[:, :-1], source_padding_mask=src_mask, target_padding_mask=tgt_mask)
            
            output = output.view(-1, output.size(-1))
            target = tgt[:, 1:].contiguous().view(-1)
            
            loss = criterion(output, target)
            total_loss += loss.item()
    
    avg_loss = total_loss / len(val_loader)
    return avg_loss

def train_g2p(model, train_loader, val_loader, criterion, optimizer, device, num_epochs=10):
    """Train the G2P model with validation"""
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        total_loss = 0
        for batch_idx, (src, tgt) in enumerate(train_loader):
            src, tgt = src.to(device), tgt.to(device)

            # Create source and target masks
            src_mask = model.create_padding_mask(src)
            tgt_mask = model.create_padding_mask(tgt[:, :-1])
            
            optimizer.zero_grad()
            output = model(src, tgt[:, :-1], source_padding_mask=src_mask, target_padding_mask=tgt_mask)  # Remove last token from target for teacher forcing
            
            # Reshape output and target for loss calculation
            output = output.view(-1, output.size(-1))
            target = tgt[:, 1:].contiguous().view(-1)  # Remove first token (<sos>) from target
            
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            if batch_idx % 100 == 0:
                print(f'Epoch: {epoch+1}, Batch: {batch_idx}, Loss: {loss.item():.4f}')
        
        avg_train_loss = total_loss / len(train_loader)
        
        # Validation phase
        val_loss = evaluate(model, val_loader, criterion, device)
        
        print(f'Epoch: {epoch+1}, Train Loss: {avg_train_loss:.4f}, Val Loss: {val_loss:.4f}')
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': val_loss,
            }, 'g2p_model_best.pt')

def predict(model, word: str, grapheme2idx: Dict[str, int], 
           idx2phoneme: Dict[int, str], device, max_len: int = 50):
    """Predict phonemes for a given word"""
    model.eval()
    
    # Prepare input
    grapheme_indices = [grapheme2idx['<sos>']]
    grapheme_indices.extend(grapheme2idx[c] for c in word)
    grapheme_indices.append(grapheme2idx['<eos>'])
    src = torch.tensor([grapheme_indices]).to(device)
    
    # Initialize target with <sos>
    tgt = torch.tensor([[grapheme2idx['<sos>']]]).to(device)
    
    with torch.no_grad():
        for _ in range(max_len):
            # Create source and target masks
            src_mask = model.create_padding_mask(src)
            tgt_mask = model.create_padding_mask(tgt)
            
            output = model(src, tgt, source_padding_mask=src_mask, target_padding_mask=tgt_mask)
            next_token = output[:, -1].argmax(dim=-1, keepdim=True)
            tgt = torch.cat([tgt, next_token], dim=1)
            
            if next_token.item() == grapheme2idx['<eos>']:
                break
    
    # Convert indices to phonemes
    phonemes = []
    for idx in tgt[0][1:]:  # Skip <sos>
        phoneme = idx2phoneme[idx.item()]
        if phoneme == '<eos>':
            break
        phonemes.append(phoneme)
    
    return ' '.join(phonemes)

def main():
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    random.seed(42)
    
    # Prepare data
    print("Preparing data...")
    (data, grapheme2idx, phoneme2idx, 
     idx2grapheme, idx2phoneme,
     grapheme_vocab_size, phoneme_vocab_size) = prepare_data()
    
    # Split data into train/val
    train_size = int(0.9 * len(data))
    train_data = data[:train_size]
    val_data = data[train_size:]
    
    # Create datasets
    train_dataset = G2PDataset(train_data, grapheme2idx, phoneme2idx)
    val_dataset = G2PDataset(val_data, grapheme2idx, phoneme2idx)
    
    # Create dataloaders
    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, 
                            shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, 
                          shuffle=False, collate_fn=collate_fn)
    
    # Initialize model
    # Check for available devices - CUDA, MPS (Mac M1/M2), or CPU
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    print(f"Using device: {device}")
    model = Transformer(
        source_vocab_size=grapheme_vocab_size,
        target_vocab_size=phoneme_vocab_size,
        model_dim=256,
        num_heads=8,
        num_encoder_layers=3,
        num_decoder_layers=3,
        ffn_dim=1024,
        dropout=0.1,
        device=device
    ).to(device)
    
    # Define loss and optimizer
    criterion = nn.CrossEntropyLoss(ignore_index=0)  # ignore padding index
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    
    # Train the model
    print("Training model...")
    train_g2p(model, train_loader, val_loader, criterion, optimizer, device, num_epochs=30)
    
    # Save the trained model
    torch.save({
        'model_state_dict': model.state_dict(),
        'grapheme2idx': grapheme2idx,
        'phoneme2idx': phoneme2idx,
        'idx2grapheme': idx2grapheme,
        'idx2phoneme': idx2phoneme
    }, 'g2p_model.pt')
    
    # Test some predictions
    test_words = ['hello', 'world', 'transformer']
    print("\nTesting predictions:")
    for word in test_words:
        phonemes = predict(model, word, grapheme2idx, idx2phoneme, device)
        print(f"{word} -> {phonemes}")

if __name__ == '__main__':
    main()
