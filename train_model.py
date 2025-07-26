import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import json
import numpy as np
import matplotlib.pyplot as plt
from transformers import GPT2Tokenizer
import os
from datetime import datetime
import argparse
from tqdm import tqdm
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ChatDataset(Dataset):
    def __init__(self, conversations, tokenizer, max_length=512):
        self.conversations = conversations
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.conversations)
    
    def __getitem__(self, idx):
        conversation = self.conversations[idx]
        
        # Format: <user>message</user><bot>response</bot>
        text = f"<user>{conversation['input']}</user><bot>{conversation['output']}</bot>"
        
        # Tokenize
        encoded = self.tokenizer.encode(text, truncation=True, max_length=self.max_length)
        
        # Convert to tensor
        input_ids = torch.tensor(encoded[:-1], dtype=torch.long)  # Input
        targets = torch.tensor(encoded[1:], dtype=torch.long)     # Targets (shifted by 1)
        
        # Pad sequences
        if len(input_ids) < self.max_length - 1:
            padding_length = self.max_length - 1 - len(input_ids)
            input_ids = torch.cat([input_ids, torch.full((padding_length,), self.tokenizer.pad_token_id)])
            targets = torch.cat([targets, torch.full((padding_length,), -100)])  # -100 is ignored in loss
            
        return {
            'input_ids': input_ids,
            'targets': targets,
            'attention_mask': (input_ids != self.tokenizer.pad_token_id).long()
        }

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        
    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        
        # Linear transformations and split into heads
        Q = self.w_q(query).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        K = self.w_k(key).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        V = self.w_v(value).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        
        # Attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.d_k)
        
        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(1)
            scores.masked_fill_(mask == 0, -1e9)
        
        # Causal mask (for autoregressive generation)
        seq_len = scores.size(-1)
        causal_mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
        scores.masked_fill_(causal_mask.unsqueeze(0).unsqueeze(0).to(scores.device), -1e9)
        
        attention_weights = torch.softmax(scores, dim=-1)
        attention_output = torch.matmul(attention_weights, V)
        
        # Concatenate heads
        attention_output = attention_output.transpose(1, 2).contiguous().view(
            batch_size, -1, self.d_model
        )
        
        return self.w_o(attention_output)

class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, n_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model)
        )
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        # Self-attention with residual connection
        attn_output = self.attention(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # Feed-forward with residual connection
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        
        return x

class CustomGPTModel(nn.Module):
    def __init__(self, vocab_size, d_model=768, n_heads=12, n_layers=12, d_ff=3072, max_seq_len=512, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        
        # Embeddings
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_seq_len, d_model)
        
        # Transformer blocks
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        
        # Output layer
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            
    def forward(self, input_ids, attention_mask=None):
        batch_size, seq_len = input_ids.shape
        
        # Create position ids
        position_ids = torch.arange(seq_len, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
        
        # Embeddings
        token_emb = self.token_embedding(input_ids)
        pos_emb = self.position_embedding(position_ids)
        x = token_emb + pos_emb
        
        # Pass through transformer blocks
        for block in self.transformer_blocks:
            x = block(x, attention_mask)
            
        # Final layer norm and output projection
        x = self.ln_f(x)
        logits = self.lm_head(x)
        
        return logits

class ChatbotTrainer:
    def __init__(self, model, tokenizer, device):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.model.to(device)
        
        self.train_losses = []
        self.val_losses = []
        
    def train_epoch(self, dataloader, optimizer, criterion):
        self.model.train()
        total_loss = 0
        
        progress_bar = tqdm(dataloader, desc="Training")
        for batch in progress_bar:
            input_ids = batch['input_ids'].to(self.device)
            targets = batch['targets'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            
            optimizer.zero_grad()
            
            # Forward pass
            logits = self.model(input_ids, attention_mask)
            
            # Calculate loss
            loss = criterion(logits.view(-1, logits.size(-1)), targets.view(-1))
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            optimizer.step()
            
            total_loss += loss.item()
            progress_bar.set_postfix({'loss': loss.item()})
            
        return total_loss / len(dataloader)
    
    def validate(self, dataloader, criterion):
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Validating"):
                input_ids = batch['input_ids'].to(self.device)
                targets = batch['targets'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                
                logits = self.model(input_ids, attention_mask)
                loss = criterion(logits.view(-1, logits.size(-1)), targets.view(-1))
                
                total_loss += loss.item()
                
        return total_loss / len(dataloader)
    
    def generate_response(self, prompt, max_length=100, temperature=0.7, top_k=50, top_p=0.9):
        self.model.eval()
        
        # Tokenize prompt
        input_text = f"<user>{prompt}</user><bot>"
        input_ids = torch.tensor([self.tokenizer.encode(input_text)], device=self.device)
        
        with torch.no_grad():
            for _ in range(max_length):
                # Get logits
                logits = self.model(input_ids)
                next_token_logits = logits[0, -1, :] / temperature
                
                # Apply top-k filtering
                if top_k > 0:
                    indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                    next_token_logits[indices_to_remove] = -float('inf')
                
                # Apply top-p filtering
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                    cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    indices_to_remove = sorted_indices_to_remove.scatter(0, sorted_indices, sorted_indices_to_remove)
                    next_token_logits[indices_to_remove] = -float('inf')
                
                # Sample next token
                probs = torch.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                
                # Add to sequence
                input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=1)
                
                # Check for end token
                if next_token.item() == self.tokenizer.encode('</bot>')[0]:
                    break
        
        # Decode response
        generated_text = self.tokenizer.decode(input_ids[0], skip_special_tokens=False)
        
        # Extract bot response
        if '<bot>' in generated_text and '</bot>' in generated_text:
            response = generated_text.split('<bot>')[1].split('</bot>')[0].strip()
        else:
            response = generated_text.split('<bot>')[-1].strip()
            
        return response

def load_training_data(data_path):
    """Load training data from JSON file"""
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def save_model(model, tokenizer, save_path):
    """Save model and tokenizer"""
    os.makedirs(save_path, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(save_path, 'model.pt'))
    tokenizer.save_pretrained(save_path)
    
def plot_training_curves(train_losses, val_losses, save_path):
    """Plot and save training curves"""
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_path, 'training_curves.png'))
    plt.close()

def main():
    parser = argparse.ArgumentParser(description='Train Custom GPT Chatbot')
    parser.add_argument('--data_path', type=str, required=True, help='Path to training data JSON file')
    parser.add_argument('--model_save_path', type=str, default='./models/chatbot', help='Path to save trained model')
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=5e-4, help='Learning rate')
    parser.add_argument('--max_length', type=int, default=512, help='Maximum sequence length')
    parser.add_argument('--d_model', type=int, default=768, help='Model dimension')
    parser.add_argument('--n_heads', type=int, default=12, help='Number of attention heads')
    parser.add_argument('--n_layers', type=int, default=12, help='Number of transformer layers')
    
    args = parser.parse_args()
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Load tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    
    # Add special tokens
    special_tokens = ['<user>', '</user>', '<bot>', '</bot>']
    tokenizer.add_tokens(special_tokens)
    
    # Load training data
    logger.info("Loading training data...")
    training_data = load_training_data(args.data_path)
    
    # Split data
    train_size = int(0.8 * len(training_data))
    train_data = training_data[:train_size]
    val_data = training_data[train_size:]
    
    # Create datasets
    train_dataset = ChatDataset(train_data, tokenizer, args.max_length)
    val_dataset = ChatDataset(val_data, tokenizer, args.max_length)
    
    # Create dataloaders
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    
    # Initialize model
    model = CustomGPTModel(
        vocab_size=len(tokenizer),
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        max_seq_len=args.max_length
    )
    
    # Initialize trainer
    trainer = ChatbotTrainer(model, tokenizer, device)
    
    # Setup training
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss(ignore_index=-100)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    # Training loop
    logger.info("Starting training...")
    best_val_loss = float('inf')
    
    for epoch in range(args.epochs):
        logger.info(f"Epoch {epoch + 1}/{args.epochs}")
        
        # Train
        train_loss = trainer.train_epoch(train_dataloader, optimizer, criterion)
        trainer.train_losses.append(train_loss)
        
        # Validate
        val_loss = trainer.validate(val_dataloader, criterion)
        trainer.val_losses.append(val_loss)
        
        # Update learning rate
        scheduler.step()
        
        logger.info(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_model(model, tokenizer, args.model_save_path)
            logger.info("Saved new best model!")
        
        # Test generation
        if (epoch + 1) % 2 == 0:
            test_prompt = "Hello, how are you?"
            response = trainer.generate_response(test_prompt)
            logger.info(f"Test generation - Input: '{test_prompt}' -> Output: '{response}'")
    
    # Plot training curves
    plot_training_curves(trainer.train_losses, trainer.val_losses, args.model_save_path)
    
    logger.info("Training completed!")
    logger.info(f"Model saved to: {args.model_save_path}")

if __name__ == "__main__":
    main()
