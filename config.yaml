# Model Configuration
model:
  d_model: 768
  n_heads: 12
  n_layers: 12
  d_ff: 3072
  max_seq_len: 512
  dropout: 0.1
  vocab_size: null  # Will be set based on tokenizer

# Training Configuration
training:
  epochs: 10
  batch_size: 8
  learning_rate: 0.0005
  weight_decay: 0.01
  gradient_clip: 1.0
  warmup_steps: 1000
  save_steps: 500
  eval_steps: 250

# Data Configuration
data:
  max_length: 512
  train_split: 0.8
  val_split: 0.1
  test_split: 0.1

# Generation Configuration
generation:
  max_length: 100
  temperature: 0.7
  top_k: 50
  top_p: 0.9
  repetition_penalty: 1.1

# Paths
paths:
  data_dir: "./data"
  model_dir: "./models"
  logs_dir: "./logs"
  checkpoints_dir: "./checkpoints"
