#!/bin/bash

# Create directories
mkdir -p data models logs checkpoints

# 1. Create sample training data
echo "Creating sample training data..."
python prepare_data.py --create_sample --output_file data/training_data.json

# 2. Train the model
echo "Starting model training..."
python train_model.py \
    --data_path data/training_data.json \
    --model_save_path models/chatbot \
    --epochs 10 \
    --batch_size 8 \
    --learning_rate 0.0005 \
    --d_model 512 \
    --n_heads 8 \
    --n_layers 6

# 3. Start the API server
echo "Starting API server..."
MODEL_PATH=./models/chatbot python inference_api.py

echo "Training and deployment complete!"
echo "API available at: http://localhost:8000"
echo "Dashboard available at: dashboard.html"
