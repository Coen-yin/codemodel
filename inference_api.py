from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
from transformers import GPT2Tokenizer
import json
import os
from train_model import CustomGPTModel, ChatbotTrainer
import uvicorn

app = FastAPI(title="Custom AI Chatbot API", version="1.0.0")

# Global variables for model and tokenizer
model = None
trainer = None
tokenizer = None
device = None

class ChatRequest(BaseModel):
    message: str
    temperature: float = 0.7
    max_length: int = 100
    top_k: int = 50
    top_p: float = 0.9

class ChatResponse(BaseModel):
    response: str
    model_info: dict

class ModelConfig(BaseModel):
    vocab_size: int
    d_model: int = 768
    n_heads: int = 12
    n_layers: int = 12
    d_ff: int = 3072
    max_seq_len: int = 512

def load_model(model_path: str):
    """Load trained model and tokenizer"""
    global model, trainer, tokenizer, device
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained(model_path)
    
    # Load model configuration
    config_path = os.path.join(model_path, 'config.json')
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = json.load(f)
    else:
        # Use default configuration
        config = {
            'vocab_size': len(tokenizer),
            'd_model': 768,
            'n_heads': 12,
            'n_layers': 12,
            'd_ff': 3072,
            'max_seq_len': 512
        }
    
    # Initialize model
    model = CustomGPTModel(**config)
    
    # Load trained weights
    model_weights_path = os.path.join(model_path, 'model.pt')
    if os.path.exists(model_weights_path):
        model.load_state_dict(torch.load(model_weights_path, map_location=device))
        print(f"Loaded trained model from {model_weights_path}")
    else:
        print("Warning: No trained weights found, using randomly initialized model")
    
    # Initialize trainer for inference
    trainer = ChatbotTrainer(model, tokenizer, device)
    
    print(f"Model loaded successfully on {device}")

@app.on_event("startup")
async def startup_event():
    """Load model on startup"""
    model_path = os.environ.get('MODEL_PATH', './models/chatbot')
    if os.path.exists(model_path):
        load_model(model_path)
    else:
        print(f"Warning: Model path {model_path} not found")

@app.get("/")
async def root():
    return {"message": "Custom AI Chatbot API", "status": "running"}

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "device": str(device) if device else None
    }

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Generate chatbot response"""
    if trainer is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        response = trainer.generate_response(
            request.message,
            max_length=request.max_length,
            temperature=request.temperature,
            top_k=request.top_k,
            top_p=request.top_p
        )
        
        model_info = {
            "model_parameters": sum(p.numel() for p in model.parameters()),
            "device": str(device),
            "vocab_size": len(tokenizer)
        }
        
        return ChatResponse(response=response, model_info=model_info)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating response: {str(e)}")

@app.get("/model/info")
async def get_model_info():
    """Get model information"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return {
        "model_parameters": sum(p.numel() for p in model.parameters()),
        "vocab_size": len(tokenizer),
        "device": str(device),
        "model_architecture": {
            "d_model": model.d_model,
            "max_seq_len": model.max_seq_len,
            "n_layers": len(model.transformer_blocks)
        }
    }

@app.post("/model/reload")
async def reload_model(model_path: str = None):
    """Reload model from specified path"""
    if model_path is None:
        model_path = os.environ.get('MODEL_PATH', './models/chatbot')
    
    try:
        load_model(model_path)
        return {"message": "Model reloaded successfully", "model_path": model_path}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reloading model: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
