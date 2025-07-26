import json
import re
import argparse
from typing import List, Dict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataProcessor:
    def __init__(self):
        self.conversations = []
        
    def clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Remove special characters that might interfere
        text = re.sub(r'[<>]', '', text)
        
        return text
    
    def process_conversation_file(self, file_path: str, format_type: str = 'json'):
        """Process conversation files in different formats"""
        
        if format_type == 'json':
            self._process_json_file(file_path)
        elif format_type == 'txt':
            self._process_txt_file(file_path)
        elif format_type == 'csv':
            self._process_csv_file(file_path)
        else:
            raise ValueError(f"Unsupported format: {format_type}")
    
    def _process_json_file(self, file_path: str):
        """Process JSON conversation file"""
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if isinstance(data, list):
            for item in data:
                if 'input' in item and 'output' in item:
                    self.conversations.append({
                        'input': self.clean_text(item['input']),
                        'output': self.clean_text(item['output']),
                        'category': item.get('category', 'general')
                    })
        
        logger.info(f"Processed {len(data)} conversations from {file_path}")
    
    def _process_txt_file(self, file_path: str):
        """Process text file with conversation pairs"""
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Split by double newlines to separate conversations
        conversations = content.split('\n\n')
        
        for conv in conversations:
            lines = conv.strip().split('\n')
            if len(lines) >= 2:
                # Assume first line is user, second is bot
                user_msg = lines[0].replace('User:', '').replace('Human:', '').strip()
                bot_msg = lines[1].replace('Bot:', '').replace('Assistant:', '').strip()
                
                if user_msg and bot_msg:
                    self.conversations.append({
                        'input': self.clean_text(user_msg),
                        'output': self.clean_text(bot_msg),
                        'category': 'general'
                    })
        
        logger.info(f"Processed {len(conversations)} conversations from {file_path}")
    
    def _process_csv_file(self, file_path: str):
        """Process CSV file with conversation pairs"""
        import csv
        
        with open(file_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if 'input' in row and 'output' in row:
                    self.conversations.append({
                        'input': self.clean_text(row['input']),
                        'output': self.clean_text(row['output']),
                        'category': row.get('category', 'general')
                    })
        
        logger.info(f"Processed conversations from {file_path}")
    
    def add_manual_conversations(self, manual_data: List[Dict]):
        """Add manually created conversations"""
        for item in manual_data:
            self.conversations.append({
                'input': self.clean_text(item['input']),
                'output': self.clean_text(item['output']),
                'category': item.get('category', 'general')
            })
    
    def augment_data(self):
        """Apply data augmentation techniques"""
        augmented = []
        
        for conv in self.conversations:
            # Original conversation
            augmented.append(conv)
            
            # Add variations (simple paraphrasing)
            variations = self._create_variations(conv)
            augmented.extend(variations)
        
        self.conversations = augmented
        logger.info(f"Data augmentation completed. Total conversations: {len(self.conversations)}")
    
    def _create_variations(self, conversation: Dict) -> List[Dict]:
        """Create variations of a conversation"""
        variations = []
        
        input_text = conversation['input']
        output_text = conversation['output']
        
        # Simple variations
        input_variations = [
            input_text.replace('?', ''),
            input_text.replace('!', ''),
            input_text.lower(),
            input_text.capitalize()
        ]
        
        for variation in input_variations:
            if variation != input_text and len(variation.strip()) > 0:
                variations.append({
                    'input': variation.strip(),
                    'output': output_text,
                    'category': conversation['category']
                })
        
        return variations[:2]  # Limit variations
    
    def save_processed_data(self, output_path: str):
        """Save processed conversations to JSON file"""
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(self.conversations, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved {len(self.conversations)} conversations to {output_path}")
    
    def get_statistics(self):
        """Get statistics about the processed data"""
        if not self.conversations:
            return {}
        
        total = len(self.conversations)
        categories = {}
        input_lengths = []
        output_lengths = []
        
        for conv in self.conversations:
            cat = conv['category']
            categories[cat] = categories.get(cat, 0) + 1
            input_lengths.append(len(conv['input'].split()))
            output_lengths.append(len(conv['output'].split()))
        
        return {
            'total_conversations': total,
            'categories': categories,
            'avg_input_length': sum(input_lengths) / len(input_lengths),
            'avg_output_length': sum(output_lengths) / len(output_lengths),
            'max_input_length': max(input_lengths),
            'max_output_length': max(output_lengths)
        }

def create_sample_data():
    """Create sample training data"""
    sample_conversations = [
        {
            "input": "Hello, how are you?",
            "output": "Hello! I'm doing well, thank you for asking. How are you doing today?",
            "category": "greeting"
        },
        {
            "input": "What's your name?",
            "output": "I'm an AI chatbot created to help and assist you. You can call me ChatBot!",
            "category": "identity"
        },
        {
            "input": "Can you help me with programming?",
            "output": "Of course! I'd be happy to help you with programming. What specific programming language or topic would you like assistance with?",
            "category": "programming"
        },
        {
            "input": "Tell me a joke",
            "output": "Why don't scientists trust atoms? Because they make up everything! 😄",
            "category": "entertainment"
        },
        {
            "input": "What's the weather like?",
            "output": "I don't have access to real-time weather data, but I'd recommend checking a weather app or website for current conditions in your area!",
            "category": "information"
        },
        {
            "input": "How do I learn Python?",
            "output": "Great choice! To learn Python, I recommend: 1) Start with online tutorials, 2) Practice with small projects, 3) Join coding communities, 4) Work on real-world projects. Would you like specific resource recommendations?",
            "category": "programming"
        },
        {
            "input": "What can you do?",
            "output": "I can help with various tasks like answering questions, providing explanations, assisting with programming, having conversations, and more. What would you like help with today?",
            "category": "capabilities"
        },
        {
            "input": "Thank you",
            "output": "You're very welcome! I'm glad I could help. Feel free to ask if you have any other questions!",
            "category": "gratitude"
        },
        {
            "input": "Goodbye",
            "output": "Goodbye! It was nice chatting with you. Have a great day and feel free to come back anytime!",
            "category": "farewell"
        },
        {
            "input": "I'm feeling sad",
            "output": "I'm sorry to hear you're feeling sad. It's okay to have difficult emotions. Would you like to talk about what's bothering you, or would you prefer some suggestions for feeling better?",
            "category": "emotional_support"
        }
    ]
    
    return sample_conversations

def main():
    parser = argparse.ArgumentParser(description='Prepare training data for chatbot')
    parser.add_argument('--input_files', nargs='+', help='Input conversation files')
    parser.add_argument('--input_formats', nargs='+', default=['json'], help='Input file formats')
    parser.add_argument('--output_file', type=str, default='training_data.json', help='Output training data file')
    parser.add_argument('--create_sample', action='store_true', help='Create sample training data')
    parser.add_argument('--augment', action='store_true', help='Apply data augmentation')
    
    args = parser.parse_args()
    
    processor = DataProcessor()
    
    # Create sample data if requested
    if args.create_sample:
        sample_data = create_sample_data()
        processor.add_manual_conversations(sample_data)
        logger.info("Added sample training data")
    
    # Process input files
    if args.input_files:
        formats = args.input_formats * len(args.input_files) if len(args.input_formats) == 1 else args.input_formats
        
        for file_path, format_type in zip(args.input_files, formats):
            try:
                processor.process_conversation_file(file_path, format_type)
            except Exception as e:
                logger.error(f"Error processing {file_path}: {e}")
    
    # Apply data augmentation
    if args.augment:
        processor.augment_data()
    
    # Save processed data
    if processor.conversations:
        processor.save_processed_data(args.output_file)
        
        # Print statistics
        stats = processor.get_statistics()
        logger.info("Data Statistics:")
        for key, value in stats.items():
            logger.info(f"  {key}: {value}")
    else:
        logger.warning("No conversations processed. Use --create_sample to generate sample data.")

if __name__ == "__main__":
    main()
