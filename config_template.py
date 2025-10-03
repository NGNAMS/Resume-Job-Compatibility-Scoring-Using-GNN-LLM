"""
Configuration Template for Resume-Job Compatibility Model

IMPORTANT: 
1. Copy this file to 'config.py' 
2. Fill in your actual API keys and credentials
3. Never commit config.py to version control (it's in .gitignore)

Usage:
    from config import OPENAI_API_KEY, MONGODB_URI
"""

# OpenAI API Configuration
OPENAI_API_KEY = "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
OPENAI_MODEL_EMBEDDING_SMALL = "text-embedding-3-small"
OPENAI_MODEL_EMBEDDING_LARGE = "text-embedding-3-large"
OPENAI_MODEL_CHAT = "gpt-4o-mini"

# MongoDB Configuration
MONGODB_URI = "mongodb+srv://username:password@cluster.mongodb.net/database?retryWrites=true&w=majority"
MONGODB_DATABASE = "cvbuilder"

# Collections
COLLECTION_JOBS = "jobs"
COLLECTION_RESUMES = "resumes"
COLLECTION_RELEVANCE = "relevance"
COLLECTION_SKILL_SENTENCES = "skill_sentences"

# Model Hyperparameters
EMBEDDING_DIMENSION_SMALL = 1536
EMBEDDING_DIMENSION_LARGE = 3072

# Graph Construction
COSINE_SIMILARITY_THRESHOLD = 0.65

# Training Configuration
BATCH_SIZE = 32
LEARNING_RATE = 0.0001
EPOCHS = 120
DROPOUT_RATE = 0.2

# Device
DEVICE = "cuda"  # or "cpu"

# Paths
MODEL_SAVE_PATH = "model2.pth"
DATASET_PATH = "small_skill_with_title_dataset_six_point_five.pt"

