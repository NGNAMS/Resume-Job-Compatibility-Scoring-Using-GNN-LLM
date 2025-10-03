## Migration Guide: Old Code → Organized Structure

This guide helps you migrate from the old unorganized code structure to the new
clean, production-ready codebase.

### 🗂️ New Project Structure

```
resume-job-compatibility/
├── src/                          # Main source code (NEW)
│   ├── config.py                 # Centralized configuration
│   ├── utils/                    # Utilities
│   │   ├── database.py          # Replaces utils/mongohelper.py
│   │   ├── text_processing.py   # Replaces utils/spacyhelper.py
│   │   └── logging.py           # New logging utilities
│   ├── embeddings/               # Embedding generation
│   │   ├── generator.py         # Replaces job_data_cleaning/embedding/main.py
│   │   └── skill_sentence_generator.py
│   ├── graph/                    # Graph construction
│   │   └── builder.py           # Replaces graph_building/model_builder.py
│   └── models/                   # Neural network models
│       ├── gnn_models.py        # Replaces learn_pg/model.py & explainer/model.py
│       ├── trainer.py           # Replaces learn_pg/final.py
│       └── evaluator.py         # Replaces explainer/accuracy.py
│
├── scripts/                      # CLI entry points (NEW)
│   ├── prepare_dataset.py       # Replaces learn_pg/main.py
│   ├── train_model.py           # Replaces learn_pg/final.py
│   ├── evaluate_model.py        # Replaces explainer/accuracy.py
│   └── explain_prediction.py    # Replaces explainer/test.py
│
├── [old directories]             # Keep for reference, will deprecate
│   ├── experminet/
│   ├── explainer/
│   ├── graph_building/
│   ├── job_data_cleaning/
│   ├── learn_pg/
│   ├── resume_data_cleaning/
│   └── utils/
```

### 🔄 Code Migration Examples

#### Example 1: Database Connection

**Old Code (utils/mongohelper.py):**

```python
from utils.mongohelper import MongoHelper

db = MongoHelper("cvbuilder")
jobs = list(db.get_collection("jobs").find({}))
```

**New Code:**

```python
from src.utils.database import DatabaseManager

db = DatabaseManager()  # Loads from config
jobs = db.find_documents("jobs", {})
```

#### Example 2: Text Processing

**Old Code (utils/spacyhelper.py):**

```python
from utils.spacyhelper import preprocess_text

text = preprocess_text("Python is a programming language")
```

**New Code:**

```python
from src.utils.text_processing import TextProcessor

processor = TextProcessor()
text = processor.preprocess("Python is a programming language")
```

#### Example 3: Embedding Generation

**Old Code (job_data_cleaning/embedding/main.py):**

```python
from openai import OpenAI

client = OpenAI(api_key="xxxxxxxxxxxx")  
vec = client.embeddings.create(input=["Python"], model="text-embedding-3-small")
```

**New Code:**

```python
from src.embeddings.generator import EmbeddingGenerator

generator = EmbeddingGenerator(model="text-embedding-3-small")  # Loads from config
embedding = generator.generate("Python")
```

#### Example 4: Graph Building

**Old Code (graph_building/model_builder.py):**

```python
from graph_building.model_builder import model_builder

model = model_builder(job_id="123", resume_id="456")
```

**New Code:**

```python
from src.graph.builder import GraphBuilder

builder = GraphBuilder()
data = builder.build_graph(job_id="123", resume_id="456")
```

#### Example 5: Model Training

**Old Code (learn_pg/final.py):**

```python
# Scattered code with print statements
for epoch in range(num_epochs):
    # Training loop
    print(f"Epoch {epoch}")
```

**New Code:**

```python
from src.models.gnn_models import ResumeJobGNN
from src.models.trainer import ModelTrainer

model = ResumeJobGNN(embedding_dim=3072)
trainer = ModelTrainer(model, learning_rate=0.0001)
history = trainer.train(dataset, epochs=120)
```

### 🔐 Security Migration

**⚠️ CRITICAL: Remove hardcoded API keys!**

#### Step 1: Create `.env` file

```bash
cp .env.example .env
nano .env  # Add your actual credentials
```

#### Step 2: Update Code

**Before:**

```python
client = OpenAI(api_key="xxxxxxxxxxxxx")
```

**After:**

```python
# Configuration automatically loads from environment
from src.config import Config, set_config

config = Config.from_env()
set_config(config)

# All modules now use secure config
from src.embeddings.generator import EmbeddingGenerator
generator = EmbeddingGenerator()  # No API key needed!
```

### 🚀 Using New CLI Scripts

Instead of running individual Python files, use the new CLI scripts:

#### Old Way:

```bash
cd learn_pg
python main.py  # Prepare dataset
python final.py  # Train model
cd ../explainer
python accuracy.py  # Evaluate
```

#### New Way:

```bash
# Set environment variables
export OPENAI_API_KEY="your-key"
export MONGODB_URI="your-uri"

# Prepare dataset
python scripts/prepare_dataset.py --output data/dataset.pt

# Train model
python scripts/train_model.py \
    --dataset data/dataset.pt \
    --epochs 120 \
    --output models/best_model.pth

# Evaluate model
python scripts/evaluate_model.py \
    --model models/best_model.pth \
    --dataset data/dataset.pt

# Explain prediction
python scripts/explain_prediction.py \
    --model models/best_model.pth \
    --job-id "66bd154fd741dcb20c94f445" \
    --resume-id "66a28dd1d2c8fab2d03d5cab"
```

### 📝 Logging Changes

**Old:** Print statements scattered everywhere

```python
print("Processing...")
print(f"Error: {e}")
```

**New:** Proper logging

```python
import logging
logger = logging.getLogger(__name__)

logger.info("Processing...")
logger.error(f"Error: {e}", exc_info=True)
```

### ✅ Migration Checklist

- [ ] Install new dependencies: `pip install -r requirements.txt`
- [ ] Create `.env` file from `.env.example`
- [ ] Remove hardcoded API keys from old code
- [ ] Test new CLI scripts
- [ ] Update any custom scripts to use `src/` modules
- [ ] Run tests to ensure everything works
- [ ] Archive or delete old code directories

### 🔧 Troubleshooting

**Issue: `ModuleNotFoundError: No module named 'src'`**

```bash
# Solution: Add project root to PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
# Or install in development mode:
pip install -e .
```

**Issue: `ValueError: OpenAI API key not provided`**

```bash
# Solution: Set environment variable
export OPENAI_API_KEY="your-key-here"
# Or create .env file
```

**Issue: Old imports still used**

```python
# Replace this:
from utils.mongohelper import MongoHelper

# With this:
from src.utils.database import DatabaseManager
```

### 📚 Additional Resources

- See `README.md` for full documentation
- See `SECURITY.md` for security best practices
- See `QUICKSTART.md` for usage examples

### 🗑️ Deprecation Plan

Once migration is complete:

1. Test all functionality with new code
2. Keep old directories as `_old_*/` for reference
3. After confirming everything works, delete old code
4. Update `.gitignore` to exclude old directories
