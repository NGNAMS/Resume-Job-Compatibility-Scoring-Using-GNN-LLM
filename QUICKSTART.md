# Quick Start Guide

This guide will help you get started with the Resume-Job Compatibility model in under 10 minutes.

## Prerequisites

- Python 3.8 or higher
- CUDA-capable GPU (recommended) or CPU
- MongoDB instance
- OpenAI API key

## Step-by-Step Setup

### 1. Clone and Install (5 minutes)

```bash
# Clone the repository
git clone https://github.com/yourusername/resume-job-compatibility.git
cd resume-job-compatibility

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install PyTorch (CUDA 11.8 example)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install PyTorch Geometric
pip install torch-geometric
pip install pyg-lib torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.0.0+cu118.html

# Install other dependencies
pip install -r requirements.txt

# Download SpaCy model
python -m spacy download en_core_web_sm
```

### 2. Configure Credentials (2 minutes)

```bash
# Copy config template
cp config_template.py config.py

# Edit config.py and add your credentials
nano config.py  # or use your preferred editor
```

**Required credentials:**
- OpenAI API key
- MongoDB connection string

### 3. Verify Installation (1 minute)

```python
# test_installation.py
import torch
import torch_geometric
import spacy
from openai import OpenAI
import pymongo

print(f"✓ PyTorch version: {torch.__version__}")
print(f"✓ CUDA available: {torch.cuda.is_available()}")
print(f"✓ PyG version: {torch_geometric.__version__}")
print(f"✓ SpaCy installed: {spacy.__version__}")
print("✓ All dependencies installed successfully!")
```

```bash
python test_installation.py
```

## Usage Examples

### Example 1: Build a Resume-Job Graph

```python
from graph_building.model_builder import model_builder
from utils.visualize_graph import draw_graph

# Build graph for a specific job-resume pair
model = model_builder(
    job_id='your_job_id_here',
    resume_id='your_resume_id_here'
)

# Inspect common skills
print(f"Common skills: {len(model['common_skills'])}")
for skill in model['common_skills']:
    print(f"  - {skill['org_obj']['skill']} (similarity: {skill['similarity']:.2f})")
```

### Example 2: Prepare Training Dataset

```python
# learn_pg/main.py is already set up
# Just run it to generate the dataset

import torch
from torch_geometric.data import HeteroData
from graph_building.model_builder import model_builder
from utils.mongohelper import MongoHelper

db = MongoHelper("cvbuilder")
relevances = list(db.get_collection("relevance").find({}))

print(f"Total job applications: {len(relevances)}")
print("Building graphs... (this may take a while)")

# The script will create: small_skill_with_title_dataset_six_point_five.pt
```

### Example 3: Train the Model

```python
# learn_pg/final.py
import torch
from model import SimpleGCN
from torch_geometric.loader import DataLoader
from sklearn.model_selection import train_test_split

# Load dataset
data_set = torch.load('small_skill_dataset_six_point_five.pt')
train_dataset, test_dataset = train_test_split(data_set, test_size=0.2, random_state=42)

# Initialize model
model = SimpleGCN(dim_in=1536, dim_h=1024)

# Training loop
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
criterion = torch.nn.MSELoss()

for epoch in range(num_epochs):
    model.train()
    for data in train_loader:
        outputs = model(data)
        targets = data.y / 100  # Normalize to 0-1
        loss = criterion(outputs, targets)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

### Example 4: Evaluate Model

```python
# explainer/accuracy.py
from sklearn.metrics import accuracy_score

model.eval()
y_pred = []
y_true = []

for x in test_data:
    score, attention_scores = model(x)
    predicted_score = score.item()
    real_score = x.y.item()
    
    y_true.append(True)
    # Check if within ±5 points
    if abs(predicted_score - real_score) <= 5:
        y_pred.append(True)
    else:
        y_pred.append(False)

accuracy = accuracy_score(y_true, y_pred)
print(f"Accuracy (±5 points): {accuracy:.2%}")
```

### Example 5: Get Explainability (Attention Scores)

```python
# explainer/test.py
import torch
import matplotlib.pyplot as plt

# Load trained model
model = torch.load('explainer/model2.pth', map_location='cpu')
model.eval()

# Get prediction with attention scores
score, attention_scores = model(data)

# Extract attention weights for resume-skill edges
attention_weights = attention_scores['skill_to_resume'][1]
skill_titles = data['skill'].title

# Visualize
plt.figure(figsize=(10, 6))
plt.barh(skill_titles, attention_weights.max(dim=1).values.numpy())
plt.xlabel('Attention Score')
plt.title('Skill Importance for Resume-Job Match')
plt.show()
```

## Pipeline Overview

```
1. Data Preparation
   ├── Clean job data        → job_data_cleaning/
   ├── Clean resume data     → resume_data_cleaning/
   └── Generate embeddings   → Use OpenAI API
   
2. Graph Construction
   └── Build graphs          → learn_pg/main.py
   
3. Model Training
   ├── Train GNN+MLP         → learn_pg/final.py
   └── Save model            → model2.pth
   
4. Evaluation
   ├── Test accuracy         → explainer/accuracy.py
   └── Visualize attention   → explainer/test.py
```

## Common Issues and Solutions

### Issue 1: CUDA Out of Memory
**Solution:** Reduce batch size or use CPU
```python
# In training script
batch_size = 16  # Instead of 32
device = 'cpu'   # If GPU memory insufficient
```

### Issue 2: MongoDB Connection Failed
**Solution:** Check connection string and network
```python
# Test connection
from utils.mongohelper import MongoHelper
db = MongoHelper("cvbuilder")
print(db.get_collection("jobs").count_documents({}))
```

### Issue 3: OpenAI API Rate Limit
**Solution:** Add delay between requests
```python
import time
for skill in skills:
    embedding = client.embeddings.create(...)
    time.sleep(0.1)  # 100ms delay
```

### Issue 4: SpaCy Model Not Found
**Solution:** Download the model
```bash
python -m spacy download en_core_web_sm
```

## Next Steps

1. **Explore the data**: Use MongoDB Compass or `pymongo` to inspect collections
2. **Customize graph**: Modify `model_builder.py` to include education/experience
3. **Tune hyperparameters**: Experiment with learning rate, batch size, epochs
4. **Visualize results**: Use `utils/scatter_visualize.py` for predictions
5. **Deploy model**: Create REST API using Flask or FastAPI

## Dataset Structure

### MongoDB Collections

**jobs** collection:
```json
{
  "_id": ObjectId("..."),
  "job_title": "Software Engineer",
  "new_skill_set": [
    {
      "skill": "Python",
      "sentence_embedding_small": [0.12, 0.34, ...],
      "vec_phrase": [...]
    }
  ]
}
```

**resumes** collection:
```json
{
  "_id": ObjectId("..."),
  "BasicInfo": {
    "ResumeTitle": "Senior Developer"
  },
  "new_skill_set": [
    {
      "org_skill_title": "Python",
      "related_skill_sentence_embedding_small": [...]
    }
  ]
}
```

**relevance** collection:
```json
{
  "_id": ObjectId("..."),
  "job_id": "66bd154fd741dcb20c94f445",
  "resume_id": "66a28dd1d2c8fab2d03d5cab",
  "score": 75
}
```

## Performance Benchmarks

On Google Colab (Tesla T4 GPU):
- **Graph construction**: ~5 seconds per job-resume pair
- **Training (120 epochs)**: ~30-45 minutes for 3,460 samples
- **Inference**: ~0.01 seconds per prediction

## Resources

- **Main README**: `README.md`
- **Security Guide**: `SECURITY.md`
- **Requirements**: `requirements.txt`
- **Configuration**: `config_template.py`

## Support

For questions or issues:
1. Check the main README.md
2. Review code comments in source files
3. Refer to the original thesis document

---

**Happy coding! 🚀**

