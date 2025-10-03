# Resume-Job Compatibility Model Based on GNN and LLM

A Graph Neural Network (GNN) and Large Language Model (LLM) based system for predicting resume-job compatibility scores, developed as part of a Master's thesis research project.

## 📋 Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Dataset](#dataset)
- [Methodology](#methodology)
- [Architecture](#architecture)
- [Results](#results)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Requirements](#requirements)
- [Citation](#citation)

## 🎯 Overview

This research proposes an automated resume screening system that combines Graph Neural Networks (GNNs) and Large Language Models (LLMs) to predict compatibility scores between resumes and job descriptions. The model achieves an **R² score of 0.803** on a dataset of 3,460 job applications in the Information Technology sector, outperforming traditional methods like TF-IDF with cosine similarity (R² = 0.2363) and LLM-based recruiter agents (R² = 0.6519).

### Problem Statement

Traditional resume screening systems face several limitations:
- **Keyword matching bias**: Missing semantic relationships between skills
- **Lack of interpretability**: Black-box decision making
- **Inconsistent evaluation**: Human bias and variability in scoring
- **Scalability issues**: Manual screening is time-consuming

This research addresses these challenges through a graph-based approach that captures semantic relationships and provides explainable predictions.

## ✨ Key Features

- **Semantic Understanding**: Uses OpenAI's `text-embedding-3-large` (3072-dimensional embeddings) to capture contextual meaning
- **Graph-Based Modeling**: Represents resumes, jobs, and skills as heterogeneous graphs
- **Explainable AI**: GAT attention scores reveal which skills contribute most to compatibility predictions
- **Proficiency Awareness**: Incorporates skill proficiency levels (1-10 scale) as edge weights
- **High Accuracy**: Achieves MAE of 0.045 on a 0-1 normalized score scale

## 📊 Dataset

**Note on Data Access:** The dataset used in this research is proprietary and private. However, if you are interested in testing the model or conducting related research, a sample dataset can be provided for academic purposes. Please contact: **neginamirsoleimani@outlook.com**

### Statistics
| Metric | Value |
|--------|-------|
| Total Job Applications | 3,460 |
| Unique Job Postings | 56 |
| Unique Resumes | ~1,355 |
| Job Category | Information Technology (IT) |
| Score Range | 25-95 (multiples of 5) |
| Average Score | 70.98 |
| Resumes per Job (avg) | 61.79 |
| Language Distribution | 80% Farsi, 20% English |

### Score Distribution
Compatibility scores were assigned by HR experts on a scale of 0-100, with scores restricted to multiples of 5. The dataset captures real-world recruiting dynamics where multiple resumes are submitted for each job posting.

## 🔬 Methodology

The workflow consists of six main steps:

### 1. Dataset Collection
- Resumes and job descriptions collected from private sources (contact for data sample)
- HR experts assigned compatibility scores (0-100) for each resume-job pair

### 2. Information Extraction

#### Named Entity Recognition (NER)
- Fine-tuned **SpaCy** NER model for skill extraction
- Generated contextual sentences using **ChatGPT-4o-mini** (3 sentences per skill)
- Example: "Python" → "The ideal candidate should have a solid understanding of Python and experience with its use in data analysis."

#### Part-of-Speech (POS) Tagging
- Extracted proficiency levels from adjectives (e.g., "basic", "strong", "expert")
- Mapped to numerical scale (1-10) for edge weights
- Example: "basic knowledge of Python" → Python (skill), proficiency = 2

### 3. Embedding Generation

Two OpenAI embedding models were evaluated:

| Model | Dimensions | MTEB Overall Rank | Best Use Case |
|-------|-----------|-------------------|---------------|
| `text-embedding-3-small` | 1536 | 54/406 | Faster processing |
| `text-embedding-3-large` | 3072 | 30/406 | Richer context ✅ |

**Preprocessing for Ambiguous Skills:**
- Single-character terms (e.g., "R") enriched with definitions
- Example: "R" → "R, a programming language used for statistical analysis and data science"

### 4. Graph Construction

**Heterogeneous Graph Structure:**
- **Nodes**: Resume (1), Job (1), Skills (variable)
- **Edges**: 
  - Job → Skills (required skills)
  - Resume → Skills (candidate skills)
  - Skill → Skill (common skills bridge resume and job)

**Common Skills Identification:**
- Cosine similarity threshold: **0.65**
- Skills exceeding threshold are shared between resume and job
- Other skills connected individually to resume or job nodes

### 5. GNN + MLP Training

#### Graph Neural Network
- **GraphConv** for job-skill edges (computational efficiency)
- **GATConv** for resume-skill edges (8 attention heads for explainability)
- Edge weights incorporate proficiency levels
- Output: Updated embeddings for job and resume nodes

#### Multi-Layer Perceptron (MLP)
```
Input: Concatenated [Job Embedding | Resume Embedding] (6144 dims)
Hidden Layers: 4096 → 2048 → 1024 → 512 → 256 → 1
Activation: LeakyReLU
Dropout: 0.2
Output: Compatibility Score (0-100)
```

**Training Configuration:**
- Optimizer: AdamW
- Learning Rate: 0.0001
- Batch Size: 32
- Epochs: 120
- Loss Function: Mean Squared Error (MSE)

### 6. Explainable AI (XAI)

**Attention Mechanism:**
- GAT layer computes attention scores for each resume-skill edge
- Scores range 0-1 (normalized via softmax)
- Identifies which skills contribute most to compatibility prediction

**Example Attention Scores:**
- TypeScript: 0.90 (high impact)
- VueJS: 0.85 (high impact)
- JavaScript: 0.60 (moderate impact)
- Git: 0.05 (low impact)

## 🏗️ Architecture

### Model Pipeline

```
┌─────────────────┐
│  Resume + Job   │
│  (Raw Text)     │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ NER + POS       │
│ (SpaCy)         │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ LLM Embeddings  │
│ (OpenAI)        │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Graph Builder   │
│ (Heterogeneous) │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ GNN Layer       │
│ (GAT + GConv)   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ MLP Regressor   │
│ (6144→1)        │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Compatibility   │
│ Score (0-100)   │
└─────────────────┘
```

### GNN Architecture Details

**Attention Score Calculation (GAT):**

```math
α_ij = exp(LeakyReLU(a^T [W h_i || W h_j || w_ij])) / Σ_k exp(LeakyReLU(a^T [W h_i || W h_k || w_ik]))
```

Where:
- `h_i`, `h_j`: Node embeddings
- `w_ij`: Edge weight (proficiency level)
- `W`, `a`: Learnable parameters
- `||`: Concatenation

**Node Update:**

```math
h'_i = σ(Σ_j α_ij · W h_j)
```

## 📈 Results

### Experiment Comparison

Four experiments were conducted to identify the optimal configuration:

| Experiment | Model | Input Type | R² | MAE | MSE |
|-----------|-------|------------|-----|-----|-----|
| 1 | text-embedding-3-small | Exact Phrase | 0.771 | 0.053 | 0.0031 |
| 2 | text-embedding-3-small | Phrase Definition | 0.787 | 0.050 | 0.0027 |
| 3 | text-embedding-3-large | Exact Phrase | 0.781 | 0.052 | 0.0031 |
| **4** | **text-embedding-3-large** | **Phrase Definition** | **0.803** | **0.045** | **0.0020** |

**Key Findings:**
- Phrase definitions consistently improved performance over exact phrases
- Larger embeddings (3072-dim) outperformed smaller ones (1536-dim)
- Best configuration: `text-embedding-3-large` + phrase definitions

### Comparison with State-of-the-Art

| Method | R² | MAE | MSE |
|--------|-----|-----|-----|
| **Proposed Method (This Research)** | **0.803** | **0.045** | **0.0020** |
| LLM-Based Recruiter Agent (Gan et al., 2024) | 0.652 | 0.092 | 0.0085 |
| TF-IDF + Cosine Similarity (Daryani et al., 2020) | 0.236 | 0.132 | 0.0237 |

**Performance Improvements:**
- **23% higher R²** compared to LLM recruiter agent
- **51% lower MAE** compared to LLM recruiter agent
- **240% higher R²** compared to TF-IDF method

### Evaluation Metrics

**R² (Coefficient of Determination)**: 0.803
- Explains 80.3% of variance in HR-assigned scores

**MAE (Mean Absolute Error)**: 0.045 (normalized scale)
- On 0-100 scale: ~4.5 points average error

**MSE (Mean Squared Error)**: 0.0020
- Low variance in prediction errors

## 📧 Contact & Data Access

**Contact:** neginamirsoleimani@outlook.com

**Dataset:** The dataset used in this research is proprietary and private. A sample dataset can be provided for academic research and testing purposes upon request. See [DATASET_INFO.md](DATASET_INFO.md) for details.

## 📁 Project Structure

```
resume-job-compatibility/
│
├── experminet/
│   └── embedding.py                 # Embedding generation experiments
│
├── explainer/
│   ├── accuracy.py                  # Model accuracy evaluation
│   ├── model.py                     # GAT model for explainability
│   └── test.py                      # XAI testing with attention scores
│
├── graph_building/
│   ├── main.py                      # Graph visualization demo
│   └── model_builder.py             # Resume-job graph construction
│
├── job_data_cleaning/
│   ├── category/
│   │   └── assign_category.py       # Job categorization
│   ├── embedding/
│   │   ├── clean_skills.py          # Skill sentence generation (ChatGPT)
│   │   ├── main.py                  # Job embedding generation
│   │   └── skills.py                # Skill processing utilities
│   └── skill_cleaning/
│       └── test.py                  # Skill cleaning tests
│
├── learn_pg/
│   ├── final.py                     # Final training script
│   ├── main.py                      # Dataset preparation
│   ├── model.py                     # SimpleGCN model definition
│   ├── hetero_conv_dblp.py         # Heterogeneous GNN examples
│   └── [other learning experiments]
│
├── resume_data_cleaning/
│   ├── embedding/
│   │   ├── clean.py                 # Resume data cleaning
│   │   └── main.py                  # Resume embedding generation
│   └── title/
│       └── matching_resume_title.py # Resume title matching
│
└── utils/
    ├── mongohelper.py               # MongoDB connection utilities
    ├── spacyhelper.py               # Text preprocessing (SpaCy)
    ├── scatter_visualize.py         # Result visualization
    └── visualize_graph.py           # Graph visualization

Key Files:
- learn_pg/main.py        → Dataset preparation (creates .pt files)
- learn_pg/model.py       → GNN architecture (SimpleGCN)
- learn_pg/final.py       → Model training loop
- explainer/model.py      → GAT model with attention mechanism
- explainer/accuracy.py   → Evaluation with ±5 point tolerance
- graph_building/model_builder.py → Graph construction logic
```

## 🚀 Installation

### Prerequisites

- Python 3.8+
- MongoDB (for data storage)
- CUDA-capable GPU (recommended for training)

### Setup

1. **Clone the repository:**
```bash
git clone https://github.com/yourusername/resume-job-compatibility.git
cd resume-job-compatibility
```

2. **Create virtual environment:**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies:**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install torch-geometric
pip install pyg-lib torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.0.0+cu118.html
pip install spacy openai pymongo scikit-learn matplotlib networkx fuzzywuzzy pandas tqdm
python -m spacy download en_core_web_sm
```

### Configuration

**Create `.env` file with your credentials:**

```bash
cp .env.example .env
nano .env  # Edit with your actual credentials
```

Add your credentials:
```bash
OPENAI_API_KEY=xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
MONGODB_URI=mongodb+srv://username:password@cluster.mongodb.net/
```

⚠️ **Important**: Never commit the `.env` file to version control. It's already in `.gitignore`.

## 💻 Usage

### 1. Data Preparation

```bash
# Generate embeddings for job skills
cd job_data_cleaning/embedding
python main.py

# Generate embeddings for resume skills
cd ../../resume_data_cleaning/embedding
python main.py
```

### 2. Build Graph Dataset

```bash
cd ../../learn_pg
python main.py
```

This creates: `small_skill_with_title_dataset_six_point_five.pt`

### 3. Train Model

```bash
python final.py
```

**Training Configuration:**
- Batch Size: 32
- Learning Rate: 0.0001
- Epochs: 120 (for large embeddings) or 180 (for small embeddings)
- Train/Test Split: 80/20

### 4. Evaluate Model

```bash
cd ../explainer
python accuracy.py
```

**Accuracy Metric:** Percentage of predictions within ±5 points of HR score

### 5. Visualize Explanations

```bash
python test.py
```

Generates:
- Graph visualization with attention scores
- Bar chart showing skill importance

### Example: Build and Visualize a Single Graph

```bash
cd graph_building
python main.py
```

Edit `main.py` to specify job and resume IDs:
```python
model = model_builder(
    job_id='66bd154fd741dcb20c94f445',
    resume_id='66a28dd1d2c8fab2d03d5cab'
)
```

## 📦 Requirements

### Core Dependencies

```
torch>=2.0.0
torch-geometric>=2.3.0
spacy>=3.5.0
openai>=1.0.0
pymongo>=4.3.0
scikit-learn>=1.2.0
networkx>=3.0
pandas>=1.5.0
matplotlib>=3.7.0
numpy>=1.24.0
```

### Optional Dependencies

```
fuzzywuzzy>=0.18.0
python-Levenshtein>=0.20.0
tqdm>=4.65.0
```

## 🔧 Model Hyperparameters

### Best Configuration (Experiment 4)

```python
# Embedding
embedding_model = "text-embedding-3-large"
embedding_dim = 3072
use_definitions = True  # Enrich phrases with ChatGPT definitions

# Graph Construction
cosine_similarity_threshold = 0.65
include_proficiency_weights = True

# GNN
gnn_job_layer = "GraphConv"
gnn_resume_layer = "GATConv"
gat_heads = 8
gat_concat = False  # Average heads
activation = "LeakyReLU"

# MLP
hidden_layers = [4096, 2048, 1024, 512, 256]
dropout = 0.2

# Training
batch_size = 32
learning_rate = 0.0001
optimizer = "AdamW"
epochs = 120
loss_function = "MSE"
```

## 🎓 Research Contributions

1. **Novel Graph Representation**: Heterogeneous graph combining resume, job, and skill nodes with proficiency-weighted edges

2. **Hybrid GNN Architecture**: Strategic combination of GraphConv (efficiency) and GAT (explainability)

3. **Contextual Embeddings**: Use of phrase definitions to improve semantic understanding of ambiguous terms

4. **Explainability**: Attention mechanism provides transparency in skill importance

5. **Real-world Dataset**: 3,460 HR-annotated job applications from production platforms

## 📝 Proficiency Level Mapping

| Descriptor | Level | Example |
|-----------|-------|---------|
| Basic, Fundamental | 1-2 | "Basic knowledge of Python" |
| Beginner, Developing | 3 | "Developing skills in JavaScript" |
| Intermediate, Capable | 4 | "Capable of managing database queries" |
| Proficient, Competent | 5 | "Proficient in React development" |
| Skilled, Reliable | 6 | "Skilled in machine learning" |
| Advanced, Accomplished | 7 | "Advanced expertise in distributed systems" |
| Experienced, Resourceful | 8 | "Experienced with large-scale deployments" |
| Expert, Specialist | 9 | "Expert in cloud architecture" |
| Strong, Master | 10 | "Strong leadership in technical projects" |

## 🔍 Limitations

1. **Dataset Scope**: Limited to IT sector and English/Farsi languages
2. **API Dependency**: Requires OpenAI API access for embeddings
3. **Computational Cost**: Large embeddings (3072-dim) require significant GPU memory
4. **Score Granularity**: Dataset scores are multiples of 5, limiting fine-grained prediction
5. **Proficiency Detection**: Relies on explicit adjectives; implicit proficiency may be missed

## 🚀 Future Work

- [ ] Extend to multiple job categories beyond IT
- [ ] Incorporate work experience and education sections into graph
- [ ] Test with open-source embedding models (e.g., Sentence-BERT)
- [ ] Implement multi-task learning for scoring and ranking
- [ ] Add temporal dynamics (experience duration, recency)
- [ ] Develop API for real-time resume screening

## 📚 References

Key papers that influenced this research:

1. **Gan et al. (2024)**: "LLM-based Recruiter Agent" - LLM-powered HR decision making
2. **Daryani et al. (2020)**: "Automated Resume Screening" - TF-IDF baseline approach
3. **Muennighoff et al. (2022)**: "MTEB: Massive Text Embedding Benchmark" - Embedding evaluation
4. **SpaCy Documentation**: Named Entity Recognition and POS tagging

## 🤝 Contributing

This is a research project developed for a Master's thesis. If you'd like to build upon this work:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/improvement`)
3. Commit your changes (`git commit -am 'Add new feature'`)
4. Push to the branch (`git push origin feature/improvement`)
5. Open a Pull Request

## 📄 License

This project is part of academic research. Please contact the author for licensing information.

## 👤 Author

- Dataset: 3,460 IT job applications with HR-assigned scores (private)
- Contact: **neginamirsoleimani@outlook.com**

## 🙏 Acknowledgments

- HR experts who annotated compatibility scores
- OpenAI for embedding models and ChatGPT API
- PyTorch Geometric team for graph learning tools
- SpaCy team for NLP utilities

---

**Note**: This README describes a research prototype. API keys visible in the code should be removed before deploying to production. Use environment variables and secure credential management in real-world applications.

**Hardware Used**: Google Colab with NVIDIA Tesla T4 GPU (~16 GB memory)

**Dataset Access**: The dataset is proprietary and private. For academic research or testing purposes, a sample dataset may be available upon request. Contact: **neginamirsoleimani@outlook.com**

