# Dataset Information

## Overview

This research uses a proprietary dataset from [cvbuilder.me](https://cvbuilder.me) and [jobstart.app](https://jobstart.app), consisting of real-world resume-job applications with compatibility scores assigned by HR experts.

## Dataset Statistics

| Metric | Value |
|--------|-------|
| Total Job Applications | 3,460 |
| Unique Job Postings | 56 |
| Unique Resumes | ~1,355 |
| Job Category | Information Technology (IT) |
| Score Range | 25-95 (multiples of 5) |
| Average Score | 70.98 |
| Language Distribution | 80% Farsi, 20% English |

## Data Structure

The dataset consists of three main MongoDB collections:

### 1. Jobs Collection
Contains job postings with:
- Job title and description
- Required skills with embeddings
- Experience requirements
- Education requirements
- Job level and category

### 2. Resumes Collection
Contains candidate resumes with:
- Personal information
- Skills with embeddings
- Work experience
- Education history
- Projects and certifications

### 3. Relevance Collection
Contains job-resume pairs with:
- Job ID
- Resume ID
- Compatibility score (0-100, assigned by HR experts)
- Application metadata

## Data Privacy

**This dataset is proprietary and NOT publicly available.** 

The data includes:
- Real job postings from companies
- Actual candidate resumes (anonymized)
- HR expert evaluations

Due to privacy concerns and data ownership, the full dataset cannot be shared publicly.

## Sample Data for Research

If you are interested in:
- Testing the model
- Reproducing results
- Academic research
- Further development

**A sample dataset can be provided for legitimate academic/research purposes.**

### Request Sample Data

To request access to a sample dataset:

1. **Email:** neginamirsoleimani@outlook.com
2. **Include in your request:**
   - Your name and affiliation
   - Purpose of the request (research, testing, etc.)
   - Brief description of your intended use
   - Confirmation that data will be used for non-commercial research only

### Sample Dataset Details

The sample dataset includes:
- ~100-200 job-resume pairs
- Representative distribution of scores
- IT sector only
- Anonymized personal information
- English language subset

**Note:** Sample data recipients are expected to:
- Use data only for research/academic purposes
- Not redistribute or publish the data
- Properly cite this work in any publications
- Respect data privacy and confidentiality

## Data Format

Sample data will be provided as:
- **Option 1:** MongoDB dump (BSON format)
- **Option 2:** PyTorch `.pt` file (preprocessed graphs)
- **Option 3:** JSON format (for easier exploration)

## Citations

If you use this work or sample data, please cite:

```bibtex
@mastersthesis{resume_job_compatibility_2024,
  title={Resume-Job Compatibility Prediction using Graph Neural Networks and Large Language Models},
  author={[Your Name]},
  year={2024},
  school={[Your University]},
  note={Dataset from cvbuilder.me and jobstart.app}
}
```

## Data Collection Methodology

The dataset was collected from production platforms:
- **cvbuilder.me**: 2.3M+ resumes created since 2018
- **jobstart.app**: Job recruitment platform with HR evaluation features

HR experts manually evaluated resume-job compatibility on a 0-100 scale based on:
- Skills match
- Experience relevance
- Education alignment
- Overall qualification fit

Scores are restricted to multiples of 5 for consistency.

## Ethical Considerations

- All personal information is anonymized
- No identifiable information is included
- Compliance with data protection regulations
- Consent obtained from platform users
- Used strictly for academic research

## Contact

For questions about the dataset or to request sample data:

**Email:** neginamirsoleimani@outlook.com

**Please note:** Response time may vary. Requests are reviewed on a case-by-case basis.

---

**Dataset License:** Proprietary - Not for public distribution

**Sample Data License:** Available for non-commercial academic research only

