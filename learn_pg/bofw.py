import pandas as pd
import torch
from sklearn.feature_extraction.text import CountVectorizer
from utils.mongohelper import MongoHelper
from utils.spacyhelper import preprocess_text


# def bogrelevance():
#     db = MongoHelper('cvbuilder')

#     jobs = pd.read_csv('jobs_bow.csv')
#     resumes = pd.read_csv('resumes_bow.csv')

#     relevances = list(db.get_collection("relevance3").find({}))

#     result = []
#     for x in relevances:
#         job = jobs.loc[jobs['_id'] == x["job_id"], jobs.columns[1:]].values.flatten().tolist()
#         job_tensor = torch.tensor(job, dtype=torch.float)
#         resume = resumes.loc[resumes['_id'] == x["resume_id"], resumes.columns[1:]].values.flatten().tolist()
#         resume_tensor = torch.tensor(resume, dtype=torch.float)
#         combined_embedding = torch.cat((job_tensor.unsqueeze(1), resume_tensor.unsqueeze(1)), dim=1)
#         result.append({"score": x["score"] / 100, "vec": job + resume})

#     return result

db = MongoHelper('cvbuilder')
final = {}
resumes_skills = list(db.get_collection("resumes").find({"JobCategory.EnTitle":"Software Development and Programming"},{"SkillInfos.SkillName":1}))
job_skills = list(db.get_collection("jobs").find({},{"skills_needed":1}))
relevances = list(db.get_collection("relevance2").find({}))

resumes = {}
for x in resumes_skills:
    temp = []
    for y in x['SkillInfos']:
        temp.append(preprocess_text(y["SkillName"]).replace(",",""))
    resumes[str(x['_id'])] = temp

jobs = {}
for x in job_skills:
    temp = []
    for y in x['skills_needed']:
        temp.append(preprocess_text(y).replace(",",""))
    jobs[str(x['_id'])] = temp

resumes = {k: ",".join(v) for k, v in resumes.items()}
jobs = {k: ",".join(v) for k, v in jobs.items()}

documents = {**resumes, **jobs}


# Extract the IDs and the text for vectorization
ids = list(documents.keys())
text_data = list(documents.values())

# Initialize the CountVectorizer
vectorizer = CountVectorizer(lowercase=False)

# Fit and transform the documents to create the BoW vector
bow_matrix = vectorizer.fit_transform(text_data)

# Convert the BoW matrix to an array for easier viewing
bow_array = bow_matrix.toarray()

# Get the feature names (skills) corresponding to the columns in the BoW array
feature_names = vectorizer.get_feature_names_out()

# Convert the BoW array to a pandas DataFrame and add the IDs
bow_df = pd.DataFrame(bow_array, columns=feature_names)
bow_df.insert(0, '_id', ids)

# Separate resumes and jobs for saving
resumes_df = bow_df.iloc[:len(resumes)].reset_index(drop=True)
jobs_df = bow_df.iloc[len(resumes):].reset_index(drop=True)

# for x in relevances:
#     res = resumes_df.loc[resumes_df['id'] == str(x['resume_id']), resumes_df.columns[2:]].values.flatten().tolist()
#
#     job = jobs_df.loc[jobs_df['id'] == str(x['job_id']), jobs_df.columns[2:]].values.flatten().tolist()
#


# Save the DataFrames as CSV files
resumes_df.to_csv('resumes_bow.csv', index=False)
jobs_df.to_csv('jobs_bow.csv', index=False)

