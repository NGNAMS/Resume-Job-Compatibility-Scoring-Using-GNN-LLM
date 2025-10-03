import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from utils.mongohelper import MongoHelper
from openai import OpenAI


db = MongoHelper("cvbuilder")
client = OpenAI(api_key="xxxxxxxxxxxxxxxxxx")
jobs = db.get_collection("jobs").find({"category":{"$eq":None}})

# category_dict = [
#     {"title": "Electrical and Electronic Engineering" ,"vector": client.embeddings.create(input=["Electrical and Electronic Engineering"],model="text-embedding-3-small").data[0].embedding},
#     {"title": "Civil Engineering / Architecture" ,"vector": client.embeddings.create(input=["Civil Engineering / Architecture"], model="text-embedding-3-small").data[0].embedding},
#     {"title": "Software Development and Programming", "vector": client.embeddings.create(input=["Software Development and Programming"], model="text-embedding-3-small").data[0].embedding},
#     {"title": "Mechanical / Aerospace Engineering", "vector":client.embeddings.create(input=["Mechanical / Aerospace Engineering"], model="text-embedding-3-small").data[0].embedding},
#
# ]

for job in jobs:

    # title_embedding = client.embeddings.create(input=[job['job_title']], model="text-embedding-3-small").data[0].embedding
    # cosine_similarities = []
    # for y in category_dict:
    #     similarity = cosine_similarity(np.array([title_embedding]), np.array([y["vector"]]))
    #     y["similarity"] = similarity[0][0]
    #     cosine_similarities.append(y)
    # df = pd.DataFrame(sorted(cosine_similarities, key=lambda x: x['similarity'], reverse=True))
    # high_candidate = df.iloc[0]
    print(job["job_title"])
    db.get_collection("jobs").delete_one({})
    #db.get_collection("jobs").update_one({"_id": job["_id"]}, {"$set": {"category": {"title": "Electrical and Electronic Engineering"}}})