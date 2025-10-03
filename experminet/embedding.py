import numpy as np
from openai import OpenAI
from sklearn.metrics.pairwise import cosine_similarity
from utils.mongohelper import MongoHelper
from utils.spacyhelper import preprocess_text

client = OpenAI(api_key="xxxxxxxxxxxxxxxxxx")

first_skill = MongoHelper("cvbuilder").get_collection("skill_sentences").find_one({"skill":"Java SE"})

second_skill = MongoHelper("cvbuilder").get_collection("skill_sentences").find_one({"skill":"JavaScript"})


v1 = client.embeddings.create(input=[first_skill["sentence"]],model="text-embedding-3-large").data[0].embedding

v2 = client.embeddings.create(input=[second_skill["sentence"]],model="text-embedding-3-large").data[0].embedding

similarity = cosine_similarity(np.array([v1]), np.array([v2]))

print(similarity)