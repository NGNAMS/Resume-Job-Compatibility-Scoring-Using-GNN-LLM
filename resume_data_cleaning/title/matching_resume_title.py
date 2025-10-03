from utils.mongohelper import MongoHelper
import pandas as pd
from openai import OpenAI
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from colorama import Fore, Style
import time
from utils.spacyhelper import preprocess_text


client = OpenAI(api_key="xxxxxxxxxxxxxxxxxxx")
db = MongoHelper("cvbuilder")

resumes = list(db.get_collection("resumesv2").find({"JobCategory" : {"$eq":None},"BasicInfo.ResumeTitleVector" : {"$ne":None}},{"BasicInfo":1,"ExperienceInfos":1}))

job_categories = pd.read_excel("job-categories.xlsx",header=None)
job_cat_dic = []
for x in job_categories.iloc:
    processed_job_category = preprocess_text(x[1])
    job_category_vector = client.embeddings.create(input=[processed_job_category],model="text-embedding-3-small").data[0].embedding
    job_cat_dic.append({"name": x[1],"processed_name": processed_job_category,"vector": job_category_vector})


for x in resumes:
    try:
        #time.sleep(1)
        cosine_similarities = []
        processed_resume_title = preprocess_text(x["BasicInfo"]["ResumeTitle"])

        # if (processed_resume_title == ""):
        #     continue
        # title_embedding = client.embeddings.create(input=[processed_resume_title], model="text-embedding-3-small").data[
        #     0].embedding
        title_embedding = x["BasicInfo"]["ResumeTitleVector"]
        for y in job_cat_dic:
            similarity = cosine_similarity(np.array([title_embedding]), np.array([y["vector"]]))
            y["target_name"] = processed_resume_title
            y["similarity"] = similarity[0][0]
            cosine_similarities.append(y)
        df = pd.DataFrame(sorted(cosine_similarities, key=lambda x: x['similarity'], reverse=True))
        high_candidate = df.iloc[0]
        print(
            f"{Fore.RED} - {high_candidate['similarity']} - {high_candidate['name']} - {high_candidate['target_name']}  {Style.RESET_ALL}")
        # db.get_collection("resumesv2").update_one({"_id": x["_id"]},
        #                                           {"$set": {"BasicInfo.ResumeTitleVector": title_embedding}})
        if high_candidate["similarity"] >= 0.5:
            print(f"{Fore.GREEN} {high_candidate['similarity']} {Style.RESET_ALL}")
            db.get_collection("resumesv2").update_one({"_id": x["_id"]}, {
                "$set": {"JobCategory": {"EnTitle": high_candidate["name"], "EnTitleVector": y["vector"]}}})

    except:
        continue

