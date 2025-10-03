import json

from bson import ObjectId
from openai import OpenAI
from job_data_cleaning.embedding.clean_skills import generate_sentences
from utils.spacyhelper import preprocess_text
from utils.mongohelper import MongoHelper


client = OpenAI(api_key="xxxxxxxxxxxxxxxxxxx")
db = MongoHelper("cvbuilder")

jobs = list(db.get_collection("jobs").find({}))
#skill_sentences = list(db.get_collection("skill_sentences").find({}))


for job in jobs:
    try:
        # job["backup_id"] = job['_id']
        # job["job_title_vector"] = client.embeddings.create(input=[preprocess_text(f"Job Opportunity Title: {job['job_title']}")],model="text-embedding-3-small").data[0].embedding
        # job["job_level_vector"] = client.embeddings.create(input=[preprocess_text(f"Job Opportunity Level of Expertise: {job['job_level']}")],model="text-embedding-3-small").data[0].embedding
        # job["skill_needed_with_vector"] = []

        #new_skill_set = []
        for x in job["new_skill_set"]:
            vec_phrase = client.embeddings.create(input=[preprocess_text(x["skill"])],
                                                      model="text-embedding-3-small").data[0].embedding
            #new_skill_set.append({"skill": x, "sentence": sentence , "sentence_embedding_large": vec_large, "sentence_embedding_small":vec_small})
            db.get_collection("jobs").update_one({"_id":job["_id"], "new_skill_set.skill": x["skill"]},{"$set":{"new_skill_set.$.vec_phrase":vec_phrase}})
            #related_sentence = [y for y in skill_sentences if str(x).lower() == str(y["skill"]).lower()]
            #if len(related_sentence) != 0:
                #new_skill_set = new_skill_set + related_sentence
            #else:
                # prompt_template = f'I have a  resume skill, and I need to generate sentence that effectively describe  skill in a way that captures its typical usage or relationship with other relevant technologies or concepts. The goal is to use these sentences for generating embeddings that reflect the practical context and associations of each skill. For  skill, please generate a sentence that includes the skill and contextualizes it within a relevant scenario, industry practice, or common use case. Skill: {x} response type in json: {{"skill":"", "sentence":""}}'
                # response = client.chat.completions.create(
                #     temperature=0.1,
                #     model="gpt-4o-mini",  # or use "gpt-3.5-turbo"
                #     messages=[
                #         {"role": "system", "content": "You are a helpful assistant."},
                #         {"role": "user", "content": prompt_template}
                #     ], response_format={"type": "json_object"}
                # )
                # Extracting the generated sentences from the response
                #sentence = json.loads(response.choices[0].message.content)
                #vec_large = client.embeddings.create(input=[preprocess_text(sentence["sentence"])],model="text-embedding-3-large").data[0].embedding
        #         vec_small = client.embeddings.create(input=[preprocess_text(sentence["sentence"])],
        #                                              model="text-embedding-3-small").data[0].embedding
        #         vec_phrase = client.embeddings.create(input=[preprocess_text(sentence["sentence"])],
        #                                              model="text-embedding-3-small").data[0].embedding
        #         new_skill_set.append({"skill": x, "sentence": sentence , "sentence_embedding_large": vec_large, "sentence_embedding_small":vec_small})
        # db.get_collection("jobs2").update_one({"_id":job["_id"]},{"$set":{"new_skill_set":new_skill_set}})
        # for x in job["skills_needed"]:
        #      if '#' in str(x):
        #          x = str(x).replace('#',' Sharp')
        #      #job["skill_needed_with_vector"].append( {"name": str(x), "vector": client.embeddings.create(input=[preprocess_text(f"Skill Name: {x}")],model="text-embedding-3-small").data[0].embedding})
        #      db.get_collection("jobs2").update_one({"_id": job["_id"],"skill_needed_with_vector.name":x}, {"$set":{"skill_needed_with_vector.$.vector":client.embeddings.create(input=[preprocess_text(f"Skill Name: {str(x)}")],model="text-embedding-3-small").data[0].embedding }})

        # job["education_needed"]["education_level_vector"] = client.embeddings.create(input=[preprocess_text(f"Education Level: {job['education_needed']['education_level']}")],model="text-embedding-3-small").data[0].embedding
        # job["education_needed"]["field_of_study_vector"] = client.embeddings.create(input=[preprocess_text(f"Field Of Study: {job['education_needed']['field_of_study']}")],model="text-embedding-3-small").data[0].embedding
        # job["experience_needed"]["field_of_experience_vector"] = client.embeddings.create(input=[preprocess_text(f"Field Of Experience: {job['experience_needed']['field_of_experience']}")],model="text-embedding-3-small").data[0].embedding
        # if(job['education_needed']['education_level'] is not None and job['education_needed']['field_of_study'] is not None):
        #     combination = f"{job['education_needed']['education_level']} {job['education_needed']['field_of_study']}"
        #     print(combination)
        #     vec = client.embeddings.create(input=[preprocess_text(combination)],model="text-embedding-3-small").data[0].embedding
        #     job['education_needed']['education_field_and_level_vec'] = vec
        # db.get_collection("jobs2").replace_one({"_id": job["_id"]}, job)
    except Exception as Argument:
        print(Argument)
