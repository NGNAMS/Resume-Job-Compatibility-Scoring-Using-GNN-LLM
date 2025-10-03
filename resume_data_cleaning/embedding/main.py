from bson import ObjectId
from openai import OpenAI

from utils.spacyhelper import preprocess_text
from utils.mongohelper import MongoHelper


client = OpenAI(api_key="xxxxxxxxxxxxxxxxxx")
db = MongoHelper("cvbuilder")

resumes = list(db.get_collection("resumesv3").find({"_id":ObjectId('66a20247d2c8fab2d03c9c33')},{"SkillInfos":1}))

for res in resumes:
    try:

        for x in res["SkillInfos"]:
            if '#' in str(x['SkillName']):
                  x['SkillName'] = str(x['SkillName']).replace('#', ' Sharp')
            db.get_collection("resumesv3").update_one({"_id": res["_id"], "SkillInfos.SkillName": x["SkillName"]}, {"$set": {"SkillInfos.$.SkillNameVector":client.embeddings.create(input=[preprocess_text(f"Skill Name: {str(x['SkillName'])}")],model="text-embedding-3-small").data[0].embedding}})

        #for x in res["EducationInfos"]:
            #x["EducationLevelVector"] = client.embeddings.create(input=[preprocess_text(f"Education Level: {x['EducationLevelTitle']}")],model="text-embedding-3-small").data[0].embedding
            #x["FieldOfStudyVector"] = client.embeddings.create(input=[preprocess_text(f"Field Of Study: {x['FieldOfStudy']}")],model="text-embedding-3-small").data[0].embedding
            #x["EducationInstitudeTitleVector"] = client.embeddings.create(input=[preprocess_text(f"University Name: {x['EducationInstitudeTitle']}")],model="text-embedding-3-small").data[0].embedding
            #combination = preprocess_text(f"{x['EducationLevelTitle']} of {x['FieldOfStudy']}")
            #vec =client.embeddings.create(input=[preprocess_text(f"Education Level And Field Of Study is {x['EducationLevelTitle']} of {x['FieldOfStudy']}")],model="text-embedding-3-small").data[0].embedding
            #x["field_of_study_and_level_of_study"] = vec
            #x["field_of_study_and_level_of_study_title"] = f"{x['EducationLevelTitle']} of {x['FieldOfStudy']}"

            #db.get_collection("resumesv3").update_one({"_id": res["_id"],"EducationInfos.FieldOfStudy": x["FieldOfStudy"],"EducationInfos.EducationLevelTitle": x["EducationLevelTitle"]},{"$set": {"EducationInfos.$.field_of_study_and_level_of_study":vec}})

         # for x in res["ExperienceInfos"]:
        #     x["JobTitleVector"] = client.embeddings.create(input=[preprocess_text(f"Job Title: {x['JobTitle']}")],
        #                                                    model="text-embedding-3-small").data[0].embedding
        #     x["ResponsibilitiesVectors"] = [client.embeddings.create(input=[preprocess_text(f"Job Task: {x}")],
        #                                                              model="text-embedding-3-small").data[0].embedding
        #                                     for x in x["Responsibilities"]]
        #db.get_collection("resumesv3").update_one({"_id": res["_id"]},{"$set":{"EducationInfos":res["EducationInfos"]}})
    except:
        continue