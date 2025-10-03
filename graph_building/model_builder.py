import numpy as np
from utils.mongohelper import MongoHelper
from bson.objectid import ObjectId
from sklearn.metrics.pairwise import cosine_similarity


def model_builder(resume_id='66a28dd1d2c8fab2d03d5cab',job_id='66bd143ad741dcb20c94f444'):
    db = MongoHelper("cvbuilder")

    job = db.get_collection("jobs").find_one({"_id": ObjectId(job_id)})
    resume = db.get_collection("resumes").find_one({"_id": ObjectId(resume_id)})

    common_education = []
    # for x in resume["EducationInfos"]:
    #     model = {"org_obj": job["education_needed"], "similarity": 0}
    #     edu_similarity = cosine_similarity(np.array([job["education_needed"]["education_field_and_level_vec"]]),np.array([x["field_of_study_and_level_of_study"]]))
    #     if model["similarity"] < edu_similarity:
    #         model["similarity"] = edu_similarity
    #         model["tar_obj"] = x
    #     if model["similarity"] > 0.6:
    #         common_education.append(model)

    common_experience = []
    for x in resume["ExperienceInfos"]:
        model = {"org_obj": job["experience_needed"], "similarity": 0}
        exp_similarity = cosine_similarity(np.array([job["experience_needed"]["field_of_experience_vector"]]),np.array([x["JobTitleVector"]]))
        if model["similarity"] < exp_similarity:
            model["similarity"] = exp_similarity
            model["tar_obj"] = x
        if model["similarity"] > 0.48:
            common_experience.append(model)

    # common_skills = []
    # for x in job["skill_needed_with_vector"]:
    #     model = {"org_obj": x, "similarity": 0}
    #     for y in resume["SkillInfos"]:
    #         similarity = cosine_similarity(np.array([x["vector"]]), np.array([y["SkillNameVector"]]))
    #         print(similarity,y['SkillName'],'-',x['name'])
    #         if model["similarity"] < similarity:
    #            model["similarity"] = similarity
    #            model["tar_obj"] = y
    #     if float(model["similarity"]) > 0.8:
    #         common_skills.append(model)

    common_skills = []
    for x in job["new_skill_set"]:
        model = {"org_obj": x, "similarity": 0}
        for y in resume["new_skill_set"]:

            similarity = cosine_similarity(np.array([x["sentence_embedding_small"]]), np.array([y["related_skill_sentence_embedding_small"]]))
            #print(similarity, y['org_skill_title'], '-', x['skill'])
            if model["similarity"] < similarity:
                model["similarity"] = similarity
                model["tar_obj"] = y
        if float(model["similarity"]) > 0.65:
            common_skills.append(model)

    final_model = {'resume': resume, "job": job, 'common_education': common_education,
                   'common_experience': common_experience, 'common_skills': common_skills}

    return final_model





