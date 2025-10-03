

from fuzzywuzzy import fuzz, process
from utils.spacyhelper import preprocess_text
from utils.mongohelper import MongoHelper

db = MongoHelper("cvbuilder")


resumes = list(db.get_collection("resumesv3").find({"JobCategory.EnTitle":"Software Development and Programming"},{"SkillInfos":1}))
#related_skills = list(db.get_collection("clean_skills").find({}))
skill_sentences = list(db.get_collection("skill_sentences").find({}))


for resume in resumes:
    try:
        new_skill_set = []
        for resume_skill in resume["SkillInfos"]:
            related_skills_temp =db.get_collection("clean_skills").find_one({"original": preprocess_text(resume_skill["SkillName"]) },{"related_skill": 1})
            if related_skills_temp is None:
                continue
            for related_skill in related_skills_temp["related_skill"]:
                sentence = db.get_collection("skill_sentences").find_one({"skill": related_skill})
                if sentence is None:
                    continue
                    # best_match = process.extractBests(related_skill, [x["skill"] for x in skill_sentences])
                    # if best_match is None:
                    #     continue
                    # sentence = db.get_collection("skill_sentences").find_one({"skill": best_match[0][0]})
                new_skill_set.append(
                    {"org_skill_title": resume_skill["SkillName"],
                     "related_skill_title": related_skill,
                     "sentence": sentence["sentence"],
                     "related_skill_sentence_embedding_small": sentence["sentence_embedding_small"],
                     "related_skill_sentence_embedding_large": sentence["sentence_embedding_large"]})
        db.get_collection("resumesv3").update_one({"_id":resume["_id"]},{"$set":{"new_skill_set":new_skill_set}})
    except Exception as ex:
        print(ex)
        continue

