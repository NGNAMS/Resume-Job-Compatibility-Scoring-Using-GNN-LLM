from fuzzywuzzy import process

from utils.mongohelper import MongoHelper

db = MongoHelper("cvbuilder")


jobs = list(db.get_collection("jobs2").find({}))
final_skills = list(db.get_collection("skill_sentences").find({}))

test = []
for job in jobs:
    for skill in job["skill_needed_with_vector"]:
        best_match = process.extractBests(str(skill['name']).lower(), [str(x["skill"]).lower() for x in final_skills])
        if best_match[0][1]  < 95:
            test.append(skill["name"])
            print(skill["name"],best_match[0])
            save = input("save in db?")
            if save == 'y':
                db.get_collection('skill_sentences2').insert_one({"skill":skill["name"]})
        #raise ValueError("Cannot divide by zero!")
print(len(set(test)))