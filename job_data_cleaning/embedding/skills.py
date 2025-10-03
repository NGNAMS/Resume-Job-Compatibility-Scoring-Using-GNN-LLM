import json

import pandas as pd
from openai import OpenAI
from utils.mongohelper import MongoHelper
from utils.spacyhelper import preprocess_text
from fuzzywuzzy import fuzz
from fuzzywuzzy import process

db = MongoHelper('cvbuilder')
client = OpenAI(api_key="xxxxxxxxxxxxxxxxxx")

skilss = list(db.get_collection("resumesv3").find({"JobCategory.EnTitle": "Software Development and Programming"},{"SkillInfos.SkillName":1}))


final_list = []
for x in skilss:
    for y in x["SkillInfos"]:
        final_list.append(preprocess_text(y["SkillName"]))

unique_list = list(set(final_list))

for skill in unique_list:
    chat = client.chat.completions.create(model='gpt-4o-mini', temperature=0.1, messages=[
        {"role": "user", "content": '''I will send you a user written resume skill some of them might have apelling error some of them can be combination of many skills and some of them could be meaningless
        I want you to send me back the most related skill that someone can mention in resume or if you are not sure please return empty array as related skills.
        I don't want you to add more skills for example for API Testing you should return just API Testing not for example Quality Assurance.  
        I you find combination of skills please seperate them and return related for eah one
        your response should be json object 
        here is the sample of json you should generate:
            {
                "original": "",
                "related_skill": [
                    
                ]
            }
            '''},
        {"role": "user", "content": skill}], response_format={"type": "json_object"})
    print(chat.choices[0].message.content)
    db.get_collection("clean_skills").insert_one(json.loads(chat.choices[0].message.content))
print("ds")