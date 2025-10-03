import json
from fuzzywuzzy import fuzz
from utils.mongohelper import MongoHelper
from openai import OpenAI

from utils.spacyhelper import preprocess_text

db = MongoHelper('cvbuilder')
client = OpenAI(api_key="xxxxxxxxxxxxxxxxxx")

skill_sentences = list(db.get_collection('skill_sentences').find({"sentence_embedding_large":{"$exists":False}},{"skill":1,"sentence":1}))


# for x in skill_sentences:
#     for y in skill_sentences:
#         if str(y["skill"]).lower() == str(x["skill"]).lower() and x["_id"] != y["_id"]:
#             db.get_collection('skill_sentences').delete_one(y)
         # similarity_ratio = fuzz.ratio(x["skill"], y['skill'])
         # if similarity_ratio > 90 and similarity_ratio < 100:
         #     print('x', x['skill'], '-', 'y', y['skill'])
         #     user_input = input("Please enter X or Y:")
         #     if user_input == 'x':
         #         db.get_collection('skill_sentences').delete_one(x)
         #     elif user_input == 'y':
         #         db.get_collection('skill_sentences').delete_one(y)




    # similarity_ratio = fuzz.ratio(x["skill"], y['skill'])
        # if similarity_ratio > 90 and similarity_ratio < 100:
        #     list.append({"similarity_ratio":similarity_ratio,"x": x["skill"],"y": y["skill"]})
        #     print(x["skill"],y["skill"])


for i,x in enumerate(skill_sentences):
    print(i)
    vec = client.embeddings.create(input=[preprocess_text(x['sentence'])],model="text-embedding-3-large").data[0].embedding
    db.get_collection("skill_sentences").update_one({"_id":x['_id']},{"$set":{"sentence_embedding_large":vec}})



def generate_sentences(skills_batch):
    skills = (", ".join(skills_batch))
    prompt_template = f'I have a list of resume skills, and I need to generate sentences that effectively describe each skill in a way that captures its typical usage or relationship with other relevant technologies or concepts. The goal is to use these sentences for generating embeddings that reflect the practical context and associations of each skill. For each skill, please generate a sentence that includes the skill and contextualizes it within a relevant scenario, industry practice, or common use case. Skills: {skills} response type in json: skills:[{{"skill":"", "sentence":""}}]'

    try:
        response = client.chat.completions.create(
            temperature=0.1,
            model="gpt-4o-mini",  # or use "gpt-3.5-turbo"
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt_template}
            ],response_format={"type": "json_object"}
        )

        # Extracting the generated sentences from the response
        generated_sentences = response.choices[0].message.content
        return generated_sentences

    except Exception as e:
        print(f"An error occurred: {e}")
        return None


# skills = list(db.get_collection("clean_skills").find({}))
# sentences = list(db.get_collection("skill_sentences").find({}))
#
#
#
# skillss = list(set([x for item in skills for x in item["related_skill"]]))
#
# final = [x for x in skillss if x not in [y["skill"] for y in sentences]]
# for x in final:
#     if x == "Redis":
#         print("s")
# for x in range(0,len(final),50):
#     sentences = json.loads(generate_sentences(final[x:x+50]))
#     for y in sentences['skills']:
#         db.get_collection("skill_sentences").insert_one(y)
#         print(y)
