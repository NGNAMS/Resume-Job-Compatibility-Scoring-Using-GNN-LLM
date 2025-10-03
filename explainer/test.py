import torch
from utils.mongohelper import MongoHelper
import os
from model import SimpleGAT
from graph_building.model_builder import model_builder
from utils.visualize_graph import draw_graph, build_graph
from bson import ObjectId
import matplotlib.pyplot as plt
import shap

class ModelWrapper:
    def __init__(self, model):
        self.model = model

    def __call__(self, data):
        # Forward pass for the model, only return the score (not attention scores)
        score, _ = self.model(data)  # We are discarding attention scores here
        return score.unsqueeze(1).detach().cpu().numpy()

path = os.path.join(os.path.dirname(__file__))
model = torch.load(f'{path}/model2.pth',map_location=torch.device('cpu') )
data_set = torch.load(f'{path}/small_skill_with_title_dataset_six_point_five.pt',map_location=torch.device('cpu'))
db = MongoHelper("cvbuilder")

target_id = '66c614cf92aa892003e64237'
i = 0
relevances = db.get_collection("relevance").find({})
for x in relevances:
  graph_model = model_builder(job_id=x["job_id"], resume_id=x["resume_id"])
  if len([y['org_obj']['skill'] for y in graph_model['common_skills']]) <= 3 and x['score'] < 50:
    target_id = str(x['_id'])
    i = i + 1
    if i == 1:
      break
relevance = db.get_collection("relevance").find_one({'_id': ObjectId(target_id)})
graph_model = model_builder(job_id=relevance["job_id"], resume_id=relevance["resume_id"])

# for i,relevance in enumerate(relevances):

target_pair = [g for g in data_set if g['graph_id'] == target_id][0]
skill_titles = target_pair['skill'].title
resume_uncommon_skill_titles = [x['org_skill_title'] for x in graph_model['resume']['new_skill_set'] if x['org_skill_title'] not in [z['tar_obj']['org_skill_title'] for z in graph_model['common_skills']]]
job_uncommon_skill_titles = [x['skill'] for x in graph_model['job']['new_skill_set'] if x['skill'] not in [z['org_obj']['skill'] for z in graph_model['common_skills']]]
common_skills_titles = [x['org_obj']['skill'] for x in graph_model['common_skills']]
combined_skills_titles = job_uncommon_skill_titles + common_skills_titles + resume_uncommon_skill_titles



model.eval()
score, attention_scores = model(target_pair)
print(target_pair)
wrapped_model = ModelWrapper(model)


# Extract the data from target_pair
skill_data = torch.tensor(target_pair.x_dict['skill'], dtype=torch.float32)
edge_index = torch.tensor(target_pair.edge_index_dict[('skill', 'to', 'resume')], dtype=torch.long)

# Wrap the model into ModelWrapper
wrapped_model = ModelWrapper(model)

# Prepare the input data for SHAP (a tuple of skill node data and edge data)
input_data = (skill_data, edge_index)

# Create the SHAP Explainer with the wrapped model
explainer = shap.DeepExplainer(wrapped_model, target_pair, framework="pytorch")

# Compute SHAP values
shap_values = explainer.expected_value

# Display the results
print(shap_values)


skill_to_resume_edge_score = torch.max(attention_scores['skill_to_resume'][1],dim=1).values.detach().numpy()
skill_to_resume_edge = attention_scores['skill_to_resume'][0].numpy()
# skill_to_job_edge_score = torch.max(attention_scores['skill_to_job'][1],dim=1).values.detach().numpy()
# skill_to_job_edge = attention_scores['skill_to_job'][0].numpy()
result = []

for i,x in enumerate(skill_to_resume_edge[0]):
  result.append({"skill":skill_titles[x] , "score":skill_to_resume_edge_score[i]})
  
# Extracting skills and scores
skills = [item['skill'] for item in result]
scores = [item['score'] for item in result]

# Plotting
plt.figure(figsize=(10, 6))
plt.barh(skills, scores, color='skyblue')
plt.xlabel('Score')
plt.title('Skill Scores')
plt.gca().invert_yaxis()  # Invert y-axis to have the highest score at the top
plt.show()

build_graph(graph_model,result)
print(result)
print('finish')