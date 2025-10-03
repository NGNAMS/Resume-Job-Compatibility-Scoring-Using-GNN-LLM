import torch
from utils.mongohelper import MongoHelper
import os
from model import SimpleGAT
from graph_building.model_builder import model_builder
from utils.visualize_graph import draw_graph, build_graph
from bson import ObjectId
import matplotlib.pyplot as plt
import shap
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score



path = os.path.join(os.path.dirname(__file__))
model = torch.load(f'{path}/model2.pth',map_location=torch.device('cpu') )
data_set = torch.load(f'{path}/small_skill_with_title_dataset_six_point_five.pt',map_location=torch.device('cpu'))
db = MongoHelper("cvbuilder")


relevances = list(db.get_collection("relevance").find({}))
train_data, test_data = train_test_split(data_set, test_size=0.2, random_state=42)

model.eval()

y_pred = []
y_true = []
for x in test_data:
  if x["graph_id"] in [str(f["_id"]) for f in relevances]:
    score, attention_scores = model(x)
    predicted_score = score.item()
    real_score = [int(z["score"]) for z in relevances if str(z["_id"]) == x["graph_id"]][0]
    y_true.append(True)
    if abs(predicted_score - real_score) <= 5:
      y_pred.append(True)
    else:
      y_pred.append(False)
  
    


print(accuracy_score(y_true, y_pred))