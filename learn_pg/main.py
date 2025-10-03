import numpy as np
import torch
from torch_geometric.data import Data, HeteroData, InMemoryDataset
from torch_geometric.utils import to_networkx
from graph_building.model_builder import model_builder
from utils.visualize_graph import draw_graph
from torch_geometric.datasets import TUDataset
from torch_geometric.nn import GCNConv
from utils.mongohelper import MongoHelper
from random import shuffle, sample




db = MongoHelper("cvbuilder")

relevances = list(db.get_collection("relevance").find({}))



result = []
for i,relevance in enumerate(relevances):
    try:
        model = model_builder(job_id=relevance["job_id"], resume_id=relevance["resume_id"])
        data = HeteroData()
        data.graph_id = str(relevance['_id'])
        data['resume'].x = torch.ones(1,1536, dtype=torch.float)
        data['job'].x = torch.ones(1,1536, dtype=torch.float)
        resume_uncommon_skill = [x['related_skill_sentence_embedding_small'] for x in model['resume']['new_skill_set'] if x['org_skill_title'] not in [z['tar_obj']['org_skill_title'] for z in model['common_skills']]]
        resume_uncommon_skill_titles = [x['org_skill_title'] for x in model['resume']['new_skill_set'] if x['org_skill_title'] not in [z['tar_obj']['org_skill_title'] for z in model['common_skills']]]

        #resume_uncommon_skill = [np.random.uniform(-1, 1, 1536).astype(np.float32) for x in model['resume']['new_skill_set'] if x['org_skill_title'] not in [z['tar_obj']['org_skill_title'] for z in model['common_skills']]]
        job_uncommon_skill = [x['sentence_embedding_small'] for x in model['job']['new_skill_set'] if x['skill'] not in [z['org_obj']['skill'] for z in model['common_skills']]]
        job_uncommon_skill_titles = [x['skill'] for x in model['job']['new_skill_set'] if x['skill'] not in [z['org_obj']['skill'] for z in model['common_skills']]]

        #job_uncommon_skill = [np.random.uniform(-1, 1, 1536).astype(np.float32) for x in model['job']['new_skill_set'] if x['skill'] not in [z['org_obj']['skill'] for z in model['common_skills']]]
        common_skills = [x['org_obj']['sentence_embedding_small'] for x in model['common_skills']]
        common_skills_titles = [x['org_obj']['skill'] for x in model['common_skills']]

        #common_skills = [np.random.uniform(-1, 1, 1536).astype(np.float32) for x in model['common_skills']]
        combined_skills = job_uncommon_skill + common_skills + resume_uncommon_skill
        combined_skills_titles = job_uncommon_skill_titles + common_skills_titles + resume_uncommon_skill_titles

        data['skill'].x = torch.tensor(combined_skills)
        data['skill'].title = combined_skills_titles  # Add titles to nodes

        test = [[_ for _ in range(len(job_uncommon_skill), (len(job_uncommon_skill) + len(resume_uncommon_skill) + len(common_skills)))],[0] * (len(resume_uncommon_skill) + len(common_skills))]
        data['skill', 'to', 'job'].edge_index = torch.tensor([[_ for _ in range(0, (len(job_uncommon_skill) + len(common_skills)))],[0] * (len(job_uncommon_skill) + len(common_skills))])
        data['skill', 'to', 'resume'].edge_index = torch.tensor(test)
        data.y = torch.tensor([relevance["score"]],dtype=torch.float)
        result.append(data)
        print(i)
    except:
        continue
    

torch.save(result, 'small_skill_with_title_dataset_six_point_five.pt')