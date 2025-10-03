import networkx as nx
from utils.visualize_graph import draw_graph
from graph_building.model_builder import model_builder

model = model_builder(job_id='66bd154fd741dcb20c94f445',resume_id='66a28dd1d2c8fab2d03d5cab')

DG = nx.DiGraph()
DG.add_node('job-title',label=model['job']['job_title'],pos=(5,3))
DG.add_node('resume-title',label=model['resume']['BasicInfo']['ResumeTitle'],pos=(-5,3))


#Adding common skills
for index,x in enumerate(model['common_skills']):
    DG.add_node(x['org_obj']['skill'], label=x['org_obj']['skill'],pos=(0,10 - (index + 2)))
    DG.add_node(x['tar_obj']['org_skill_title'], label=x['tar_obj']['org_skill_title'], pos=(-3, 10 - (index + 2)))
    DG.add_edge('job-title',x['org_obj']['skill'],label='requires')
    DG.add_edge('resume-title', x['tar_obj']['org_skill_title'],label='has')
    DG.add_edge(x['tar_obj']['org_skill_title'], x['org_obj']['skill'],label='fulfills')

#Adding Uncommon skills to job node
for index,x in enumerate([y for y in model['job']['new_skill_set'] if y['skill'] not in [z['org_obj']['skill'] for z in model['common_skills']]]):
     DG.add_node(x['skill'], label=x['skill'], pos=(10, 10 - (index + 2)))
     DG.add_edge('job-title', x['skill'],label='to')

#Adding Uncommon skills to resume node
for index,x in enumerate([y for y in model['resume']['new_skill_set'] if y['org_skill_title'] not in [z['tar_obj']['org_skill_title'] for z in model['common_skills']]]):
     DG.add_node(x['org_skill_title'], label=x['org_skill_title'], pos=(-10, 10 - (index + 2)))
     DG.add_edge('resume-title', x['org_skill_title'],label='to')

#Adding job required education
#DG.add_node(model["job"]["education_needed"]["field_of_study_and_level_of_study_title"], label=model["job"]["education_needed"]["field_of_study_and_level_of_study_title"], pos=(3, 10 - (len(model['common_skills']) * 1.5)))
#DG.add_edge('job-title', model["job"]["education_needed"]["field_of_study_and_level_of_study_title"], label='requires')

#Adding resume common  education
# for index,x in enumerate(model['common_education']):
#     DG.add_node(x['tar_obj']['field_of_study_and_level_of_study_title'], label=x['tar_obj']["field_of_study_and_level_of_study_title"], pos=(-3, 10 - (len(model['common_skills']) * 1.5)))
#     DG.add_edge('resume-title', x['tar_obj']['field_of_study_and_level_of_study_title'], label='has')
#     DG.add_edge(x['tar_obj']['field_of_study_and_level_of_study_title'], model["job"]["education_needed"]["field_of_study_and_level_of_study_title"], label='fulfills')

#Adding resume remaining education
# for index,x in enumerate([y for y in model['resume']['EducationInfos'] if y['SkillName'] not in [z['tar_obj']['SkillName'] for z in model['common_skills']]]):
#     DG.add_node(x['SkillName'], label=x['SkillName'], pos=(-10, 10 - (index + 2)))
#     DG.add_edge('resume-title', x['SkillName'],label='has')

#Adding job required expereince
#DG.add_node('general-job-ex-req',label=model["job"]["experience_needed"]["field_of_experience"],pos=(3,-3))
#DG.add_edge('job-title','general-job-ex-req',label='requires')

#Adding resume common experience
# for index,x in enumerate(model['common_experience']):
#     DG.add_node(x['tar_obj']['JobTitle'], label=x['tar_obj']['JobTitle'], pos=(-4, -2))
#     DG.add_edge('resume-title', x['tar_obj']['JobTitle'], label='has')
#     DG.add_edge(x['tar_obj']['JobTitle'], 'general-job-ex-req', label='fulfill')



draw_graph(DG)

