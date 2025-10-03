import matplotlib.pyplot as plt
import networkx as nx

def draw_graph(G):
    
    pos = nx.get_node_attributes(G, 'pos')

    # Use a layout algorithm for nodes without specified positions
    #remaining_pos = nx.spring_layout(G, pos=pos, fixed=pos.keys(), k=0.5, iterations=50)

    # Combine manual and automatic positions
    #pos.update(remaining_pos)
    #pos = nx.spring_layout(G)
    labels = nx.get_node_attributes(G, 'label')
    edge_labels = nx.get_edge_attributes(G, 'label')
    plt.figure(figsize=(30, 50))
    nx.draw_networkx(G, with_labels=False, node_color='white',edgecolors='black',edge_color='#2929291a', node_size=1000, font_size=6, font_color='black',
            font_weight='normal',pos=pos)
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=8, font_color='black', font_weight='normal',verticalalignment='center',horizontalalignment='center')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8, font_color='red')
    plt.title('NetworkX Graph Visualization')
    plt.show()
#     plt.pause(1)  # Pause for 1 second to allow viewing
#     plt.clf()  # Clear the figure for the next plot


def build_graph(model,edge_scores=None):
        DG = nx.DiGraph()
        DG.add_node('job-title',label=model['job']['job_title'],pos=(5,3))
        DG.add_node('resume-title',label=model['resume']['BasicInfo']['ResumeTitle'],pos=(-5,3))


        #Adding common skills
        for index,x in enumerate(model['common_skills']):
                DG.add_node(x['org_obj']['skill'], label=x['org_obj']['skill'],pos=(0,10 - (index + 2)))
                #DG.add_node(x['tar_obj']['org_skill_title'], label=x['tar_obj']['org_skill_title'], pos=(-3, 10 - (index + 2)))
                DG.add_edge('job-title',x['org_obj']['skill'],label='requires')
                DG.add_edge('resume-title', x['org_obj']['skill'],label=[y['score'] for y in edge_scores if y['skill'] == x['org_obj']['skill']] if edge_scores else 'has' )
                #DG.add_edge(x['tar_obj']['org_skill_title'], x['org_obj']['skill'],label='fulfills')

        #Adding Uncommon skills to job node
        for index,x in enumerate([y for y in model['job']['new_skill_set'] if y['skill'] not in [z['org_obj']['skill'] for z in model['common_skills']]]):
                DG.add_node(x['skill'], label=x['skill'], pos=(10, 10 - (index + 2)))
                DG.add_edge('job-title', x['skill'],label='to')

        #Adding Uncommon skills to resume node
        for index,x in enumerate([y for y in model['resume']['new_skill_set'] if y['org_skill_title'] not in [z['tar_obj']['org_skill_title'] for z in model['common_skills']]]):
                 DG.add_node(x['org_skill_title'], label=x['org_skill_title'], pos=(-10, 10 - (index + 2)))
                 DG.add_edge('resume-title', x['org_skill_title'],label=[y['score'] for y in edge_scores if y['skill'] == x['org_skill_title']] if edge_scores else 'has' )

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


