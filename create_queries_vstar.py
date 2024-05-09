# index,sample_id,possible_answers,query_type,query,answer,image_name
# 0,0,purple,,What color do you get if you combine the colors of the viper and the flower?,purple,viper_flower.png
# 0,0,,,Tell me about the competition between the two skyscrapers in the image.,,skyscrapers.png

# Get filepath for questions as input
import sys
import json
from tqdm import tqdm
import os
import pandas as pd


# Get filepath for questions as input

annotations_filepath = sys.argv[1]

#Read questions file json and iterate over questions

data_dict= {'sample_id':[],'possible_answers':[],'query_type':[],'query':[],'answer':[],'image_name':[]}


for filename in os.listdir(annotations_filepath):
    if filename.endswith(".json"):
        annotations_sample_filepath = os.path.join(annotations_filepath, filename)
        with open(annotations_sample_filepath) as f:
            annotations = json.load(f)
            data_dict['sample_id'].append(filename.split('.')[0].split('_')[-1])
            data_dict['query_type'].append(annotations_filepath.split('/')[-1])
            #append options to query?
            data_dict['query'].append(annotations['question'])
            data_dict['image_name'].append(filename.split('.')[0]+'.jpg')
            data_dict['answer'].append(annotations['options'][0])
            data_dict['possible_answers'].append(annotations['options'])
        





#Save data_dict as csv file
df = pd.DataFrame(data_dict)
save_path = 'data/V_Star/v_star_'+annotations_filepath.split('/')[-1]+'.csv'
df.to_csv(save_path, index=True)

#print first row of data_dict
print(data_dict['sample_id'][0],data_dict['possible_answers'][0],data_dict['query_type'][0],data_dict['query'][0],data_dict['answer'][0],data_dict['image_name'][0])




        








