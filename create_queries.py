# index,sample_id,possible_answers,query_type,query,answer,image_name
# 0,0,purple,,What color do you get if you combine the colors of the viper and the flower?,purple,viper_flower.png
# 0,0,,,Tell me about the competition between the two skyscrapers in the image.,,skyscrapers.png

# Get filepath for questions as input
import sys
import json
from tqdm import tqdm

# Get filepath for questions as input
if len(sys.argv) < 2:
    questions_filepath = 'data/OkVQA/OpenEnded_mscoco_val2014_questions.json'
    annotations_filepath = 'data/OkVQA/mscoco_val2014_annotations.json'
else:
    questions_filepath = sys.argv[1]
    annotations_filepath = sys.argv[2]

#Read questions file json and iterate over questions

data_dict= { 'sample_id':[],'possible_answers':[],'query_type':[],'query':[],'answer':[],'image_name':[]}

with open(questions_filepath) as f:
    questions = json.load(f)
    # print(questions.keys())
    for question in tqdm(questions['questions']):
        data_dict['sample_id'].append(question['question_id'])
        data_dict['query_type'].append('')
        data_dict['query'].append(question['question'])
        data_dict['image_name'].append('COCO_val2014_'+str(question['image_id']).zfill(12)+'.jpg')


with open(annotations_filepath) as f:
    annotations = json.load(f)
    for item in tqdm(data_dict['sample_id']):
        for annotation in annotations['annotations']:
            if item == annotation['question_id']:
                data_dict['answer'].append(annotation['answers'][0]['answer'])
                data_dict['possible_answers'].append('')



#Save data_dict as csv file
import pandas as pd
df = pd.DataFrame(data_dict)
df.to_csv('data/OkVQA/OkVQA_val.csv', index=True)

#print first row of data_dict
print(data_dict['sample_id'][0],data_dict['possible_answers'][0],data_dict['query_type'][0],data_dict['query'][0],data_dict['answer'][0],data_dict['image_name'][0])




        








