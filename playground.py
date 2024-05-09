import openai

from openai import OpenAI
client = None

import json

with open('api_key.json') as f:
    data = json.load(f)
    print(data['personal_org'])
    print(data['secret_key'])
    openai_client = OpenAI( 
        organization= data['fried_nlp'],
        api_key= data['secret_key']
    )


model =  'gpt-3.5-turbo'
messages =  [{'role': 'user', 'content': 'What is the type of this plant?'}]
max_tokens =  256
temperature =  0.0
stream =  False
stop =  None
top_p =  1
frequency_penalty =  0
presence_penalty =  0
n_votes =  1

prompt =  ['What is the type of this plant?']


response_og = openai_client.completions.create(
    model=model,
    prompt=prompt,
    max_tokens=max_tokens,
    temperature=temperature,
    stream=stream,
    stop=stop,
    top_p=top_p,
    frequency_penalty=frequency_penalty,
    presence_penalty=presence_penalty,
    n=n_votes,
)
print("GPT-3 response: ", response_og)


# completion = client.chat.completions.create(
#   model="gpt-3.5-turbo",
#   messages=[
#     {"role": "system", "content": "You are a poetic assistant, skilled in explaining complex programming concepts with creative flair."},
#     {"role": "user", "content": "Compose a poem that explains the concept of recursion in programming."}
#   ]
# )

# prompt = ['Compose a poem that explains the concept of recursion in programming.']


# messages = [{"role": "user", "content": p} for p in prompt]
# response = client.chat.completions.create(
#     model="gpt-3.5-turbo",
#     messages=messages,
#     max_tokens=40,
# )





# print(completion.choices[0].message)

# #Read openai_responses.pkl file
# import pickle
# with open('openai_responses.pkl', 'rb') as f:
#     openai_responses = pickle.load(f)



# # Code for processing VQA images
# dataSubType ='train2014'
# imgId = randomAnn[0]['image_id']
# imgFilename = 'COCO_' + dataSubType + '_'+ str(imgId).zfill(12) + '.jpg'
# if os.path.isfile(imgDir + imgFilename):
#   I = io.imread(imgDir + imgFilename)
#   plt.imshow(I)
#   plt.axis('off')
#   plt.show()
    

#Search if a file exists in a directory
# import os

# imgDir = 'data/OkVQA/val2014'

# image_file_name = 'COCO_val2014_000000297147.jpg'

# if os.path.isfile(imgDir + '/' + image_file_name):
#     print('File exists')
# else:
#     print('File does not exist')