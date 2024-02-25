from openai import OpenAI
client = None

import json

with open('api_key.json') as f:
    data = json.load(f)
    print(data['personal_org'])
    print(data['secret_key'])
    client = OpenAI( 
        organization= data['personal_org'],
        api_key= data['secret_key']
    )


codex_prompt = "./prompts/chatapi.prompt"

with open(codex_prompt) as f:
    base_prompt = f.read().strip()


p = "Who might play here?"


new_prompt = base_prompt.replace("INSERT_QUERY_HERE", p)

# print(new_prompt)

response1 = 'def execute_command(image):\n    image_patch = ImagePatch(image)\n    # Find the patches containing the word "play"\n    play_patches = image_patch.find("play")\n    # Sort the patches based on their vertical position\n    play_patches.sort(key=lambda x: x.vertical_center)\n    # Get the patch with the lowest vertical position\n    lowest_play_patch = play_patches[0]\n    # Ask the question about who might play in that location\n    return lowest_play_patch.llm_query("Who might play here?")'

image_caption = "image depicts a living room with a couch, a television, and a baby play area. The baby play area is filled with toys, including a baby walker, and a high chair."

response2 = 'def execute_command(image):\n    image_patch = ImagePatch(image)\n    # Find the patches containing the word "play"\n    play_patches = image_patch.find("play")\n    # Sort the patches based on their vertical position\n    play_patches.sort(key=lambda x: x.vertical_center)\n    # Get the patch with the lowest vertical position\n    lowest_play_patch = play_patches[0]\n    # Check if the lowest play patch contains a baby walker or high chair\n    if lowest_play_patch.exists("baby walker") or lowest_play_patch.exists("high chair"):\n        # Ask the question about who might play in that location\n        return lowest_play_patch.llm_query("Who might play here?")\n    else:\n        return "No specific information available about who might play here.'

responses = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            # {"role": "system", "content": "You are a helpful assistant."},
            {"role": "system", "content": "Only answer with a function starting def execute_command."},
            {"role": "user", "content": new_prompt},
            {"role": "assistant", "content": response1},
            {"role": "user", "content": f"Given the image caption, refine your code to make it more specific to the image. {image_caption}"},
            {"role": "assistant", "content": response2},
            {"role": "user", "content": """Note that llm_query function only takes a formatted string as an input using variables from previous steps. Correct the code and only answer with a function starting def execute_command.
             Example: object_patch.llm_query(f"Who invented \{object_name\}?
             """},
        ],
        temperature=0.,
        max_tokens=512,
        top_p = 1.,
        frequency_penalty=0,
        presence_penalty=0,
        stop=["\n\n"],
        )
            



print(responses.choices[0].message.content)



#Response-1
"""
def execute_command(image):\n    
image_patch = ImagePatch(image)\n    
# Find the patches containing the word "play"\n    
play_patches = image_patch.find("play")\n    
# Sort the patches based on their vertical position\n    
play_patches.sort(key=lambda x: x.vertical_center)\n    
# Get the patch with the lowest vertical position\n    
lowest_play_patch = play_patches[0]\n    
# Ask the question about who might play in that location\n    
return lowest_play_patch.llm_query("Who might play here?")'
"""

#Response-2
"""
def execute_command(image):\n    
image_patch = ImagePatch(image)\n    
# Find the patches containing the word "play"\n    
play_patches = image_patch.find("play")\n    
# Sort the patches based on their vertical position\n    
play_patches.sort(key=lambda x: x.vertical_center)\n    
# Get the patch with the lowest vertical position\n    
lowest_play_patch = play_patches[0]\n    
# Check if the lowest play patch contains a baby walker or high chair\n    
if lowest_play_patch.exists("baby walker") or lowest_play_patch.exists("high chair"):\n        
# Ask the question about who might play in that location\n        
return lowest_play_patch.llm_query("Who might play here?")\n    
else:\n        
return "No specific information available about who might play here.
"""


#Response-3
"""
def execute_command(image):\n    
image_patch = ImagePatch(image)\n    
# Find the patches containing the word "play"\n    
play_patches = image_patch.find("play")\n    
# Sort the patches based on their vertical position\n    
play_patches.sort(key=lambda x: x.vertical_center)\n    
# Get the patch with the lowest vertical position\n    
lowest_play_patch = play_patches[0]\n    
# Check if the lowest play patch contains a baby walker or high chair\n    
if lowest_play_patch.exists("baby walker") or lowest_play_patch.exists("high chair"):\n        
# Get the name of the object in the lowest play patch\n        
object_name = lowest_play_patch.simple_query()\n        
# Ask the question about who might play in that location\n        
return lowest_play_patch.llm_query(f"Who might play here with {object_name}?")\n    
else:\n        
return "No specific information available about who might play here."
"""

