from openai import OpenAI
import time
client = None

import json

OPEN_AI_MODEL = "gpt-4"

with open('api_key.json') as f:
    data = json.load(f)
    print(data['personal_org'])
    print(data['secret_key'])
    client = OpenAI( 
        organization= data['fried_nlp'],
        api_key= data['secret_key']
    )


codex_prompt = "./prompts/chatapi.prompt"

with open(codex_prompt) as f:
    base_prompt = f.read().strip()




q = "What toy is this?"
image_caption = "A man is sitting in a pew at a church, wearing a backpack and holding a teddy bear. The teddy bear is positioned on his back, and he appears to be looking at it. There are other people in the church as well."


example_q = "In what sort of building might this be found?"
example_caption = "The image features a small, white hospital bed with a white pillow on it. The bed is placed in a room with a wooden floor."



new_prompt = base_prompt.replace("INSERT_QUERY_HERE", q)


original_context = ""

def feedback_conversation(original_context):

    print()
    print("Original Context (Message to Code Gen LLM): ", original_context)

    bot_1_response = client.chat.completions.create(
        model=OPEN_AI_MODEL,
        messages=[
            {"role": "system", "content": "Only answer with a function starting def execute_command."},
            {"role": "user", "content": new_prompt}
        ],
        temperature=0.,
        max_tokens=512,
        top_p = 1.,
        frequency_penalty=0,
        presence_penalty=0,
        stop=["\n\n"],
    )

    print(f"Code Generator LLM: {bot_1_response.choices[0].message.content}\n")

    original_context = bot_1_response.choices[0].message.content + "\n Caption: "  + image_caption + "\n Query: " + q

    print("Message to Reviewer LLM: ", original_context)

    bot_2_response = client.chat.completions.create(
        model=OPEN_AI_MODEL,
        messages=[
            {"role": "system", "content": f"""As a code peer reviewer, your role is to determine if something is wrong in the code based on the caption.
            Think step-by-step as to why the code is correct or not to make sure the following 2 conditions are met and explain why for each of the conditions: \n
            Condition-1: Has the generated code utilized hints from the image caption? Give your explanation on what part of image caption is relevant to the image and how can the code be refined to leverage the hints. \n
            Condition-2: llm_query() function only takes a f-string as an argument which begins with the letter 'f'. A formatted string ALWAYS begins with the letter 'f'. Example: object_patch.llm_query(f"Who invented {{object_name\}}?"). Here the string f"Who invented {{object_name\}}?" starts with an f before the quotes. Incorrect: object_patch.llm_query("Who invented this object?") \n 
             
            If both conditions are satisfied, then the code is correct and no further feedback is required.
                        
            In Context Example:
            Code:
            def execute_command(image):
                image_patch = ImagePatch(image)
                building_patches = image_patch.find(""building"")
                building_patch = building_patches[0]
                building_name = building_patch.simple_query(""What is the name of the building?"")
                return building_patch.llm_query(""In what sort of building might this be found?"")
            Caption: {example_caption} \n
            Query: {q} \n
            
            Answer:
            Condition-1: The code is relevant to the caption since it tries to find the building which the caption is describing. The code can leverage the fact that the caption describes a hospital.  \n
            Condition-2: llm_query() function usage is incorrect as the string does not start with an 'f' and there are no variables inside the f-string from previous steps. \n
            """},
            {"role": "user", "content": original_context}
        ],
        temperature=0.,
        max_tokens=512,
        top_p = 1.,
        frequency_penalty=0,
        presence_penalty=0,
        # stop=["\n\n"],
    )

    print(f"Code Reviewer LLM: {bot_2_response.choices[0].message.content}\n")

    original_context = original_context + "\n" + bot_2_response.choices[0].message.content

    return original_context

    # return bot_2_response.choices[0].message.content


for i in range(2):
    print("Feedback Loop Iteration: ", i)
    original_context = feedback_conversation(original_context) + "\n Refine the code based on the feedback so that Condition-1 and Condition-2 are satisfied. \n"
    # print("Original Context after conversation: ", original_context + "\n Refine the code based on the feedback. \n")
    time.sleep(10)

