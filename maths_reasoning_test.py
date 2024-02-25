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

llm_response0 = """The total number of letters in the given string is 13. The number of 'q' is 4, 'k' is 8, and 'l' is 1.

The probability of picking 'q' first is 4/13.

Then, the probability of picking 'q' second (without replacement) is 3/12.

Finally, the probability of picking 'l' third is 1/11.

So, the probability of the sequence 'qql' is (4/13) * (3/12) * (1/11) = 1/143."""


user_response1  = "Is the count of the sequnce correct? Think in a step by step manner and correct your solution."


def feedback_conversation(original_context):

    print()
    print("Original Context (Message to Code Gen LLM): ", original_context)

    bot_1_response = client.chat.completions.create(
        model=OPEN_AI_MODEL,
        messages=[
            # {"role": "system", "content": "Only answer with a function starting def execute_command."},
            {"role": "user", "content": original_context},
            {"role": "assistant", "content": llm_response0},
            {"role": "user", "content": user_response1},
        ],
        temperature=0.,
        max_tokens=512,
        top_p = 1.,
        frequency_penalty=0,
        presence_penalty=0,
        # stop=["\n\n"],
    )
    print(f"GPT-4 LLM Response: {bot_1_response.choices[0].message.content}\n")

original_context = "Three letters picked without replacement from qqqkkklkqkkk. Give prob of sequence qql."

# original_context = """
# You are an expert at mathematical reasoning, given a problem statement and a solution by a student, provide feedback on whats wrong with the solution and how it can be corrected. Do not directly provide the correct solution.

# Problem: "Three letters picked without replacement from qqqkkklkqkkk. Give prob of sequence qql."
# Solution:
# To calculate the probability of the sequence "qql" occurring when three letters are picked without replacement from the given string "qqqkkklkqkkk," we can determine the total number of possible outcomes and the number of favorable outcomes.

# The string has a total of 13 letters: qqqkkklkqkkk.

# 1. Determine the total number of ways to pick 3 letters without replacement:
#    \[ \text{Total ways} = \binom{13}{3} \]

#    This is because you are choosing 3 letters out of a set of 13.

#    \[ \binom{n}{r} = \frac{n!}{r!(n-r)!} \]

#    \[ \binom{13}{3} = \frac{13!}{3!(13-3)!} = \frac{13 \times 12 \times 11}{3 \times 2 \times 1} = 286 \]

# 2. Determine the number of ways to pick the sequence "qql":
#    - First, there are 3 'q's in the string. Choose 2 of them: \(\binom{3}{2} = 3\) ways.
#    - Then, there is 1 'l' in the string. Choose 1 of them: \(\binom{1}{1} = 1\) way.

#    \[ \text{Favorable ways} = 3 \times 1 = 3 \]

# 3. Calculate the probability:
#    \[ P(\text{"qql"}) = \frac{\text{Favorable ways}}{\text{Total ways}} = \frac{3}{286} \]

# So, the probability of the sequence "qql" occurring is \( \frac{3}{286} \)."""

feedback_conversation(original_context)


