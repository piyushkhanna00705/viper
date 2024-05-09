from image_patch import ImagePatch, llm_query, best_image_match, distance, bool_to_yesno, select_answer
from PIL import Image

from llava_image_caption import run_llava_captions

import torchvision.transforms as T
from torchvision.utils import save_image



def get_image(image_path):
    with open(image_path, "rb") as f:
        pil_image = Image.open(f).convert("RGB")
        image = pil_image
        return image


num_patches_widths = [4, 2, 1]
blip_ques = "Write a detailed yet concise description of this image, in under 100 words."

transform = T.ToPILImage()


# def execute_command_29509(image, possible_answers, query, ImagePatch, llm_query, bool_to_yesno, distance, best_image_match, select_answer):
#     # Answer is:def execute_command(image, possible_answers, query)->str:
#     image_patch = ImagePatch(image)
#     print("image_patch: ", image_patch)


    # search_object = """little girl"""
    # # image_patch.simple_query("What is this?")
    # for num_patches_width in num_patches_widths:
    #     print("Num Patches Width: ", num_patches_width)
    #     intermediate_search_patch, i, j = image_patch.visual_search(num_patches_width, blip_ques = blip_ques, query = query, search_object = search_object)
    #     if not intermediate_search_patch:
    #         print("Intermediate Search Patch is None. No relevant image patch found! Moving to smaller patches...")
    #         continue
    #     print("Intermediate Search Patch: ", intermediate_search_patch)
    #     print("Intermediate Search Patch Search Object Exist? (BLIP Exists?): ", intermediate_search_patch.exists(search_object))
    #     # if intermediate_search_patch.exists(search_object):
    #     print("Intermediate Search Patch Search Object Exist? (Llava Prompting): ")
    #     print(run_llava_captions(transform(intermediate_search_patch.cropped_image), prompt = "Does the image contain a little girl? Respond with only Yes or No"))
    #     # if run_llava_captions(transform(intermediate_search_patch.cropped_image), prompt = "Does the image contain a little girl? Respond with only Yes or No") == "Yes":
    #     print(f"Saving Intermediate Search Patch Image {i}{j}")
    #     save_image(intermediate_search_patch.cropped_image, 'sample_patch_intermediate_num_patches_width_sa_17_' + str(num_patches_width) + '.png')
    #     if intermediate_search_patch.exists(search_object):
    #         answer_blip = intermediate_search_patch.simple_query(query)
    #         print("Final Answer Blip: ", answer_blip)
    #         answer_llava = run_llava_captions(transform(intermediate_search_patch.cropped_image), prompt = query)
    #         print(f"Final Answer Llava with query as {query}: ", answer_llava)

    #         llava_mcq_prompt = f"""
    #         {query}
    #         A. {possible_answers[0]}
    #         B. {possible_answers[1]}
    #         C. {possible_answers[2]}
    #         D. {possible_answers[3]}
    #         """

    #         print("Llava MCQ Prompt: ", llava_mcq_prompt)

    #         answer_llava_mcq = run_llava_captions(transform(intermediate_search_patch.cropped_image), prompt = llava_mcq_prompt)
    #         print(f"Final Answer Llava MCQ with query as {query}: ", answer_llava_mcq)
    #         break

    

possible_answers = [
        "The color of the little girl's shirt is pink.",
        "The color of the little girl's shirt is white.",
        "The color of the little girl's shirt is yellow.",
        "The color of the little girl's shirt is black."
    ]

query = "What is the color of the little girl's shirt?"








num_patches_widths = [4, 2, 1]
transform = T.ToPILImage()
def execute_command_29509(image, possible_answers, query, ImagePatch, llm_query, bool_to_yesno, distance, best_image_match, select_answer):
    # image_patch = ImagePatch(image)
    # print("image_patch: ", image_patch)
    # search_object = "little girl"
    # print("Visual Search Final Answer: ", image_patch.visual_search(search_object, query, possible_answers))

    image_patch = ImagePatch(image)


    info = None
    answer_info_none = select_answer(info, query, possible_answers)
    print(answer_info_none)
    return

    flag_patches = image_patch.find("""little girl""")


    # if flag_patches:
    #     flag_patch = flag_patches[0]
    #     flag_patch_colors = flag_patch.simple_query(query)
    #     info = {"""Color of little girl's""": flag_patch_colors}
    #     answer = select_answer(info, query, possible_answers)
    #     return answer
    # else:
    # print("Llava object exsists answer: ", image_patch.object_exists("little girl"))

    print(run_llava_captions(transform(image_patch.cropped_image)))
    
    lock_likely_loc = run_llava_captions(transform(image_patch.cropped_image), prompt = "What is the most likely location of the lock?")

    print("lock_likely_loc: ", lock_likely_loc)

    print(run_llava_captions(transform(image_patch.cropped_image), prompt = "Provide the bounding box coordinate of the region this sentence describes: lock"))

    bounding_box_str = run_llava_captions(transform(image_patch.cropped_image), prompt = "Provide the bounding box coordinate of the region this sentence describes: lock")
    values = bounding_box_str.strip("[]").split(",")
    x1 = min(float(values[0]), float(values[2]))
    y1 = max(float(values[1]), float(values[3]))
    x2 = max(float(values[0]), float(values[2]))
    y2 = min(float(values[1]), float(values[3]))
    height = image_patch.height
    width = image_patch.width
    cropped_img = image_patch.crop(x1 * width, height - y1 * height, x2 * width, height - y2 * height)

    save_image(cropped_img.cropped_image, 'sample_patch_intermediate_sa_70112_cropped_llava.png')

    search_object = """little girl"""


    # visual_search_ans = image_patch.visual_search(search_object, query, possible_answers)
    # if visual_search_ans:
    #     return visual_search_ans
        # return """No little girl found in the image even after visual search"""


possible_answers = ["The color of the little girl's shirt is pink.", 
                    "The color of the little girl's shirt is white.",
                    "The color of the little girl's shirt is yellow.", 
                    "The color of the little girl's shirt is black."]

query = "What is the color of the lock?"

image_path = "/data/tir/projects/tir6/general/piyushkh/viper/data/V_Star/images/direct_attributes/sa_70112.jpg"

# image_path = "/data/tir/projects/tir6/general/piyushkh/viper/sa_70112_cropped.jpg"

pil_image = get_image(image_path)

print(execute_command_29509(pil_image, possible_answers, query, ImagePatch, llm_query, bool_to_yesno, distance, best_image_match, select_answer))
