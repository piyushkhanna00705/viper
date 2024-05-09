
# Importing necessary libraries
from transformers import AutoTokenizer, BitsAndBytesConfig
import sys
import os

from PIL import Image

os.environ['HF_HOME'] = '/data/tir/projects/tir6/general/piyushkh/hf/'
sys.path.append('LLaVA')

from llava.model import LlavaLlamaForCausalLM
import torch

from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path
from llava.eval.run_llava_captions import eval_model
from llava.eval.run_llava_captions import run_llava

import requests
from io import BytesIO


model_path = "liuhaotian/llava-v1.5-7b"
# model_path = "liuhaotian/llava-v1.5-13b-lora"
# model_path = "liuhaotian/llava-v1.5-13b"

tokenizer, model, image_processor, context_len = load_pretrained_model(
    model_path=model_path,
    model_base=None,
    # model_base="lmsys/vicuna-13b-v1.5",
    model_name=get_model_name_from_path(model_path)
)

# prompt = "Is there a cyclist in the image, if yes what is the color of the cyclist's bag?, answer in under 100 words."





def load_image(image_file):
    if image_file.startswith("http") or image_file.startswith("https"):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert("RGB")
    else:
        image = Image.open(image_file).convert("RGB")
    return image




def run_llava_captions(image, prompt = "Write a detailed yet concise description of this image, in under 100 words."):

    args = type('Args', (), {
        "generation_type": "captions",
        # "images_dirpath": "/data/tir/projects/tir6/general/piyushkh/viper/sample_images_llava/",
        # "output_filepath": "/data/tir/projects/tir6/general/piyushkh/viper/sample_images_llava/llava_image_captions.json",
        "model_path": model_path,
        # "model_base": "lmsys/vicuna-13b-v1.5",
        "model_base": None,
        "model_name": get_model_name_from_path(model_path),
        "query": prompt,
        "conv_mode": None,
        "temperature": 0.2,
        "top_p": None,
        "num_beams": 1,
        "max_new_tokens": 256,
        "sep": ",",
    })()


    output = run_llava(args, args.model_name, tokenizer, model, image_processor, context_len, image)
    return output

    # for i in range(2):
    #     for j in range(3):
    #         # print("Generating caption for image patch {}_{}".format(i, j))
    #         image_filepath = "/data/tir/projects/tir6/general/piyushkh/viper/sample_patch{}{}.png".format(i, j)
    #         image = load_image(image_filepath)
    #         outputs = run_llava(args, args.model_name, tokenizer, model, image_processor, context_len, image)
    #         print(f"Caption {i}{j}: " + outputs)



# run_llava_captions()