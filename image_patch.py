from __future__ import annotations

import numpy as np
import re
import torch
from dateutil import parser as dateparser
from PIL import Image
from rich.console import Console
from torchvision import transforms
from torchvision.ops import box_iou
from typing import Union, List
from word2number import w2n
from torchvision.utils import save_image
import torchvision
import torchvision.transforms as T
import math


from llama_index.core.multi_modal_llms.generic_utils import load_image_urls
from llama_index.core.schema import ImageDocument

import torch
from torchmetrics.multimodal.clip_score import CLIPScore



from llava_image_caption import run_llava_captions

# from torchmetrics.multimodal.clip_score import CLIPScore

import os

os.environ['HF_HOME'] = '/data/tir/projects/tir6/general/piyushkh/hf/'


from utils import show_single_image, load_json
from vision_processes import forward, config

console = Console(highlight=False)


class ImagePatch:
    """A Python class containing a crop of an image centered around a particular object, as well as relevant
    information.
    Attributes
    ----------
    cropped_image : array_like
        An array-like of the cropped image taken from the original image.
    left : int
        An int describing the position of the left border of the crop's bounding box in the original image.
    lower : int
        An int describing the position of the bottom border of the crop's bounding box in the original image.
    right : int
        An int describing the position of the right border of the crop's bounding box in the original image.
    upper : int
        An int describing the position of the top border of the crop's bounding box in the original image.

    Methods
    -------
    find(object_name: str)->List[ImagePatch]
        Returns a list of new ImagePatch objects containing crops of the image centered around any objects found in the
        image matching the object_name.
    exists(object_name: str)->bool
        Returns True if the object specified by object_name is found in the image, and False otherwise.
    verify_property(property: str)->bool
        Returns True if the property is met, and False otherwise.
    best_text_match(option_list: List[str], prefix: str)->str
        Returns the string that best matches the image.
    simple_query(question: str=None)->str
        Returns the answer to a basic question asked about the image. If no question is provided, returns the answer
        to "What is this?".
    compute_depth()->float
        Returns the median depth of the image crop.
    crop(left: int, lower: int, right: int, upper: int)->ImagePatch
        Returns a new ImagePatch object containing a crop of the image at the given coordinates.
    """

    def __init__(self, image: Union[Image.Image, torch.Tensor, np.ndarray], left: int = None, lower: int = None,
                 right: int = None, upper: int = None, parent_left=0, parent_lower=0, queues=None,
                 parent_img_patch=None):
        """Initializes an ImagePatch object by cropping the image at the given coordinates and stores the coordinates as
        attributes. If no coordinates are provided, the image is left unmodified, and the coordinates are set to the
        dimensions of the image.

        Parameters
        -------
        image : array_like
            An array-like of the original image.
        left : int
            An int describing the position of the left border of the crop's bounding box in the original image.
        lower : int
            An int describing the position of the bottom border of the crop's bounding box in the original image.
        right : int
            An int describing the position of the right border of the crop's bounding box in the original image.
        upper : int
            An int describing the position of the top border of the crop's bounding box in the original image.

        """

        if isinstance(image, Image.Image):
            image = transforms.ToTensor()(image)
        elif isinstance(image, np.ndarray):
            image = torch.tensor(image).permute(1, 2, 0)
        elif isinstance(image, torch.Tensor) and image.dtype == torch.uint8:
            image = image / 255

        if left is None and right is None and upper is None and lower is None:
            self.cropped_image = image
            self.left = 0
            self.lower = 0
            self.right = image.shape[2]  # width
            self.upper = image.shape[1]  # height
        else:
            self.cropped_image = image[:, image.shape[1]-upper:image.shape[1]-lower, left:right]
            self.left = left + parent_left
            self.upper = upper + parent_lower
            self.right = right + parent_left
            self.lower = lower + parent_lower

        self.height = self.cropped_image.shape[1]
        self.width = self.cropped_image.shape[2]

        self.transform_to_pil = T.ToPILImage()
        
        # print("Type of self.cropped_image", type(self.cropped_image))
        # print("self.cropped_image: ", self.cropped_image)

        self.cache = {}
        self.queues = (None, None) if queues is None else queues

        self.parent_img_patch = parent_img_patch

        self.horizontal_center = (self.left + self.right) / 2
        self.vertical_center = (self.lower + self.upper) / 2

        if self.cropped_image.shape[1] == 0 or self.cropped_image.shape[2] == 0:
            raise Exception("ImagePatch has no area")

        self.possible_options = load_json('./useful_lists/possible_options.json')

    def forward(self, model_name, *args, **kwargs):
        return forward(model_name, *args, queues=self.queues, **kwargs)

    @property
    def original_image(self):
        if self.parent_img_patch is None:
            return self.cropped_image
        else:
            return self.parent_img_patch.original_image

    def find(self, object_name: str) -> list[ImagePatch]:
        """Returns a list of ImagePatch objects matching object_name contained in the crop if any are found.
        Otherwise, returns an empty list.
        Parameters
        ----------
        object_name : str
            the name of the object to be found

        Returns
        -------
        List[ImagePatch]
            a list of ImagePatch objects matching object_name contained in the crop
        """
        if object_name in ["object", "objects"]:
            all_object_coordinates = self.forward('maskrcnn', self.cropped_image)[0]
        else:

            if object_name == 'person':
                object_name = 'people'  # GLIP does better at people than person

            all_object_coordinates = self.forward('glip', self.cropped_image, object_name)
        if len(all_object_coordinates) == 0:
            return []

        threshold = config.ratio_box_area_to_image_area
        if threshold > 0:
            area_im = self.width * self.height
            all_areas = torch.tensor([(coord[2]-coord[0]) * (coord[3]-coord[1]) / area_im
                                      for coord in all_object_coordinates])
            mask = all_areas > threshold
            # if not mask.any():
            #     mask = all_areas == all_areas.max()  # At least return one element
            all_object_coordinates = all_object_coordinates[mask]


        return [self.crop(*coordinates) for coordinates in all_object_coordinates]

    def exists(self, object_name) -> bool:
        """Returns True if the object specified by object_name is found in the image, and False otherwise.
        Parameters
        -------
        object_name : str
            A string describing the name of the object to be found in the image.
        """
        if object_name.isdigit() or object_name.lower().startswith("number"):
            object_name = object_name.lower().replace("number", "").strip()

            object_name = w2n.word_to_num(object_name)
            answer = self.simple_query("What number is written in the image (in digits)?")
            return w2n.word_to_num(answer) == object_name

        patches = self.find(object_name)

        filtered_patches = []
        for patch in patches:
            if "yes" in patch.simple_query(f"Is this a {object_name}?"):
                filtered_patches.append(patch)
        return len(filtered_patches) > 0

    def _score(self, category: str, negative_categories=None, model='clip') -> float:
        """
        Returns a binary score for the similarity between the image and the category.
        The negative categories are used to 
        compare to (score is relative to the scores of the negative categories).
        """
        if model == 'clip':
            res = self.forward('clip', self.cropped_image, category, task='score',
                               negative_categories=negative_categories)
        elif model == 'tcl':
            res = self.forward('tcl', self.cropped_image, category, task='score')
        else:  # xvlm
            task = 'binary_score' if negative_categories is not None else 'score'
            res = self.forward('xvlm', self.cropped_image, category, task=task, negative_categories=negative_categories)
            res = res.item()

        return res
    
    def object_exists(self, object_name: str):
        return run_llava_captions(self.transform_to_pil(self.cropped_image), prompt = f"Does the image contain {object_name}? Respond with only Yes or No") == 'Yes'

    def simple_query_llava(self, query: str):
        return run_llava_captions(self.transform_to_pil(self.cropped_image), prompt = query)

    def visual_search(self, search_object, query, possible_answers):
        num_patches_widths = [4, 2, 1]
        # num_patches_widths = [32]
        blip_ques = "Write a detailed yet concise description of this image, in under 100 words."
        working_image = self
        for num_patches_width in num_patches_widths:
            intermediate_search_patch, caption_num = working_image._visual_search_patch(num_patches_width, blip_ques = blip_ques, query = query, search_object = search_object, overlap_pct = 0)
            if not intermediate_search_patch:
                continue


            working_image = intermediate_search_patch
            
            #Delete this line after debugging
            # save_image(working_image.cropped_image, 'sample_patch_working_image_32patch_sa_17_' + str(num_patches_width) + '.png')
            
            

            # print("Intermediate Search Patch Search Object Exists Blip? ", working_image.exists(search_object))
            # print("Intermediate Search Patch Search Object Exists Llava? ", run_llava_captions(self.transform_to_pil(working_image.cropped_image), prompt = f"Does the image contain {search_object}? Respond with only Yes or No"))            

            if run_llava_captions(self.transform_to_pil(working_image.cropped_image), prompt = f"Does the image contain {search_object}? Respond with only Yes or No") == 'Yes':
                llava_mcq_prompt = f"""
                {query}
                A. {possible_answers[0]}
                B. {possible_answers[1]}
                C. {possible_answers[2]}
                D. {possible_answers[3]}
                """
                answer_llava_mcq = run_llava_captions(self.transform_to_pil(intermediate_search_patch.cropped_image), prompt = llava_mcq_prompt)
                print("Llava answer = ", answer_llava_mcq)
                return answer_llava_mcq
        return None



    
    def _visual_search_patch(self, num_patches_width, blip_ques = None, query = None, search_object = None, overlap_pct = 0):
        prompt = f"""Choose the most appropriate caption that helps you answer the question:
        {query} and explain why you chose it."""
        channels, height, width = self.cropped_image.shape

        num_patches_height = 2  # Divide height into 2 patches

        patch_height = height // num_patches_height
        patch_width = width // num_patches_width

        patches = self.cropped_image.unfold(1, patch_height, math.ceil(patch_height - overlap_pct*patch_height)).unfold(2, patch_width, math.ceil(patch_width - overlap_pct*patch_width))
        patches = patches.contiguous().view(channels, -1, patch_height, patch_width)
        print(patches.shape)
        all_image_patches = []
        map_patch_to_index = {}

        print("patches.shape[1] length: ", patches.shape[1])

        for i in range(patches.shape[1]):
        # for i in range(num_patches_height):
        #     for j in range(num_patches_width):
                # small_patch = patches[:,i,j,:,:]
                small_patch = patches[:,i,:,:]
                # if i==1:
                #     save_image(small_patch, 'sample_patch_contiguous' + str(i) + '.png')
                curr_img_patch = ImagePatch(small_patch, 0, 0, patch_width, patch_height, 0, 0, queues=self.queues, parent_img_patch=self)
                map_patch_to_index[str(i)] = curr_img_patch
                # img_descr_patch = self.forward('blip', all_image_patches[-1].cropped_image, blip_ques, task='qa')
                img_descr_patch = run_llava_captions(self.transform_to_pil(curr_img_patch.cropped_image))

                # print(f"CLIP Score b/w patch {i}{j} and {search_object}: ", get_clip_score(all_image_patches[-1].cropped_image, query))

                prompt = prompt + f"\n Caption {i}: " + img_descr_patch
        
        # print("GPT-3.5 Prompt with captions: ", prompt)
        
        gpt_patch_answer = llm_query(prompt)
        print("GPT-3.5 final answer for choosing correct image crop: ", gpt_patch_answer)

        caption_number = re.search(r'Caption (\d+)', gpt_patch_answer)
        if caption_number:
            print("Found Caption match at: ")
            print(caption_number.group(1))
            # print("Mapped index to patch: ", map_patch_to_index[str(caption_number.group(1))])
            self = map_patch_to_index[str(caption_number.group(1))]
            return self, caption_number.group(1)
        return None, None 

        # pattern = r'Caption (\d)(\d)(\d)'
        # match = re.search(pattern, gpt_patch_answer)
        # print("Caption Match: ", match)
        # if match:
        #     i = int(match.group(1))
        #     j = int(match.group(2))
        #     print("Caption Match found at: ", i, j)
            
        #     self = all_image_patches[map_patch_to_index[(i, j)]]
        #     return self, i, j
        # else:
        #     return None
        #             # save_image(small_patch, 'sample_patch' + str(i) + str(j) + '.png')
        # return


        # metric = CLIPScore(model_name_or_path="openai/clip-vit-base-patch16")

        """If no question is provided, returns the answer to "What is this?"""
        # cropped_image_left_half = self.crop(self.left, self.lower, (self.left + self.right) // 2, self.upper)
        cropped_image_left_half = ImagePatch(self.cropped_image, self.left, self.lower, (self.left + self.right) // 2, self.upper, self.left, self.lower, queues=self.queues,
                          parent_img_patch=self)

        # cropped_image_right_half = self.crop((self.left + self.right) // 2, self.lower, self.right, self.upper)
        cropped_image_right_half = ImagePatch(self.cropped_image, (self.left + self.right) // 2, self.lower, self.right, self.upper, self.left, self.lower, queues=self.queues,
                          parent_img_patch=self)
        save_image(cropped_image_right_half.cropped_image, 'sample_right.png')

        img_descr_left = self.forward('blip', cropped_image_left_half.cropped_image, blip_ques, task='qa')

        # print("cropped_image_left_half.cropped_image", cropped_image_left_half.cropped_image)
        # print("cropped_image_left_half.cropped_image type: ", type(cropped_image_left_half.cropped_image))
        # print("cropped_image_left_half.cropped_image shape: ", type(cropped_image_left_half.cropped_image.shape))

        # clip_score_left = metric(cropped_image_left_half.cropped_image, "trash can", "openai/clip-vit-base-patch16")
        # print("clip_score_left: ", clip_score_left)

        print("img_descr_left = ", img_descr_left)
        save_image(cropped_image_left_half.cropped_image, 'sample_left.png')

        img_descr_right = self.forward('blip', cropped_image_right_half.cropped_image, blip_ques, task='qa')

        # clip_score_right = metric(cropped_image_right_half.cropped_image, "trash can", "openai/clip-vit-base-patch16")
        # print("clip_score_left: ", clip_score_right)

        print("img_descr_right = ", img_descr_right)

        answer = select_answer_image_crop(ques, img_descr_left, img_descr_right)

        print("GPT-3.5 answer for image crop: ", answer)

        if answer == "Left":
            return cropped_image_left_half
        else:
            return cropped_image_right_half

        




    def _detect(self, category: str, thresh, negative_categories=None, model='clip') -> bool:
        return self._score(category, negative_categories, model) > thresh

    def verify_property(self, object_name: str, attribute: str) -> bool:
        """Returns True if the object possesses the property, and False otherwise.
        Differs from 'exists' in that it presupposes the existence of the object specified by object_name, instead
        checking whether the object possesses the property.
        Parameters
        -------
        object_name : str
            A string describing the name of the object to be found in the image.
        attribute : str
            A string describing the property to be checked.
        """
        name = f"{attribute} {object_name}"
        model = config.verify_property.model
        negative_categories = [f"{att} {object_name}" for att in self.possible_options['attributes']]
        if model == 'clip':
            return self._detect(name, negative_categories=negative_categories,
                                thresh=config.verify_property.thresh_clip, model='clip')
        elif model == 'tcl':
            return self._detect(name, thresh=config.verify_property.thresh_tcl, model='tcl')
        else:  # 'xvlm'
            return self._detect(name, negative_categories=negative_categories,
                                thresh=config.verify_property.thresh_xvlm, model='xvlm')

    def best_text_match(self, option_list: list[str] = None, prefix: str = None) -> str:
        """Returns the string that best matches the image.
        Parameters
        -------
        option_list : str
            A list with the names of the different options
        prefix : str
            A string with the prefixes to append to the options
        """
        option_list_to_use = option_list
        if prefix is not None:
            option_list_to_use = [prefix + " " + option for option in option_list]

        model_name = config.best_match_model
        image = self.cropped_image
        text = option_list_to_use
        if model_name in ('clip', 'tcl'):
            selected = self.forward(model_name, image, text, task='classify')
        elif model_name == 'xvlm':
            res = self.forward(model_name, image, text, task='score')
            res = res.argmax().item()
            selected = res
        else:
            raise NotImplementedError

        return option_list[selected]

    def simple_query(self, question: str):
        """Returns the answer to a basic question asked about the image. If no question is provided, returns the answer
        to "What is this?". The questions are about basic perception, and are not meant to be used for complex reasoning
        or external knowledge.
        Parameters
        -------
        question : str
            A string describing the question to be asked.
        """
        print("simple_query question: ", question)
        return self.forward('blip', self.cropped_image, question, task='qa')

    def compute_depth(self):
        """Returns the median depth of the image crop
        Parameters
        ----------
        Returns
        -------
        float
            the median depth of the image crop
        """
        original_image = self.original_image
        depth_map = self.forward('depth', original_image)
        depth_map = depth_map[original_image.shape[1]-self.upper:original_image.shape[1]-self.lower,
                              self.left:self.right]
        return depth_map.median()  # Ideally some kind of mode, but median is good enough for now

    def crop(self, left: int, lower: int, right: int, upper: int) -> ImagePatch:
        """Returns a new ImagePatch containing a crop of the original image at the given coordinates.
        Parameters
        ----------
        left : int
            the position of the left border of the crop's bounding box in the original image
        lower : int
            the position of the bottom border of the crop's bounding box in the original image
        right : int
            the position of the right border of the crop's bounding box in the original image
        upper : int
            the position of the top border of the crop's bounding box in the original image

        Returns
        -------
        ImagePatch
            a new ImagePatch containing a crop of the original image at the given coordinates
        """
        # make all inputs ints
        left = int(left)
        lower = int(lower)
        right = int(right)
        upper = int(upper)

        if config.crop_larger_margin:
            left = max(0, left - 10)
            lower = max(0, lower - 10)
            right = min(self.width, right + 10)
            upper = min(self.height, upper + 10)

        return ImagePatch(self.cropped_image, left, lower, right, upper, self.left, self.lower, queues=self.queues,
                          parent_img_patch=self)

    def overlaps_with(self, left, lower, right, upper):
        """Returns True if a crop with the given coordinates overlaps with this one,
        else False.
        Parameters
        ----------
        left : int
            the left border of the crop to be checked
        lower : int
            the lower border of the crop to be checked
        right : int
            the right border of the crop to be checked
        upper : int
            the upper border of the crop to be checked

        Returns
        -------
        bool
            True if a crop with the given coordinates overlaps with this one, else False
        """
        return self.left <= right and self.right >= left and self.lower <= upper and self.upper >= lower

    def llm_query(self, question: str, long_answer: bool = True) -> str:
        return llm_query(question, None, long_answer)

    def print_image(self, size: tuple[int, int] = None):
        show_single_image(self.cropped_image, size)

    def __repr__(self):
        return "ImagePatch({}, {}, {}, {})".format(self.left, self.lower, self.right, self.upper)


def best_image_match(list_patches: list[ImagePatch], content: List[str], return_index: bool = False) -> \
        Union[ImagePatch, None]:
    """Returns the patch most likely to contain the content.
    Parameters
    ----------
    list_patches : List[ImagePatch]
    content : List[str]
        the object of interest
    return_index : bool
        if True, returns the index of the patch most likely to contain the object

    Returns
    -------
    int
        Patch most likely to contain the object
    """
    if len(list_patches) == 0:
        return None

    model = config.best_match_model

    scores = []
    for cont in content:
        if model == 'clip':
            res = list_patches[0].forward(model, [p.cropped_image for p in list_patches], cont, task='compare',
                                          return_scores=True)
        else:
            res = list_patches[0].forward(model, [p.cropped_image for p in list_patches], cont, task='score')
        scores.append(res)
    scores = torch.stack(scores).mean(dim=0)
    scores = scores.argmax().item()  # Argmax over all image patches

    if return_index:
        return scores
    return list_patches[scores]


def get_clip_score(image: torch.Tensor, caption: str) -> float:
    """Returns the CLIP score for the similarity between the image and the text.
    Parameters
    ----------
    image : torch.Tensor
        the image tensor
    text : str
        the text to compare the image to

    Returns
    -------
    float
        the CLIP score for the similarity between the image and the text
    """
    metric = CLIPScore(model_name_or_path="openai/clip-vit-base-patch16")
    score = metric(image, caption)
    print(score.detach())

    return score.detach().item()


def distance(patch_a: Union[ImagePatch, float], patch_b: Union[ImagePatch, float]) -> float:
    """
    Returns the distance between the edges of two ImagePatches, or between two floats.
    If the patches overlap, it returns a negative distance corresponding to the negative intersection over union.
    """

    if isinstance(patch_a, ImagePatch) and isinstance(patch_b, ImagePatch):
        a_min = np.array([patch_a.left, patch_a.lower])
        a_max = np.array([patch_a.right, patch_a.upper])
        b_min = np.array([patch_b.left, patch_b.lower])
        b_max = np.array([patch_b.right, patch_b.upper])

        u = np.maximum(0, a_min - b_max)
        v = np.maximum(0, b_min - a_max)

        dist = np.sqrt((u ** 2).sum() + (v ** 2).sum())

        if dist == 0:
            box_a = torch.tensor([patch_a.left, patch_a.lower, patch_a.right, patch_a.upper])[None]
            box_b = torch.tensor([patch_b.left, patch_b.lower, patch_b.right, patch_b.upper])[None]
            dist = - box_iou(box_a, box_b).item()

    else:
        dist = abs(patch_a - patch_b)

    return dist


def bool_to_yesno(bool_answer: bool) -> str:
    """Returns a yes/no answer to a question based on the boolean value of bool_answer.
    Parameters
    ----------
    bool_answer : bool
        a boolean value

    Returns
    -------
    str
        a yes/no answer to a question based on the boolean value of bool_answer
    """
    return "yes" if bool_answer else "no"


def llm_query(query, context=None, long_answer=True, queues=None):
    """Answers a text question using GPT-3. The input question is always a formatted string with a variable in it.

    Parameters
    ----------
    query: str
        the text question to ask. Must not contain any reference to 'the image' or 'the photo', etc.
    """
    if long_answer:
        return forward(model_name='gpt3_general', prompt=query, queues=queues)
    else:
        return forward(model_name='gpt3_qa', prompt=[query, context], queues=queues)
    



def select_answer(info = None, question = None, options=None) -> str:
    def format_dict(x):
        if isinstance(x, dict):
            x = ''.join([f'\n\t- {k}: {format_dict(v)}' for k, v in x.items()])
        return x
    with open(config.select_answer_prompt, 'r') as f:
        prompt = f.read()
    if info is not None:
        info_formatting = '\n'.join([f"- {k}: {format_dict(v)}" for k, v in info.items()])
        prompt = prompt.format(info=info_formatting, question=question, options=options)
    else:
        prompt = prompt.format(info="", question=question, options=options)
    # print("MCQ FEW SHOT PROMPT: ", prompt)
    answer = forward(model_name = 'gpt3_general', prompt = prompt)
    answer = answer.strip()
    return answer

def select_answer_image_crop(question: str, left_image_caption, right_image_caption) -> str:
    def format_dict(x):
        if isinstance(x, dict):
            x = ''.join([f'\n\t- {k}: {format_dict(v)}' for k, v in x.items()])
        return x
    with open("/data/tir/projects/tir6/general/piyushkh/viper/prompts/gpt3/crop_image_prompt.txt", 'r') as f:
        prompt = f.read()
    prompt = prompt.format(question=question, left_image_caption=left_image_caption, right_image_caption=right_image_caption)
    # print("MCQ FEW SHOT PROMPT: ", prompt)
    answer = forward(model_name = 'gpt3_general', prompt = prompt)
    answer = answer.strip()
    return answer




def process_guesses(prompt, guess1=None, guess2=None, queues=None):
    return forward(model_name='gpt3_guess', prompt=[prompt, guess1, guess2], queues=queues)


def coerce_to_numeric(string, no_string=False):
    """
    This function takes a string as input and returns a numeric value after removing any non-numeric characters.
    If the input string contains a range (e.g. "10-15"), it returns the first value in the range.
    # TODO: Cases like '25to26' return 2526, which is not correct.
    """
    if any(month in string.lower() for month in ['january', 'february', 'march', 'april', 'may', 'june', 'july',
                                                 'august', 'september', 'october', 'november', 'december']):
        try:
            return dateparser.parse(string).timestamp().year
        except:  # Parse Error
            pass

    try:
        # If it is a word number (e.g. 'zero')
        numeric = w2n.word_to_num(string)
        return numeric
    except ValueError:
        pass

    # Remove any non-numeric characters except the decimal point and the negative sign
    string_re = re.sub("[^0-9\.\-]", "", string)

    if string_re.startswith('-'):
        string_re = '&' + string_re[1:]

    # Check if the string includes a range
    if "-" in string_re:
        # Split the string into parts based on the dash character
        parts = string_re.split("-")
        return coerce_to_numeric(parts[0].replace('&', '-'))
    else:
        string_re = string_re.replace('&', '-')

    try:
        # Convert the string to a float or int depending on whether it has a decimal point
        if "." in string_re:
            numeric = float(string_re)
        else:
            numeric = int(string_re)
    except:
        if no_string:
            raise ValueError
        # No numeric values. Return input
        return string
    return numeric
