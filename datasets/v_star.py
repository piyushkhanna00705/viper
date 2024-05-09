from pathlib import Path

import decord
from decord import cpu, gpu
import numpy as np
from PIL import Image
import pandas as pd
from torch.utils.data import Dataset
import base64

from datasets import accuracy as general_accuracy


class V_Star(Dataset):
    def __init__(self, split, data_path="", input_type='image', image_transforms=None, fps=30, max_num_frames=30,
                 max_samples=None, start_sample=0, dataset_size='full', task_type = "direct_attributes", captions=False, **kwargs):
        """
        Args:
            split (str): Data split.
            data_path (str): Path to the data folder
            input_type (str): Type of input. One of ["image", "video"]
            image_transforms (callable, optional): Optional transform to be applied on an image. Only used if input_type
                is "image".
            fps (int): Frames per second. Only used if input_type is "video".
            max_num_frames (int): Maximum number of frames to use. Only used if input_type is "video".
            max_samples (int, optional): Maximum number of samples to load. If None, load all samples.
            start_sample (int, optional): Index of the first sample to load. If None, start from the beginning.
        """

        self.split = split
        self.data_path = Path(data_path)
        self.input_type = input_type
        self.image_transforms = image_transforms
        self.fps = fps
        self.max_num_frames = max_num_frames
        self.task_type = task_type

        # Load questions, answers, and image ids
        if dataset_size == 'full':
            file_name = 'queries.csv'
        elif dataset_size == 'small':
            file_name = 'queries_small.csv'
        elif dataset_size == 'selected':
            file_name = 'queries_selected.csv'

        if captions:
            file_name = file_name.replace('queries', 'queries_captions')
        

        print("Loading {}".format(file_name))

        print("Reading csv from: ", self.data_path / self.task_type / file_name)

        with open(self.data_path / self.task_type / file_name, 'r') as f:
            # The csv has the rows [query, answer, image_name or video_name]
            self.df = pd.read_csv(f, index_col=None, keep_default_na=False)

        if max_samples is not None:
            self.df = self.df.iloc[start_sample:start_sample + max_samples]

        self.n_samples = len(self.df)

        #append options to df['query']
        # self.df['query'] = self.df['query'] + ' Options: ' + self.df['possible_answers'].apply(lambda x: ' '.join(x.split(',')))

    #Make change here for processing images.
    def get_sample_path(self, index):
        sample_name = self.df.iloc[index][f"{self.input_type}_name"]
        #                 data/V_Star /       images/          direct_attributes/ sa_70112.jpg
        sample_path = self.data_path / f"{self.input_type}s" / self.task_type / sample_name
        print("Sample path: ", sample_path)
        return sample_path

    def get_image(self, image_path):
        with open(image_path, "rb") as f:
            pil_image = Image.open(f).convert("RGB")
        if self.image_transforms:
            image = self.image_transforms(pil_image)[:3]
        else:
            image = pil_image

        # print("Saving image")
        # #get last part of path after _ and save as jpg
        # image_path_temp = str(image_path)
        # image_path_temp = image_path_temp.split('_')[-1].split('.')[0]
        # save_path= "temp_img/test_{}.jpeg".format(image_path_temp)
        # print("Saving image to {}".format(save_path))
        # image.save(save_path)
        return image

    def get_image_base64(self, image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    def get_video(self, video_path):
        # If fixed width and height are required, VideoReader takes width and height as arguments.
        video_reader = decord.VideoReader(str(video_path), num_threads=1, ctx=cpu(0))
        decord.bridge.set_bridge('torch')
        vlen = len(video_reader)
        original_fps = video_reader.get_avg_fps()
        num_frames = int(vlen * self.fps / original_fps)
        num_frames = min(self.max_num_frames, num_frames)
        frame_idxs = np.linspace(0, vlen, num_frames, endpoint=False).astype(np.int)
        video = video_reader.get_batch(frame_idxs).byte()
        video = video.permute(0, 3, 1, 2)
        return video

    def __getitem__(self, index):

        out_dict = self.df.iloc[index].to_dict()

        sample_path = self.get_sample_path(index)

        # Load and transform image
        image = self.get_image(sample_path) if self.input_type == "image" else self.get_video(sample_path)

        image_base64 = self.get_image_base64(sample_path)

        out_dict["image"] = image
        out_dict["index"] = index
        out_dict["image_base64"] = image_base64

        # if 'extra_context' not in out_dict:
        out_dict['extra_context'] = out_dict['possible_answers']

        return out_dict

    def __len__(self):
        return self.n_samples

    #Prediction: all_results
    #Ground Truth: all_answers
    @classmethod
    def accuracy(cls, all_results, all_answers, all_possible_answers, all_query_types):
        # print("INSIDE V_STAR ACCURACY METRIC")
        score = 0
        for pred, gt in zip(all_results, all_answers):
            # print("Pred: ", pred)
            # print("GT: ", gt)
            if pred == gt:
                score += 1
        return score / len(all_results)


        
