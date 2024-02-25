import os
import pandas as pd
import csv
import json
from tqdm import tqdm


def create_queries_captions(queries_path, captions_path):
    queries = pd.read_csv(queries_path, index_col=None, keep_default_na=False)
    captions = json.load(open(captions_path))
    image_captions = []
    for index, row in tqdm(queries.iterrows()):
        image_name = row['image_name']
        image_caption = captions[image_name]
        image_captions.append(image_caption)
    queries['image_caption'] = image_captions

    print("Saving queries with captions to {}".format(queries_path.replace('queries', 'queries_caption')))
    queries.to_csv(queries_path.replace('queries', 'queries_captions'), index=False)
    return


create_queries_captions('data/OkVQA/queries.csv', 'data/OkVQA/OkVQA_image_captions.json')
create_queries_captions('data/OkVQA/queries_small.csv', 'data/OkVQA/OkVQA_image_captions.json')



