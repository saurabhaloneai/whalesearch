import os
import time

import faiss
import gradio as gr
import open_clip
import torch
from lightning_cloud.utils.data_connection import add_s3_connection

RAW_DATA_DIR = "/teamspace/s3_connections/imagenet-1m-template/raw/"

# 1. Add the raw dataset to your teamspace

# TODO:
# 1. get the local data to seach thorugh it
# 2. add the data into s3
# 3. try to get online data don't know how but might work

# 2. Load the model
device = "cuda" if torch.cuda.is_available() else "cpu"

model, _, _ = open_clip.create_model_and_transforms(
    "ViT-B-32", pretrained="laion2b_s34b_b79k"
)
model.to(device)
tokenizer = open_clip.get_tokenizer("ViT-B-32")

# 3. Load the index
index = faiss.read_index("index.bin")
paths = torch.load("paths.bin")


# 4. Define the search function
def search_images(query):
    with torch.inference_mode():
        text = tokenizer([query])
        text_embedding = model.encode_text(text.to(device)).cpu().numpy()
        faiss.normalize_L2(text_embedding)

    _, indexes = index.search(x=text_embedding, k=12)

    return [os.path.join(RAW_DATA_DIR, paths[idx]) for idx in indexes[0]]
