#TODO:
#1. load the embedding 
#2. create index
#3. Train
#4. add everything
#5. save the index

import faiss, os, torch
from tqdm import tqdm

EMBEDDINGS_DIR = "./embeddings"

# 1. Load the embeddings and indices
paths = []
embeddings = []
num_items = 0

all_files = os.listdir(EMBEDDINGS_DIR)

for embedding_file in tqdm(all_files):
    stored_embedding = torch.load(f"{EMBEDDINGS_DIR}/{embedding_file}")

    embeddings.append(stored_embedding["embeddings"])
    paths = paths + [p.removeprefix("data/") for p in stored_embedding["paths"]]

embeddings = torch.cat(embeddings).numpy()
faiss.normalize_L2(embeddings)

# 2. Create the index
d = embeddings.shape[1]

quantizer = faiss.IndexHNSWFlat(d, 32, faiss.METRIC_INNER_PRODUCT)
index = faiss.IndexIVFFlat(quantizer, d, 1024, faiss.METRIC_INNER_PRODUCT)

# 3. Train
index.train(embeddings)

# 4. Add everything
index.add(embeddings)

# 5. Save index and paths to a file
faiss.write_index(index, "index.bin")
torch.save(paths, "paths.bin")
