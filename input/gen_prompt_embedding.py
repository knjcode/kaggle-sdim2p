import sys
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

target_file = sys.argv[1]

st_model = SentenceTransformer('all-MiniLM-L6-v2')

df = pd.read_table(target_file, low_memory=False)

full_prompts = df.prompt.to_list()

full_emb = st_model.encode(full_prompts)
print(f"full_emb.shape: {full_emb.shape}")
save_path = target_file.replace('.tsv', '.npy')
np.save(save_path, full_emb)
print(f"saved: {save_path}")

