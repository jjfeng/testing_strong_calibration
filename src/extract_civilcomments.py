import pandas as pd
import numpy as np

from wilds import get_dataset
from wilds.common.data_loaders import get_train_loader

from sentence_transformers import SentenceTransformer

IDENTITY_VARS = [
    'male',
    'female',
    'transgender',
    'other_gender',
    'heterosexual',
    'homosexual_gay_or_lesbian',
    'bisexual',
    'other_sexual_orientation',
    'christian',
    'jewish',
    'muslim',
    'hindu',
    'buddhist',
    'atheist',
    'other_religion',
    'black',
    'white',
    'asian',
    'latino',
    'other_race_or_ethnicity',
    'physical_disability',
    'intellectual_or_learning_disability',
    'psychiatric_or_mental_illness',
    'other_disability',
]

np.random.seed(1)
model = SentenceTransformer('all-MiniLM-L6-v2')
dataset = pd.read_csv("data/civilcomments_v1.0/all_data_with_identities.csv", index_col=0)
dataset = dataset.iloc[np.random.choice(dataset.shape[0], dataset.shape[0], replace=False)].reset_index(drop=True)
print(dataset)
batch_size = 20000
prev_df = None
for i in range(dataset.shape[0]//batch_size):
    start_idx = batch_size * i
    end_idx = batch_size * (i + 1)
    batch_df = dataset.iloc[start_idx:end_idx].reset_index(drop=True)
    print("IDX", i, start_idx, end_idx)
    print("batch_df", batch_df)
    embeddings = model.encode(batch_df.comment_text)
    metadata = batch_df[IDENTITY_VARS]
    y = batch_df.toxicity >= 0.5
    curr_df = pd.concat([
        metadata,
        pd.DataFrame(embeddings, columns=[f"embedding{idx}" for idx in range(embeddings.shape[1])]),
        y
    ], axis=1)
    curr_df.columns = [f"demographic_{var_name}" for var_name in IDENTITY_VARS] + [f"embedding{idx}" for idx in range(embeddings.shape[1])] + ["y"]
    print(curr_df)
    
    # if prev_df is not None:
    #     prev_df = pd.concat([prev_df, curr_df])
    # else:
    #     prev_df = curr_df

    # convert array into dataframe
    curr_df.to_csv(f"data/civil_comments_{i}.csv", index=False)
