# these are the libraries that we need
import pandas as pd
import os
import numpy as np
import torch
from sentence_transformers import SentenceTransformer #install with pip install -U sentence-transformers
import pickle

# load a log example
df = pd.read_excel('SQL_random.xlsx', engine='openpyxl')

#we check if GPU is available, for big dataset is necessary for embedding
if torch.cuda.is_available():
    device_count = torch.cuda.device_count()
    print(f"Number of GPUs available: {device_count}")
    for i in range(device_count):
        device = torch.device(f"cuda:{i}")
        print(f"Device {i}: {torch.cuda.get_device_name(device)}")
else:
    device = torch.device("cpu")
    print("CUDA is not available")


# we load a little, fast but accurate model, more info on www.sbert.net
modelST = 'all-MiniLM-L6-v2'
model = SentenceTransformer(modelST, device=device) 

# the function to create the embedding
def embeddings(text):
    return model.encode(text, device = device, batch_size=64, normalize_embeddings=True) 
#normalize_embeddings – If set to true, returned vectors will have length 1. In that case, the faster dot-product (util.dot_score) instead of cosine similarity can be used.

df['SQL_Embedded'] = df['SQL'].apply(embeddings)

# we can store the embedding to disk so we don't have to recompute every time
with open('embeddings_test.pkl', 'wb') as fOut:
    pickle.dump(df['SQL_Embedded'], fOut, protocol=pickle.HIGHEST_PROTOCOL)

"""
#and if needed we can load it
#Load sentences & embeddings from disc
with open('embeddings_test.pkl', "rb") as fIn:
    stored_data = pickle.load(fIn)
    df['SQL_Embedded'] = stored_data['embeddings_test']
"""

# now we calculate the average value, we use a library that use multithread, again, is usefull if our dataset is big
import concurrent.futures

def calculate_mean_embedding(df):
    return np.mean(df['SQL_Embedded'])

with concurrent.futures.ThreadPoolExecutor() as executor:
    future = executor.submit(calculate_mean_embedding, df)
    embedding_avg = future.result()

print('the average value of the embedded strings is:', embedding_avg)

# now that we have the 'average meaning encoded' we check the similarity of all the others queries

# Calculate the anomaly score for each SQL string related to embedding_avg
print('Computing anomalies...')
scores = np.dot(df['SQL_Embedded'].tolist(), embedding_avg)

# to make easier to read, we convert the value to integers from 0 to 10
scores_abs = [int(score*10) for score in scores]

# we add the value to the dataset
df['Dataset_Anomaly'] = scores_abs

#we print some brief report
print('Number of Dataset_Anomaly 0: ', len(df[df['Dataset_Anomaly'] == 0]))
print('Number of Dataset_Anomaly 1: ', len(df[df['Dataset_Anomaly'] == 1]))
print('Number of Dataset_Anomaly 2: ', len(df[df['Dataset_Anomaly'] == 2]))
print('Number of Dataset_Anomaly 3: ', len(df[df['Dataset_Anomaly'] == 3]))

#so we have now for every line a score of anomaly, the most interesting are low value
df

#we can save it as a new file, then analyze with Excel
df.to_excel('results.xlsx', index=False, engine='openpyxl')