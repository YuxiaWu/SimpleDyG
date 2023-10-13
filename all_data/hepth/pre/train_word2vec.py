import gensim
from gensim.models import Word2Vec
import numpy as np
import nltk
nltk.download('punkt')  
import os

# Define the root directory
root_directory = "hep-th-abs"
# Define the file extension you want to read
file_extension = ".abs"
feat_dim = 172

# Recursive function to read data from ".abs" files in "yyy" subdirectories
def read_abs_files(directory):
    documents = []
    paper_ids = []
    for root, dirs, files in os.walk(directory):
        for file_name in files:
            # Check if the file has the desired extension
            if file_name.endswith(file_extension):
                # Construct the full file path
                file_path = os.path.join(root, file_name)
                
                # Open and read the ".abs" file
                with open(file_path, 'r') as file:
                    content = file.read()
                    documents.append(content.split('\\\\')[-2])
                    paper_ids.append(content.split('\\\\')[1].split('\n')[1].split('/')[-1])
                   
    return documents, paper_ids

# Start reading ".abs" files from the root directory
documents, paper_ids = read_abs_files(root_directory)

tokenized_documents = [nltk.word_tokenize(doc.lower()) for doc in documents]

model = Word2Vec(sentences=tokenized_documents, vector_size=feat_dim, window=5, min_count=1, sg=0)

document_features = {}

for idx, doc_tokens in enumerate(tokenized_documents):
    doc_vector = np.mean([model.wv[token] for token in doc_tokens if token in model.wv], axis=0)
    document_features[paper_ids[idx]] = doc_vector

# save the document_features dic 
import pickle
with open('document_features.pkl', 'wb') as f:
    pickle.dump(document_features, f)
