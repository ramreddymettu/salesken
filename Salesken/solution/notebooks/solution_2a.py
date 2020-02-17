#!/usr/bin/env python
# coding: utf-8

# # Problem 2
# - Find the Semantic Similarity.
# - Given a list of sentences (list_of_setences.txt) write an algorithm which computes the semantic similarity and return the similar sentences together.
# 
# # Algorithm
# - Load the sentences.
# - To calculate sentence similarities, we are using **Spacy Embedding Vectors**.(Can use advanced embeddings such as Universal Senctence Encoder emddings, but they are computationally expensive).
#   - Embedding Vectors are learned(Symantic and Syntactic) weights for each word on huge corpora.
# - For a given sentence, load the each embedding vector corresponding to a word in sentence.
# - Average every word embedding embedding vector to get the sendtence embedding.
# - Find the **Cosine** similarities between each sentence for finding **Semantic Similarity**.
# - Decide a **threshold(>0.75 here)** for semantic similarity to find similar sentences.
# - After deciding threshold, we iterate over each sentence embedding to calculate the similarity. Get group of Index with > threshold value. 
# - With list list of indices in hand, get respective senteses list and return them.
# 

# In[25]:


import spacy
import numpy as np


# In[26]:


nlp = spacy.load("en_core_web_md")


# In[27]:


def similatity_matrix(sent_doc):
    sim_mat = np.zeros((len(sent_doc), len(sent_doc)))
    for i, row in enumerate(sent_doc):
        for j, col in enumerate(sent_doc):
            #print(row, col)
            sim_val = row.similarity(col)
            sim_mat[i][j] = sim_val
    return sim_mat


# In[28]:


def similar_sent_idx(sim_matrix, threshold = 0.78):
    idx = np.arange(len(sim_matrix))
    seen_idx = []
    sim_idx = [] 
    for i in range(sim_matrix.shape[0]):
        if i not in seen_idx:
            tmp = [t+i+1 for t in list(np.where( sim_matrix[i][i+1:] > threshold )[0])]
            seen_idx.extend(tmp)
            tmp.append(i)
            sim_idx.append(tmp)
    return sim_idx


# In[29]:


def similar_sentences(sent_data):
    sent_array = np.array(sent_data)
    sent_doc = list(map(nlp, sent_data))
    sim_mat = similatity_matrix(sent_doc)
    sim_sent_idx = similar_sent_idx(sim_mat)
    similar_sentences = [sent_array[idx].tolist() for idx in sim_sent_idx]
    return similar_sentences


# In[24]:


if __name__ == "__main__":
    with open("../../problem/list_of_sentences", 'r') as txt:
        sent_data = txt.read().strip().split("\n")
    from pprint import pprint
    #pprint(similar_sentences(sent_data))
    
   
    
    pprint(similar_sentences(["Football is played in Brazil" ,
                              "Cricket is played in India",
                              "Traveling is good for health",
                              "People love traveling in winter"]))


# In[ ]:




