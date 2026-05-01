import torch
from vendi_score import vendi
from prdc import compute_prdc
import numpy as np

def mics(list_of_clusters):
    
    # clusters are of shape (N_gen, timesteps, embedding_dim)
    # we want to compute the mean intra cluster similarity (MICS) for each cluster
    
    mics_ = []
    
    for e_ in list_of_clusters:
        avg_pooled = torch.mean(e_, dim=1)
        avg_pooled = torch.nn.functional.normalize(avg_pooled, p=2, dim=-1)
        mics__ = avg_pooled @ avg_pooled.t()
        # do not count the diagonal
        mics__ = mics__ - torch.eye(mics__.shape[0]).to(mics__.device)
        count_ = mics__.shape[0]**2 - mics__.shape[0]
        mics_.append(mics__.sum() / count_)
        
    return torch.stack(mics_).mean()


def mivs(list_of_clusters):
    # same thing with vendi score 
    mivs_ = []
    
    for e_ in list_of_clusters:
        avg_pooled = torch.mean(e_, dim=1)
        avg_pooled = torch.nn.functional.normalize(avg_pooled, p=2, dim=-1)
        mivs__ = avg_pooled @ avg_pooled.t()
        mivs_.append(vendi.score_K(mivs__))
        
        
    return np.mean(mivs_)

def vendi_score(embeddings):
    
    #get the cosine similarity matrix
    embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=-1)
    sim_matrix = embeddings @ embeddings.t()
    vendi_score = vendi.score_K(sim_matrix)
    
    return vendi_score

def density_coverage_precision_recall(grounding_embeddings, generated_embeddings, nearest_k = 5):
    
    return compute_prdc(real_features = grounding_embeddings, fake_features = generated_embeddings, nearest_k = nearest_k)
    
    
    