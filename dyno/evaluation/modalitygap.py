import torch

def reciprocal_gap(text_embeds, audio_embeds):
    # compute the reciprocal gap between text and audio embeddings
    # by computing the magnitude of difference between the centroids of L2-normalized embeddings of both modalities
    normalized_text_embeds = torch.nn.functional.normalize(text_embeds, p=2, dim=-1)
    normalized_audio_embeds = torch.nn.functional.normalize(audio_embeds, p=2, dim=-1)
    text_centroid = normalized_text_embeds.mean(dim=0)
    audio_centroid = normalized_audio_embeds.mean(dim=0)
    return torch.norm(text_centroid - audio_centroid, p=2)

def mutual_gap(embeds):
    # compute the mutual gap between embeddings of a modality by getting 
    # the centroid of L2-normalized embeddings and computing the average difference
    # between each embedding and the centroid
    normalized_embeds = torch.nn.functional.normalize(embeds, p=2, dim=-1)
    centroid = normalized_embeds.mean(dim=0)
    distances = torch.norm(normalized_embeds - centroid, p=2, dim=-1)
    return distances.mean()