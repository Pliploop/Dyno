import scipy.linalg as linalg
import numpy as np
from dataset_statistics import OnlineStats
from torchaudio.functional import frechet_distance as fd_
from rich.pretty import pprint


def frechet_distance(ground_online_stats, generated_online_stats):
    
    
    if isinstance(ground_online_stats, OnlineStats):
        mean_ground = ground_online_stats.mean
        mean_generated = generated_online_stats.mean
    elif isinstance(ground_online_stats, dict):
        mean_ground = ground_online_stats['mean']
        mean_generated = generated_online_stats['mean']   
    else:
        raise ValueError('ground_online_stats and generated_online_stats should be either of type OnlineStats or dict') 
    
    if isinstance(ground_online_stats, OnlineStats):
        cov_ground = ground_online_stats.covar_matrix
        cov_generated = generated_online_stats.covar_matrix
    elif isinstance(ground_online_stats, dict):
        cov_ground = ground_online_stats['covar_matrix']
        cov_generated = generated_online_stats['covar_matrix']
        
    mean_ground = torch.from_numpy(mean_ground) if not isinstance(mean_ground, torch.Tensor) else mean_ground
    mean_generated = torch.from_numpy(mean_generated) if not isinstance(mean_generated, torch.Tensor) else mean_generated
    cov_ground = torch.from_numpy(cov_ground) if not isinstance(cov_ground, torch.Tensor) else cov_ground
    cov_generated = torch.from_numpy(cov_generated) if not isinstance(cov_generated, torch.Tensor) else cov_generated
    
    #all to float32
    mean_ground = mean_ground.float()
    mean_generated = mean_generated.float()
    cov_ground = cov_ground.float()
    cov_generated = cov_generated.float()
    
    return fd_(mean_ground, cov_ground, mean_generated, cov_generated).item()



import torch
import numpy as np
import itertools
from tqdm import tqdm
import json


def generate_from_prompts(model, prompts, **kwargs):

    preds = model.inference(prompts, model.inference_scheduler, **kwargs) if hasattr(model, 'inference_scheduler') else model.inference(prompts, **kwargs)

    return preds


def get_embeddings_and_preds(model, datum,preextracted_features=True, **kwargs):
    

    prompt = datum['prompt']

    prompt = [prompt] if isinstance(prompt, str) else prompt

    audio = datum.get('audio', None)

    text_embeds = model.encoder_pair.get_text_embedding(prompt)
    # print(text_embeds)
    audio = audio if preextracted_features else model.encoder_pair.get_audio_embedding_from_data(audio)

    audio = torch.stack(audio) if isinstance(audio, list) else audio

    preds = generate_from_prompts(
        model, prompts=prompt, **kwargs)
    
    # invert the last two dimensions of the preds tensor no matter the shape
    
    preds = preds.transpose(-1, -2)
    

    return {
        'text_embeds': text_embeds,
        'audio_embeds': audio,
        'preds': preds
    }

    
    


def pred_dataset(model, dataset,limit_n=-1, preextracted_features=True, **kwargs):

    model.eval()

    file_idx = []

    audio_embeds, text_embeds, preds = [], [], []
    
    len_dataset = len(dataset)
    
    pbar = tqdm(itertools.islice(enumerate(dataset), 0, limit_n), total=limit_n if limit_n > 0 else len_dataset)

    for i, datum in pbar:

        embeddings_and_preds = get_embeddings_and_preds(
            model, datum, preextracted_features, **kwargs)
        
        audio_embeds.append(embeddings_and_preds['audio_embeds']) if isinstance(
            embeddings_and_preds['audio_embeds'], torch.Tensor) else embeddings_and_preds['audio_embeds']['embedding_proj']
        
        text_embeds.append(
            embeddings_and_preds['text_embeds'].get('projected_pooler_output',torch.zeros(1,1)))
        
        preds.append(embeddings_and_preds['preds'])
        
        msg = f'text_embeds: {text_embeds[-1].shape}' + ' ' + f'audio_embeds: {audio_embeds[-1].shape}' + ' ' + f'preds: {preds[-1].shape}'
        # update the tqdm bar with the message
        pprint(msg) if i==0 else None
        
        file_idx.append(datum['file_idx'])
        
    file_idx = [[i for i, x in enumerate(file_idx) if x == idx] for idx in file_idx]

    audio_embeds = [embed.detach().cpu() for embed in audio_embeds]
    text_embeds = [embed.detach().cpu() for embed in text_embeds]
    preds = [pred.detach().cpu() for pred in preds]

    return {
        'audio_embeds': audio_embeds,
        'text_embeds': text_embeds,
        'preds': preds,
        'file_idx': file_idx
    }
