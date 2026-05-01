import torch
import numpy as np
import itertools
from tqdm import tqdm
import json
from rich.pretty import pprint


def generate_from_prompts(model, prompts, **kwargs):

    preds = model.inference(prompts, model.inference_scheduler, **kwargs) if hasattr(model, 'inference_scheduler') else model.inference(prompts, **kwargs)

    return preds


# def get_embeddings_and_preds(model, datum,preextracted_features=True, **kwargs):
    

#     prompt = datum['prompt']

#     prompt = [prompt] if isinstance(prompt, str) else prompt

#     audio = datum.get('audio', None)

#     text_embeds = model.encoder_pair.get_text_embedding(prompt)
#     # print(text_embeds)
#     audio = audio if preextracted_features else model.encoder_pair.get_audio_embedding_from_data(audio)

#     audio = torch.stack(audio) if isinstance(audio, list) else audio

#     preds = generate_from_prompts(
#         model, prompts=prompt, **kwargs)
    
#     # invert the last two dimensions of the preds tensor no matter the shape
    
#     preds = preds.transpose(-1, -2)
    
#     # preds = preds.mean(dim=0, keepdim=True)
    
    

#     return {
#         'text_embeds': text_embeds,
#         'audio_embeds': audio,
#         'preds': preds
#     }

def get_embeddings(model, iterator, preextracted_features = True):
    
    text_embeds = []
    audio_embeds = []
    file_idx = []
    
    for batch in tqdm(iterator):
        audio = batch.get('audio', None)
        text = batch.get('prompt', None)
        file_idx_ = batch.get('file_idx', None)
        text = [text] if isinstance(text, str) else text
        audio = audio if preextracted_features else model.encoder_pair.get_audio_embedding_from_data(audio)
        audio = torch.stack(audio) if isinstance(audio, list) else audio
        
        text_embeds_ = model.encoder_pair.get_text_embedding(text)
        
        audio_embeds.append(audio) if isinstance(audio, torch.Tensor) else audio_embeds.append(audio['embedding_proj'])
        text_embeds.append(text_embeds_.get('projected_pooler_output',torch.zeros(1,1)))
        file_idx.append(file_idx_)
        
    return {
        'audio_embeds': audio_embeds,
        'text_embeds': text_embeds,
        'file_idx': file_idx
    }
    
def get_preds(model, datum, **kwargs):

    prompt = datum['prompt']

    prompt = [prompt] if isinstance(prompt, str) else prompt
    preds = generate_from_prompts(
        model, prompts=prompt, **kwargs)
    
    preds = preds.transpose(-1, -2)
    

    return {
        'preds': preds
    }


def compute_distance(x,y,distance = 'cosine', agg = None):
    if x.dim() == 3:
        x_s = []
        for i in range(x.shape[1]):
            x_s.append(compute_distance_(x[:,i,:],y,distance=distance))
        x_s = torch.stack(x_s, dim=1)
        print(f'COMBINED SIMS: {x_s.shape}')
        if distance == 'cosine':
            x_s= x_s.max(dim=1).values if agg == 'max' else x_s.mean(dim=1)
        elif distance == 'euclidean':
            x_s = x_s.min(dim=1).values if agg == 'min' else x_s.mean(dim=1)
        print(f'COMBINED SIMS: {x_s.shape}')
        return x_s
    elif x.dim() == 4:
        ## x is a batch of sequences and B,N,T,D and y is B,T,D
        ## get the max similarity along T and D axes, into B,B
        x_s = []
        for i in range(x.shape[1]):
            for j in range(x.shape[2]):
                for k in range(y.shape[1]):
                    x_s.append(compute_distance_(x[:,i,j,:],y[:,k,:],distance=distance))
        x_s = torch.stack(x_s, dim=1)
        print(f'COMBINED SIMS: {x_s.shape}')
        
        if distance == 'cosine':
            x_s= x_s.max(dim=1).values if agg == 'max' else x_s.mean(dim=1)
        elif distance == 'euclidean':
            x_s = x_s.min(dim=1).values if agg == 'min' else x_s.mean(dim=1)
        
    else:
        return compute_distance_(x,y,distance=distance)

def compute_distance_(x,y, distance='cosine'):
    
    if x.dtype == torch.float16 or y.dtype == torch.float16:
        x = x.float()
        y = y.float()
    
    if distance == 'cosine':
        x = x / x.norm(dim=-1, keepdim=True)
        y = y / y.norm(dim=-1, keepdim=True)
        return x @ y.t() 
    elif distance == 'euclidean':
        return torch.cdist(x, y, p=2)
    else:
        raise ValueError(f"Distance {distance} not supported")


def compute_sims(text_embeds, audio_embeds, preds, distance='cosine', agg = None):

    # print(f"text_embeds: {text_embeds.shape}")
    # print(f"audio_embeds: {audio_embeds.shape}")
    # print(f"preds: {preds.shape}")
    
    
    print('====BEFORE SIMS=====')
    print(f"audio_embeds: {audio_embeds.shape}")
    print(f"preds: {preds.shape}")
    
    
    audio_embeds = audio_embeds.mean(dim=1) if 'sequence' not in str(agg) else audio_embeds
    
    if 'sequence' not in str(agg):
        preds = preds.mean(dim=2) if preds.dim() == 4 else preds.mean(dim=1)
    
    print('====AFTER SIMS=====')
    print(f"audio_embeds: {audio_embeds.shape}")
    print(f"preds: {preds.shape}")
    # if the embedding dimensions are the same for text embeds and audio embeds, we can compute the similarities directly


    retrieve_gt_text_from_gt_audio = None
    retrieve_gt_audio_from_gt_text = None
    retrieve_gt_text_from_pred_audio = None

    if text_embeds.shape[-1] == audio_embeds.shape[-1]:

        # retrieve_gt_audio_from_gt_text = text_embeds @ audio_embeds.t()
        
        retrieve_gt_audio_from_gt_text = compute_distance(text_embeds, audio_embeds, distance=distance, agg=agg)
        retrieve_gt_text_from_gt_audio = compute_distance(audio_embeds, text_embeds, distance=distance, agg=agg)
        retrieve_gt_text_from_pred_audio = compute_distance(preds, text_embeds, distance=distance, agg=agg)
    
    
    else:
        retrieve_gt_audio_from_gt_text = compute_distance(audio_embeds, audio_embeds, distance=distance, agg=agg)
    retrieve_gt_audio_from_pred_text = compute_distance(preds, audio_embeds, distance=distance, agg=agg)

    
    out_ = {
        'retrieve_gt_text_from_gt_audio': retrieve_gt_text_from_gt_audio,
        'retrieve_gt_audio_from_gt_text': retrieve_gt_audio_from_gt_text,
        'retrieve_gt_text_from_pred_audio': retrieve_gt_text_from_pred_audio,
        'retrieve_gt_audio_from_pred_text': retrieve_gt_audio_from_pred_text
    }

    # for k, v in out_.items():
    #     print(f"{k}: {v.shape}") if v is not None else None

    return out_


def compute_clap_score(sims_dict):

    retrieve_gt_text_from_gt_audio = sims_dict['retrieve_gt_text_from_gt_audio']
    retrieve_gt_audio_from_gt_text = sims_dict['retrieve_gt_audio_from_gt_text']
    retrieve_gt_text_from_pred_audio = sims_dict['retrieve_gt_text_from_pred_audio']
    retrieve_gt_audio_from_pred_text = sims_dict['retrieve_gt_audio_from_pred_text']
    
    _query_shape = min(*retrieve_gt_text_from_pred_audio.shape)
    
    retrieve_gt_text_from_gt_audio = retrieve_gt_text_from_gt_audio[:_query_shape, :_query_shape] if retrieve_gt_text_from_gt_audio is not None else None
    retrieve_gt_audio_from_gt_text = retrieve_gt_audio_from_gt_text[:_query_shape, :_query_shape] if retrieve_gt_audio_from_gt_text is not None else None
    retrieve_gt_text_from_pred_audio = retrieve_gt_text_from_pred_audio[:_query_shape, :_query_shape] if retrieve_gt_text_from_pred_audio is not None else None
    retrieve_gt_audio_from_pred_text = retrieve_gt_audio_from_pred_text[:_query_shape, :_query_shape] if retrieve_gt_audio_from_pred_text is not None else None

    
    retrieve_gt_text_from_gt_audio_diag = torch.diag(retrieve_gt_text_from_gt_audio) if retrieve_gt_text_from_gt_audio is not None else None
    retrieve_gt_audio_from_gt_text_diag = torch.diag(retrieve_gt_audio_from_gt_text) if retrieve_gt_audio_from_gt_text is not None else None
    retrieve_gt_text_from_pred_audio_diag = torch.diag(retrieve_gt_text_from_pred_audio) if retrieve_gt_text_from_pred_audio is not None else None
    retrieve_gt_audio_from_pred_text_diag = torch.diag(retrieve_gt_audio_from_pred_text) if retrieve_gt_audio_from_pred_text is not None else None
    
    clap_score = {
        'diagonals': {
            'retrieve_gt_text_from_gt_audio_CLAP': retrieve_gt_text_from_gt_audio_diag.mean().item() if retrieve_gt_text_from_gt_audio_diag is not None else None,
            'retrieve_gt_audio_from_gt_text_CLAP': retrieve_gt_audio_from_gt_text_diag.mean().item() if retrieve_gt_audio_from_gt_text_diag is not None else None,
            'retrieve_gt_text_from_pred_audio_CLAP': retrieve_gt_text_from_pred_audio_diag.mean().item() if retrieve_gt_text_from_pred_audio_diag is not None else None,
            'retrieve_gt_audio_from_pred_text_CLAP': retrieve_gt_audio_from_pred_text_diag.mean().item() if retrieve_gt_audio_from_pred_text_diag is not None else None
        },
        'averages': {
            'retrieve_gt_text_from_gt_audio_CLAP': (retrieve_gt_text_from_gt_audio - torch.diag(retrieve_gt_text_from_gt_audio_diag)).mean().item() if retrieve_gt_text_from_gt_audio_diag is not None else None,
            'retrieve_gt_audio_from_gt_text_CLAP': (retrieve_gt_audio_from_gt_text - torch.diag(retrieve_gt_audio_from_gt_text_diag)).mean().item() if retrieve_gt_audio_from_gt_text_diag is not None else None,
            'retrieve_gt_text_from_pred_audio_CLAP': (retrieve_gt_text_from_pred_audio - torch.diag(retrieve_gt_text_from_pred_audio_diag)).mean().item() if retrieve_gt_text_from_pred_audio_diag is not None else None,
            'retrieve_gt_audio_from_pred_text_CLAP': (retrieve_gt_audio_from_pred_text - torch.diag(retrieve_gt_audio_from_pred_text_diag)).mean().item() if retrieve_gt_audio_from_pred_text_diag is not None else None
        }
    }

    return clap_score


def compute_retrieval_metrics(query_key_sim, ground_truth_idx, ks=[1, 3, 5, 10], distance='cosine'):

    

    metrics = {f"mean_rank": 0, f"median_rank": 0}
    
    metrics['Recall'] = {}
    metrics['Precision'] = {}
    metrics['mAP'] = {}
    metrics['Hit Rate'] = {}
    
    for k in ks:
        metrics['Recall'][k] = 0
        metrics['Precision'][k] = 0
        metrics['mAP'][k] = 0
        metrics['Hit Rate'][k] = 0
    
    ranks_ = []
    ranks__ = []


    descending = True if distance in ['cosine'] else False


    for i in range(query_key_sim.shape[0]):
        ground_truth_idxx = torch.tensor(ground_truth_idx[i]).unsqueeze(-1)
        ranking = torch.argsort(query_key_sim[i], descending=descending)
        
        
        # print(f"ranking: {ranking}")
        # print(f"ground_truth_idxx: {ground_truth_idxx}")
        
        ranks = torch.isin(ranking, ground_truth_idxx).cpu().numpy()
        ranks = np.where(ranks)[0]

        # Rank Metrics
        ranks_.append(ranks)
        ranks__.append(ranking.cpu())
        
        # Precision, Recall, and mAP
        for k in ks:
            relevant_in_top_k = np.sum(ranks < k)
            total_relevant = len(ground_truth_idx[i])
            
            metrics['Recall'][k] += relevant_in_top_k / total_relevant  # Recall@k
            metrics['Precision'][k] += relevant_in_top_k / k  # Precision@k
            metrics['Hit Rate'][k] += 1 if relevant_in_top_k > 0 else 0  # Hit Rate@k

            # mAP@k
            precisions = [(r < k) * (1 / (r + 1)) for r in ranks]
            if len(precisions) > 0:
                metrics['mAP'][k] += np.sum(precisions) / min(k, total_relevant)
            else:
                metrics['mAP'][k] += 0.0

    
    num_queries = query_key_sim.shape[0]

    # get the mean rank and median rank from ranks_
    ranks_ = np.concatenate(ranks_)
    metrics['mean_rank'] = np.mean(ranks_ + 1)
    metrics['median_rank'] = np.floor(np.median(ranks_)) + 1

    for key in metrics.keys():
        if key not in ['mean_rank', 'median_rank']:
            for k in ks:
                metrics[key][k] /= num_queries
    
                
    return metrics, ranks__


@torch.no_grad()
def eval_dataset(model, dataset, limit_n=-1, distance='cosine', preextracted_features=True, strict_retrieval = False, agg = None, **kwargs):

    model.eval()

    all_metrics = {}
    all_caption_retrieval = {}

    out_ = pred_dataset(
        model=model,
        dataset=dataset,
        limit_n=limit_n,
        preextracted_features=preextracted_features,
        **kwargs
    )

    audio_embeds, text_embeds, preds, file_idx = out_['audio_embeds'], out_[
        'text_embeds'], out_['preds'], out_['file_idx']


    audio_embeds = torch.stack(audio_embeds).cpu()
    text_embeds = torch.cat(text_embeds).cpu()
    preds = torch.stack(preds).cpu()
    preds_ = preds.mean(dim=1, keepdim=False)
    sims_dict = compute_sims(text_embeds, audio_embeds, preds_, distance=distance) if agg is None else agg(text_embeds, audio_embeds, preds, distance=distance, agg=agg)
    clap_sims = compute_sims(text_embeds, audio_embeds, preds_, distance='cosine')
    
    
    
    clap_score = compute_clap_score(clap_sims)

    all_metrics.update(clap_score)
    
    file_idx = [[x] for x in list(range(len(file_idx)))] if strict_retrieval else file_idx

    # if we're retrieving from text to audio or audio to text, the gt idx is the file idx

    # for key in ['gt_text_audio_sims_avg', 'gt_text_audio_sims_max', 'pred_audio_gt_text_sims_avg', 'pred_audio_gt_text_sims_max', 'pred_audio_gt_audio_sims_avg', 'pred_audio_gt_audio_sims_max']:
        
    
    to_compute_ = {
        x:sims_dict[x] for x in ['retrieve_gt_audio_from_gt_text',
                    'retrieve_gt_text_from_gt_audio',
                    'retrieve_gt_text_from_pred_audio',
                    'retrieve_gt_audio_from_pred_text'] if sims_dict[x] is not None
        }
        
    for key in to_compute_.keys():
        retrieval_metrics, ranking = compute_retrieval_metrics(to_compute_[key], file_idx, ks=[1, 3, 5, 10], distance=distance)

        all_metrics.update({
            key : retrieval_metrics
        })
        
        ## from the ranking and the dataset, get the captions of the retrievals in top order
        caption_retrieval = []
        
        for i in range(len(preds)):
            
            gt_caption = dataset[file_idx[i][0]]['prompt']
            argrank = ranking[i].tolist()
            
            retrieved_captions = [dataset[rank]['prompt'] for rank in argrank[:5]]
            
            caption_retrieval += [{'gt': gt_caption, 'retrieved': retrieved_captions}]
            
        all_caption_retrieval.update({
            key: caption_retrieval
        })
        
    out_ = {
        'gt_audio': audio_embeds,
        'gt_text': text_embeds,
        'pred_audio': preds,
    }

    return all_metrics, clap_sims, out_, all_caption_retrieval


def pred_dataset(model, dataset,limit_n=-1, preextracted_features=True, **kwargs):

    model.eval()
    len_dataset = len(dataset)

    pbar = tqdm(itertools.islice(enumerate(dataset), 0, limit_n), total=limit_n if limit_n > 0 else len_dataset)
    
    
    pprint('iterating over dataset to get ground truth embeddings')
    embeddings = get_embeddings(model, dataset, preextracted_features=preextracted_features)
    audio_embeds, text_embeds, file_idx = embeddings['audio_embeds'], embeddings['text_embeds'], embeddings['file_idx']
    
    preds = []
    pprint('iterating over dataset to get predictions')
    for i, datum in pbar:

        # embeddings_and_preds = get_embeddings_and_preds(
        #     model, datum, preextracted_features, **kwargs)
        
        # audio_embeds.append(embeddings_and_preds['audio_embeds']) if isinstance(
        #     embeddings_and_preds['audio_embeds'], torch.Tensor) else embeddings_and_preds['audio_embeds']['embedding_proj']
        
        # text_embeds.append(
        #     embeddings_and_preds['text_embeds'].get('projected_pooler_output',torch.zeros(1,1)))
        
        preds.append(get_preds(model, datum, **kwargs)['preds'])
    
    file_idx = [[i for i, x in enumerate(file_idx) if x == idx] for idx in file_idx]


    return {
        'audio_embeds': audio_embeds,
        'text_embeds': text_embeds,
        'preds': preds,
        'file_idx': file_idx
    }
