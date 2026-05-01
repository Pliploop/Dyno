# make this scriptable later on


import json
from dyno.evaluation.fidelity_diversity.features import get_fixed_length_motion_features, get_variable_length_motion_features
from dyno.evaluation.fidelity_diversity.diversity import vendi_score, mics, mivs, density_coverage_precision_recall
from dyno.evaluation.fidelity_diversity.fidelity import frechet_distance
from dataset_statistics import OnlineStats
from dyno.dataloading.dataloaders import *

import pickle
import wandb

import torch
import os
import json
from rich.pretty import pprint

from dyno.evaluation.utils import load_model_and_dataset_eval



def log_results(data_dict, experiment_config=None, experiment_name=None, task=None, guidance_scale=None, log=False, num_samples_per_prompt=None, ks=[1, 3, 5, 10], training_steps=None):
    # Extract CLAP scores from 'diagonals' and 'averages' into a DataFrame
    
    
    if 'unet_model_config' in experiment_config['model']:
        config_key = 'unet_model_config'
    else:
        config_key = 'mlp_model_config'
    

    training_guidance = experiment_config['model'][config_key].get(
        'classifier_free_guidance_strength', 0.0)

    inference_guidance = guidance_scale
    original_name = experiment_name
    training_config = experiment_config

    project = 'DiffGAR-LDM'

    # get the run id from the experiment name
    import wandb
    api = wandb.Api()
    run = api.runs(project)
    for r in run:
        if r.name == experiment_name:
            run = r
            break

    # training_config = run.config
    id_ = run.id

    config = {
        'model_class': run.config['model'].get('class_path', 'LightningDiffGar').split('.')[-1],
        'training_guidance': training_guidance,
        'inference_guidance': inference_guidance,
        'task': task,
        'training_steps': training_steps,
        'experiment_name': experiment_name,
        'num_samples_per_prompt': num_samples_per_prompt,
        'original_name': original_name,
        'training_config': training_config,
        'training_dataset': training_config['data']['tasks'],
        'model_scale': training_config['model'][config_key].get('name', None),
        'training_encoder_pair': training_config['model']['encoder_pair'],
        'text_encoder': training_config['model'].get('text_encoder', None),
        'contrastive_loss_weight': training_config['model'].get('contrastive_loss_kwargs', {}).get('weight', None),
    }

    config_copy = config.copy()
    config_copy.pop('experiment_name')
    config_copy.pop('training_config')
    config_copy.pop('original_name')

    # resume the run
    wandb.init(project=project, id=id_, resume=True) if log else None

    metrics = {}
    for key in data_dict.keys():
        try:
            metrics[key] = data_dict[key]
        except Exception as e:
            pprint(f'Failed to log {key}, {e}')

    log_dict = {'Fidelity': {}, 'Diversity': {}}

    
    for key in metrics.keys():
        if 'frechet_distance' in key:
            log_dict['Fidelity'][key] = metrics[key]
        else:
            log_dict['Diversity'][key] = metrics[key]

    for metric_, key_ in log_dict.items():
        if log:
            if metric_ == 'Fidelity':
                for key__, value_ in key_.items():

                    table_key = f'Fidelity/{task}/{key__}'
                    table_key = table_key.replace(' ', '_')
                    new_table_key = table_key.replace('/', '').replace(' ', '_')

                    try:
                        table = wandb.use_artifact(
                            f'{project}/run-{id_}-{new_table_key}:latest', type='run_table') if log else None
                        table = table.get(table_key) if log else None
                        table = wandb.Table(columns=['k', 'num_samples_per_prompt', 'inference_guidance',
                                            'steps', 'encoded', 'generated'], data=table.data) if log else None

                    except Exception as e:
                        table = wandb.Table(columns=[
                                            'k', 'num_samples_per_prompt', 'inference_guidance', 'steps', 'encoded', 'generated']) if log else None

                    row = [num_samples_per_prompt, inference_guidance,
                        training_steps, value_['encoded'], value_['generated']]
                    table.add_data(*row)

                    try:
                        wandb.log({table_key: table})
                    except Exception as e:
                        pprint(f'Failed to log {table_key}, {e}')
                        


    wandb.finish() if log else None

    config_copy['metrics'] = log_dict
    config_copy['training_config'] = None
    return config_copy


def eval_dataset(model, dataset,  stats_path, limit_n=None, disable_progress=True, num_samples_per_prompt=1, **kwargs):
    from dyno.evaluation.fidelity_diversity.fidelity import pred_dataset
    
    gen_static_stats = OnlineStats('sequence_gen')
    gen_fixed_length_motion_stats = OnlineStats('fixed_length_motion_gen')
    gen_variable_length_motion_stats = OnlineStats('variable_length_motion_gen')
    
    encoder_static_stats = OnlineStats('sequence_encoder')
    encoder_fixed_length_motion_stats = OnlineStats('fixed_length_motion_encoder')
    encoder_variable_length_motion_stats = OnlineStats('variable_length_motion_encoder')
    
    
    ground_static_stats = OnlineStats('ground_truth_static')
    ground_fixed_length_motion_stats = OnlineStats('ground_truth_fixed_length_motion')
    ground_variable_length_motion_stats = OnlineStats('ground_truth_variable_length_motion')
    
    ground_static_stats.load_from_pkl(stats_path+'/sequence_stats.pkl')
    ground_fixed_length_motion_stats.load_from_pkl(stats_path+'/fixed_length_motion_features_stats.pkl')
    ground_variable_length_motion_stats.load_from_pkl(stats_path+'/variable_length_motion_features_stats.pkl')
                                      
    out_ = pred_dataset(model=model, dataset=dataset, limit_n=limit_n, preextracted_features=True, disable_progress=disable_progress, num_samples_per_prompt=num_samples_per_prompt, **kwargs)
    
    audio_embeds, text_embeds, preds, _ = out_['audio_embeds'], out_[
        'text_embeds'], out_['preds'], out_['file_idx']
    
    
    mics_ = mics(preds)
    mivs_ = mivs(preds)
    
    preds = torch.cat(preds).detach().cpu()
    audio_embeds = torch.cat([a_.unsqueeze(0) for a_ in audio_embeds]).detach().cpu()
    
    vendi_ = vendi_score(preds.mean(dim=1))
    prdc = density_coverage_precision_recall(
        grounding_embeddings=audio_embeds.mean(dim=1),
        generated_embeddings=preds.mean(dim=1),
        nearest_k=5
        )
    
    
    pprint(f'Generated shape: {preds.shape}')
    pprint(f'Audio embeddings shape: {audio_embeds.shape}')
    
    gen_static_stats.fit(preds,preprocess_fn=lambda x: x.mean(0), preprocess_kwargs={})
    gen_fixed_length_motion_stats.fit(preds,preprocess_fn=get_fixed_length_motion_features, preprocess_kwargs={'fixed_length': 64})
    gen_variable_length_motion_stats.fit(preds,preprocess_fn=get_variable_length_motion_features, preprocess_kwargs={'fixed_length': 64})
    
    encoder_static_stats.fit(audio_embeds,preprocess_fn=lambda x: x.mean(0), preprocess_kwargs={})
    encoder_fixed_length_motion_stats.fit(audio_embeds,preprocess_fn=get_fixed_length_motion_features, preprocess_kwargs={'fixed_length': 64})
    encoder_variable_length_motion_stats.fit(audio_embeds,preprocess_fn=get_variable_length_motion_features, preprocess_kwargs={'fixed_length': 64})
    
    metrics = {
        'static_frechet_distance' :
            {'encoded': frechet_distance(encoder_static_stats, ground_static_stats),
             'generated': frechet_distance(gen_static_stats, ground_static_stats)},
        'fixed_length_motion_frechet_distance' :
            {'encoded': frechet_distance(encoder_fixed_length_motion_stats, ground_fixed_length_motion_stats),
             'generated': frechet_distance(gen_fixed_length_motion_stats, ground_fixed_length_motion_stats)},
        'variable_length_motion_frechet_distance' :
            {'encoded': frechet_distance(encoder_variable_length_motion_stats, ground_variable_length_motion_stats),
             'generated': frechet_distance(gen_variable_length_motion_stats, ground_variable_length_motion_stats)},
        'mics': mics_.item(),
        'mivs': mivs_.item(),
        'vendi': vendi_.item(),
        **{
            k:v.item() for k,v in prdc.items()
        }
    }
    
    return metrics
    
    
def run_eval(guidance_scales, model_names,  model_steps, task, log=True, device='cuda:4', distance='cosine', num_samples_per_prompt=[1], limit_n=None, return_full_dataset=False, **kwargs):

    metrics = []
    captions = []

    if model_steps is None:
        model_steps = [None] * len(model_names)

    for model_name, model_step in zip(model_names, model_steps):

        model, dataset, experiment_name, config, stats_path = load_model_and_dataset_eval(
            model_name, model_step, task, device=device, return_stats_path=True, return_full_dataset=return_full_dataset)

        limit_n = limit_n if limit_n is not None else len(dataset)

        for guidance_scale in guidance_scales:
            for num_samples_per_prompt_ in num_samples_per_prompt:
                try:
                    metrics_ = eval_dataset(model, dataset, limit_n=limit_n, disable_progress=True, num_steps=50,
                                                                    guidance_scale=guidance_scale, num_samples_per_prompt=num_samples_per_prompt_, stats_path=stats_path)
                    metrics_ = log_results(metrics_, experiment_config=config, experiment_name=experiment_name,
                                           task=task, guidance_scale=guidance_scale, log=log, num_samples_per_prompt=num_samples_per_prompt_, training_steps=model_step)

                    pprint(metrics_)

                    metrics.append(metrics_)
                    config_copy = metrics_.copy()
                    config_copy.pop('metrics')

                except Exception as e:
                    pprint(
                        f'Failed to evaluate {model_name} with guidance_scale {guidance_scale} and task {task}, {e}')
                    raise e
    return metrics


def update_json_file(file_path, task, model_name, metrics_):

    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    if os.path.exists(file_path):
        with open(file_path, 'r') as f:
            data = json.load(f)
    else:
        data = {}

    if task not in data.keys():
        data[task] = {model_name: metrics_}
    else:
        data[task].update({model_name: metrics_})

    with open(file_path, 'w') as f:
        json.dump(data, f, indent=4)


def update_pickle_file(file_path, task, model_name, sims_):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    if os.path.exists(file_path):
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
    else:
        data = {}

    if task not in data.keys():
        data[task] = {model_name: sims_}
    else:
        data[task].update({model_name: sims_})

    with open(file_path, 'wb') as f:
        pickle.dump(data, f)


if __name__ == '__main__':

    # get the device from the command line --device argument
    import argparse
    from omegaconf import OmegaConf
    args = OmegaConf.from_cli()
    
    device = args.device if hasattr(args, 'device') else 'cpu'
    log = args.log if hasattr(args, 'log') else False
    file_postfix = args.file_postfix if hasattr(args, 'file_postfix') else ''
    save_metrics = args.save_metrics if hasattr(args, 'save_metrics') else False
    save_sims = args.save_sims if hasattr(args, 'save_sims') else False
    save_embeddings = args.save_embeddings if hasattr(args, 'save_embeddings') else False
    save_captions = args.save_captions if hasattr(args, 'save_captions') else False
    save = args.save if hasattr(args, 'save') else False
    if save:
        save_metrics, save_sims, save_embeddings, save_captions = True, True, True, True


    pprint(args)

    test_run = args.test_run if hasattr(args, 'test_run') else False
    if test_run:
        pprint('Running in test mode')
    limit_n = 10 if test_run else None

    experiments = {
        'base': {
            'task': {
                'upmm': {
                    # 'CLAPT5': {'model_name': 'diffgar-training-2024-10-12-00-59-43-7lnzqj-ip-10-2-239-154.ec2.internal'},
                    # 'CLAPCLAP': {'model_name': 'diffgar-training-2024-10-09-15-32-35-9ngrhp-ip-10-0-73-91.ec2.internal'},
                    # 'MULET5': {'model_name': 'diffgar-training-2024-10-22-23-39-20-2xguwt-ip-10-2-125-92.ec2.internal'},
                    # 'MUSCALL': {'model_name': 'diffgar-training-2024-11-22-22-12-17-rrane1-ip-10-0-115-118.ec2.internal'},
                    # 'MUSCALLT5': {'model_name': 'diffgar-training-2024-11-23-17-03-22-oyiqx1-ip-10-2-118-41.ec2.internal'},
                },
                'song_describer': {
                    # 'CLAPT5': {'model_name': 'diffgar-training-2024-10-12-00-32-45-9i65xk-ip-10-0-73-210.ec2.internal'},
                    # 'CLAPCLAP': {'model_name': 'diffgar-training-2024-10-08-15-39-16-394tsk-ip-10-2-207-31.ec2.internal'},
                    # 'MULET5': {'model_name': 'diffgar-training-2024-10-23-10-24-58-ecrjda-ip-10-2-200-242.ec2.internal'},
                    # 'MUSCALL': {'model_name': 'dandy-blaze-263'},
                    # 'MUSCALLT5': {'model_name': 'fancy-forest-260'},
                }

            }
        },
        'model_scale': {
            'task': {
                'upmm': {
                    'CLAPT5': {
                        # 'xlarge': {'model_name': 'diffgar-training-2024-11-19-02-41-34-a877a0-ip-10-2-125-78.ec2.internal'},
                        # 'large': {'model_name': 'diffgar-training-2024-11-19-01-53-40-cc99d2-ip-10-2-103-60.ec2.internal'},
                        # 'small': {'model_name': 'diffgar-training-2024-11-19-02-57-35-yki4n1-ip-10-0-150-137.ec2.internal'},
                        # 'tiny': {'model_name': 'diffgar-training-2024-11-19-02-20-37-nn22pe-ip-10-0-157-183.ec2.internal'},
                    },
                    'CLAPCLAP': {
                        # 'xlarge': {'model_name': 'diffgar-training-2024-10-10-08-56-46-3a54dk-ip-10-0-136-252.ec2.internal'},
                        # 'large': {'model_name': 'diffgar-training-2024-10-09-15-48-35-v5sxgr-ip-10-0-86-191.ec2.internal'},
                        # 'small': {'model_name': 'diffgar-training-2024-10-09-15-17-25-8sarg2-ip-10-0-206-118.ec2.internal'},
                        # 'tiny': {'model_name': 'diffgar-training-2024-10-09-16-00-15-1y2807-ip-10-2-224-244.ec2.internal'},

                    },
                    'MULET5': {
                        # 'xlarge': {'model_name': 'diffgar-training-2024-10-23-01-08-58-gr6g8k-ip-10-0-125-16.ec2.internal'},
                        # 'large': {'model_name': 'diffgar-training-2024-10-23-00-57-51-azhmyc-ip-10-0-180-242.ec2.internal'},
                        # 'small': {'model_name': 'diffgar-training-2024-10-23-00-41-42-vra4x4-ip-10-2-91-162.ec2.internal'},
                        # 'tiny': {'model_name': 'diffgar-training-2024-10-23-00-07-24-e4fja8-ip-10-2-221-193.ec2.internal'},
                    },
                    'MUSCALL': {
                        # 'xlarge': {'model_name': 'diffgar-training-2024-11-22-20-13-09-ejoi2c-ip-10-2-70-16.ec2.internal'},
                        # 'large': {'model_name': 'diffgar-training-2024-11-22-22-22-20-1qyom7-ip-10-0-72-185.ec2.internal'},
                        # 'small': {'model_name': 'diffgar-training-2024-11-23-01-19-17-z47m0t-ip-10-2-104-202.ec2.internal'},
                        # 'tiny': {'model_name': 'diffgar-training-2024-11-23-01-41-07-14x1x3-ip-10-2-231-233.ec2.internal'},
                    },
                    'MUSCALLT5': {
                        # 'xlarge': {'model_name': 'diffgar-training-2024-11-23-16-45-27-dn2byj-ip-10-0-192-145.ec2.internal'},
                        # 'large': {'model_name': 'diffgar-training-2024-11-23-18-43-50-uzvsr9-ip-10-0-86-33.ec2.internal'},
                        # 'small': {'model_name': 'diffgar-training-2024-11-23-16-21-41-fx8bzd-ip-10-0-234-253.ec2.internal'},
                        # 'tiny': {'model_name': 'diffgar-training-2024-11-23-15-54-17-ss9kcf-ip-10-0-99-239.ec2.internal'},
                    },
                },
                'song_describer': {
                    'CLAPT5': {
                        # 'xlarge' : {'model_name' : ''},
                        # 'large' : {'model_name' : ''},
                        # 'small': {'model_name': 'smooth-lake-225'},
                        # 'tiny': {'model_name': 'breezy-wind-226'},
                    },
                    'CLAPCLAP': {
                        # 'xlarge': {'model_name': 'diffgar-training-2024-10-08-16-11-19-p6io6d-ip-10-0-156-223.ec2.internal'},
                        # 'large': {'model_name': 'diffgar-training-2024-10-09-08-20-15-pwappv-ip-10-0-121-167.ec2.internal'},
                        # 'small': {'model_name': 'diffgar-training-2024-10-08-14-53-57-zqszhl-ip-10-0-154-119.ec2.internal'},
                        # 'tiny': {'model_name': 'diffgar-training-2024-10-08-14-46-11-jwnby2-ip-10-2-94-233.ec2.internal'}
                    },
                    'MULET5': {
                        # 'xlarge' : {'model_name' : ''},
                        # 'large' : {'model_name' : ''},
                        # 'small': {'model_name': 'trim-sunset-227'},
                        # 'tiny': {'model_name': 'rare-blaze-228'},
                    },
                    'MUSCALL': {
                        # 'xlarge' : {'model_name' : ''},
                        # 'large' : {'model_name' : ''},
                        # 'small': {'model_name': 'worthy-cosmos-262'},
                        # 'tiny': {'model_name': 'royal-elevator-261'},
                    },
                    'MUSCALLT5': {
                        # 'xlarge' : {'model_name' : ''},
                        # 'large' : {'model_name' : ''},
                        # 'small': {'model_name': 'wandering-snowball-259'},
                        # 'tiny': {'model_name': 'efficient-blaze-258'},
                    },
                },
            },
        },
    }

    tasks = [
        'song_describer',
        # 'musiccaps'
        # 'upmm'
    ]
    
    

    # get all the experiments from the dict and build a model_names list

    additional_model_names = [
        'diffgar-training-2025-03-21-01-28-07-kq8uhb-ip-10-0-193-207.ec2.internal', ##diffunet
        'misunderstood-glade-392', #MLPMSE
        'fine-elevator-388', #diffMLP
        'deep-smoke-404'
        
        
        
    ]


    def get_models_to_run(dict_, additional_model_names = None):
        model_names = []
        for key, value in dict_.items():
            if key == 'model_name':
                model_names.append(value)

            elif isinstance(value, dict):
                model_names.extend(get_models_to_run(value))
                
        if additional_model_names is not None:
            model_names.extend(additional_model_names)

        return model_names
    
    
    
    

    model_names = get_models_to_run(experiments, additional_model_names=additional_model_names)

    metrics = {}

    model_steps = [
        [
            # 5000,
            # 10000,
            # 15000,
            # 20000,
            50000,
            # 100000,
            # 'best'
        ] for _ in model_names
    ]

    # guidance_scales = [0,0.1,0.3,0.5,1,5,10]
    guidance_scales = [
        # 0,
        # 1,
        3,
        # 5,
        # 10,
        # 20,
    ]
    num_samples_per_prompt = [
        # 1,
        # 5,
        10,
        # 20,
        # 50,
        # 100
    ]
    
    for task in tasks:
        for i, model_name in enumerate(model_names):
            # try:
            for model_steps_ in model_steps[i]:
                # check if the experiment is in the json file
                if os.path.exists(f'results/fidelity/{task}/metrics{file_postfix}.json'):
                    with open(f'results/fidelity/{task}/metrics{file_postfix}.json', 'r') as f:
                        data = json.load(f)
                else:
                    data = {}

                new_name = model_name+'-'+str(model_steps_)+'-steps'

                # if the task is not present or the model is not present
                if new_name not in data.get(task, {}).keys() or (new_name in data.get(task, {}).keys() and len(data[task][new_name]) != len(num_samples_per_prompt)*len(guidance_scales)):
                    # if True:
                    try:
                        metrics_ = run_eval(
                            num_samples_per_prompt=num_samples_per_prompt,
                            guidance_scales=guidance_scales,
                            model_names=[model_name],
                            model_steps=[model_steps_],
                            task=task,
                            log=log,
                            device=device,
                            distance=args.distance if hasattr(args, 'distance') else 'cosine',
                            limit_n=limit_n,
                            return_full_dataset=args.return_full_dataset
                        )
                        model_name_ = model_name+'-'+str(model_steps_)+'-steps'

                    
                        json_path = f'results/fidelity/{task}/metrics{file_postfix}.json'
                        
                        pprint(metrics_)

                        update_json_file(
                            json_path, task, model_name_, metrics_) if save_metrics else None
                    except Exception as e:
                        pprint(
                            f'Failed to evaluate {new_name} with task {task}, {e}')
                        
                else:
                    pprint(f'{model_name} already evaluated for task {task}')
