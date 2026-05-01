# make this scriptable later on


import json
from dyno.evaluation.retrieval.gen_retrieval import *
from dyno.dataloading.dataloaders import *

import pickle
import wandb

import os
import json
import boto3

from dyno.evaluation.utils import load_model_and_dataset_eval
from rich.pretty import pprint


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
        'training_dataset': training_config['data']['task'],
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
    metrics_r = {}
    for key in data_dict.keys():
        try:
            metrics[key] = {}
            retrieval_metrics = data_dict[key]
            new_dict = {}

            for k in ks:
                new_dict = {}
                new_dict_r = {}
                for metric, values in retrieval_metrics.items():
                    if isinstance(values, dict) and values[k] is not None:
                        new_dict[metric] = round(values[k], 2)
                    else:
                        new_dict[metric] = round(
                            values, 2) if values is not None else None

                metrics[key][k] = new_dict

        except Exception as e:
            print(f'Failed to log {key}, {e}')

    log_dict = {'CLAP': {}, 'Retrieval': {}}

    diags, averages = metrics['diagonals'], metrics['averages']
    for k in ks:
        diag_metrics = diags[k]
        avg_metrics = averages[k]

    for key in diag_metrics.keys():
        log_dict['CLAP'][key] = {
            'diagonal': diag_metrics[key],
            'average': avg_metrics[key]
        }

    for key in metrics.keys():
        if key not in ['diagonals', 'averages']:
            log_dict['Retrieval'][key] = {}
            for k in ks:
                keys = metrics[key][k].keys()

            for key_ in keys:

                m_ = {}
                if key_ not in ['mean_rank', 'median_rank']:
                    for k in ks:
                        m_[k] = metrics[key][k][key_]
                else:
                    m_ = metrics[key][k][key_]

                log_dict['Retrieval'][key][key_] = m_

    if log:

        for metric_, key_ in log_dict.items():

            if metric_ == 'CLAP':
                for key__, value_ in key_.items():

                    table_key = f'CLAP/{task}/{key__}'
                    table_key = table_key.replace(' ', '_')
                    new_table_key = table_key.replace(
                        '/', '').replace(' ', '_')

                    try:
                        table = wandb.use_artifact(
                            f'{project}/run-{id_}-{new_table_key}:latest', type='run_table') if log else None
                        table = table.get(table_key) if log else None
                        table = wandb.Table(columns=['k', 'num_samples_per_prompt', 'inference_guidance',
                                            'steps', 'diagonal', 'average'], data=table.data) if log else None

                    except Exception as e:
                        table = wandb.Table(columns=[
                                            'k', 'num_samples_per_prompt', 'inference_guidance', 'steps', 'diagonal', 'average']) if log else None

                    row = [k, num_samples_per_prompt, inference_guidance,
                           training_steps, value_['diagonal'], value_['average']]
                    table.add_data(*row)

                    try:
                        wandb.log({table_key: table}) if log else None
                    except Exception as e:
                        print(f'Failed to log {table_key}, {e}')

            elif metric_ == 'Retrieval':

                for t2a_a2t, value_ in key_.items():
                    for metric, key__ in value_.items():

                        table_key = f'Retrieval/{task}/{t2a_a2t}/{metric}'
                        table_key = table_key.replace(' ', '_')
                        new_table_key = table_key.replace(
                            '/', '').replace(' ', '_')

                        try:
                            table = wandb.use_artifact(
                                f'{project}/run-{id_}-{new_table_key}:latest', type='run_table') if log else None
                            table = table.get(table_key) if log else None
                            table = wandb.Table(columns=[
                                                'k', 'num_samples_per_prompt', 'inference_guidance', 'steps', 'metric'], data=table.data) if log else None

                        except Exception as e:
                            table = wandb.Table(columns=[
                                                'k', 'num_samples_per_prompt', 'inference_guidance', 'steps', 'metric']) if log else None

                        if isinstance(key__, dict):

                            for k in ks:
                                value = key__[k]
                                row = [k, num_samples_per_prompt,
                                       inference_guidance, training_steps, value]
                                table.add_data(*row)

                        else:
                            row = [k, num_samples_per_prompt,
                                   inference_guidance, training_steps, key__]
                            table.add_data(*row)

                        try:
                            wandb.log({table_key: table}) if log else None
                        except Exception as e:
                            print(f'Failed to log {table_key}, {e}')

    wandb.finish()

    config_copy['metrics'] = log_dict
    config_copy['training_config'] = None
    return config_copy


def run_eval(guidance_scales, model_names,  model_steps, task, log=True, device='cuda:4', distance='cosine', num_samples_per_prompt=[1], limit_n=None, agg=None, return_full_dataset=False):

    metrics = []
    sims = []
    out = []
    captions = []

    if model_steps is None:
        model_steps = [None] * len(model_names)

    for model_name, model_step in zip(model_names, model_steps):

        model, dataset, experiment_name, config = load_model_and_dataset_eval(
            model_name, model_step, task, device=device, return_full_dataset=return_full_dataset)
        
        print(f'Dataset length: {len(dataset)}')

        limit_n = limit_n if limit_n is not None else len(dataset)

        for guidance_scale in guidance_scales:
            for num_samples_per_prompt_ in num_samples_per_prompt:
                print(f'num_samples_per_prompt: {num_samples_per_prompt_}')
                print(f'guidance_scale: {guidance_scale}')
                try:
                    metrics_, sims_, out_, captions_ = eval_dataset(model, dataset, limit_n=limit_n, disable_progress=True, num_steps=50, strict_retrieval=True,
                                                                    guidance_scale=guidance_scale, distance=distance, num_samples_per_prompt=num_samples_per_prompt_, agg=agg)

                    metrics_ = log_results(metrics_, experiment_config=config, experiment_name=experiment_name,
                                           task=task, guidance_scale=guidance_scale, log=log, num_samples_per_prompt=num_samples_per_prompt_, training_steps=model_step)

                    pprint(metrics_)

                    metrics.append(metrics_)
                    config_copy = metrics_.copy()
                    config_copy.pop('metrics')
                    sims.append({**config_copy, 'sims': sims_})
                    out.append({**config_copy, 'out': out_})
                    captions.append({**config_copy, 'captions': captions_})

                except Exception as e:
                    print(
                        f'Failed to evaluate {model_name} with guidance_scale {guidance_scale} and task {task}')
                    raise e

    return metrics, sims, out, captions


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
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda:7')
    parser.add_argument('--save-metrics', type=bool, default=False)
    parser.add_argument('--save-sims', type=bool, default=False)
    parser.add_argument('--save-embeddings', type=bool, default=False)
    parser.add_argument('--save-captions', type=bool, default=False)
    parser.add_argument('--save', type=bool, default=False)
    parser.add_argument('--distance', type=str, default='cosine')
    parser.add_argument('--log', type=bool, default=False)
    parser.add_argument('--file-postfix', type=str, default='')
    parser.add_argument('--test_run', type=bool, default=False)
    parser.add_argument('--limit_n', type=int, default=None)
    parser.add_argument('--return_full_dataset', type=bool, default=False)
    parser.add_argument('--overwrite', type=bool, default=False)
    args = parser.parse_args()
    device = args.device
    log = args.log
    file_postfix = args.file_postfix
    save_metrics, save_sims, save_embeddings, save_captions, save = args.save_metrics, args.save_sims, args.save_embeddings, args.save_captions, args.save
    if save:
        save_metrics, save_sims, save_embeddings, save_captions = True, True, True, True

    print(f'Saving metrics: {save_metrics}')
    print(f'Saving sims: {save_sims}')
    print(f'Saving embeddings: {save_embeddings}')

    print(f'Running on device {device}')

    test_run = args.test_run
    if test_run:
        print('Running in test mode')
    limit_n = 10 if (test_run and args.limit_n is None) else args.limit_n
    
    

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


    # get all the experiments from the dict and build a model_names list
    additional_model_names = [
        # 'diffgar-training-2025-01-01-21-07-18-esphfm-ip-10-2-85-221.ec2.internal', #MLPLDM CLAPCLAP,
        # 'diffgar-training-2025-01-01-20-29-43-d2c1dv-ip-10-2-87-6.ec2.internal', #MSEGar CLAPCLAP
        # 'diffgar-training-2025-01-01-21-42-38-d6ph7q-ip-10-2-201-19.ec2.internal', #MLPMSE CLAPCLAP
        # 'diffgar-training-2025-01-02-02-35-32-t0in2u-ip-10-2-91-172.ec2.internal', #MLPMSE CLAPt5
        # 'diffgar-training-2025-01-02-02-25-42-tge36s-ip-10-2-224-40.ec2.internal', #MLPLDM CLAPt5
        # 'diffgar-training-2025-01-01-21-07-18-esphfm-ip-10-2-85-221.ec2.internal',
        # 'diffgar-training-2025-01-02-02-15-06-btjhmi-ip-10-2-86-108.ec2.internal', #MSEGar CLAPt5
        'diffgar-training-2025-03-18-13-31-12-301bfl-ip-10-0-161-12.ec2.internal'
        
    ]

    def get_models_to_run(dict_, additional_model_names=None):
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
    sims = {}
    out = {}
    captions = {}


    model_steps = [
        [
            # 5000,
            # 10000,
            15000,
            20000,
            # 50000,
            # 100000,
            # 'best'
        ] for _ in model_names
    ]
    
    
    tasks = [
        'song_describer',
        # 'musiccaps'
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
        5,
        # 10,
        # 20,
        # 50,
        # 100
    ]
    for task in tasks:
        for i, model_name in enumerate(model_names):
            # try:
            for model_steps_ in model_steps[i]:
                # check if the experiment is in the json file
                if os.path.exists(f'results/retrieval/{task}/metrics{file_postfix}.json'):
                    with open(f'results/retrieval/{task}/metrics{file_postfix}.json', 'r') as f:
                        data = json.load(f)
                else:
                    data = {}

                new_name = model_name+'-'+str(model_steps_)+'-steps'


                # if the task is not present or the model is not present
                already_run = new_name not in data.get(task, {}).keys() or (new_name in data.get(task, {}).keys() and len(data[task][new_name]) != len(num_samples_per_prompt)*len(guidance_scales))
                # run_again = args.overwrite or not already_run
                if True:
                    
                    # if True:
                    try:
                        metrics_, sims_, out_, captions_ = run_eval(
                            num_samples_per_prompt=num_samples_per_prompt,
                            guidance_scales=guidance_scales,
                            model_names=[model_name],
                            model_steps=[model_steps_],
                            task=task,
                            log=log,
                            device=device,
                            distance=args.distance,
                            limit_n=limit_n,
                            return_full_dataset=args.return_full_dataset,
                            agg=None
                        )
                        model_name_ = model_name+'-'+str(model_steps_)+'-steps'

                        json_path = f'results/retrieval/{task}/metrics{file_postfix}.json'
                        # TODO: Configure these paths according to your local setup
                        pickle_path = f'PATH_TO_RESULTS_DIR/results/retrieval/{task}/sims{file_postfix}.pkl'
                        embedding_path = f'PATH_TO_RESULTS_DIR/results/retrieval/{task}/embeddings{file_postfix}.pkl'
                        caption_path = f'results/retrieval/{task}/captions/captions/{file_postfix}.json'

                        update_json_file(
                            json_path, task, model_name_, metrics_) if save_metrics else None
                        update_pickle_file(
                            pickle_path, task, model_name_, sims_) if save_sims else None
                        update_pickle_file(
                            embedding_path, task, model_name_, out_) if save_embeddings else None
                        update_json_file(
                            caption_path, task, model_name_, captions_) if save_captions else None
                    except Exception as e:
                        print(
                            f'Failed to evaluate {new_name} with task {task}')
                else:
                    print(f'{model_name} already evaluated for task {task}')
