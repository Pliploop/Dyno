import boto3
import wandb
from gdr.dataloading.dataloaders import TextAudioDataModule
from gdr.models.ldm.diffusion import LightningDiffGar,LightningMLPldm,LightningMLPMSE,LightningMSEGar
from rich.pretty import pprint

def load_model_and_dataset_eval(model_name, model_step, task, device='cuda:4', return_stats_path = False, return_full_dataset = False):

    model_step = model_step if model_step is not None else 100000

    path = f's3://maml-aimcdt/storage/julien/DiffGAR/training_checkpoints/{model_name}'
    experiment_name = path.split('/')[-1]
    config_path = path + '/config.yaml'
    
    # get the keys in the bucket
    s3 = boto3.client('s3')
    objects = s3.list_objects_v2(Bucket='maml-aimcdt', Prefix=f'storage/julien/DiffGAR/training_checkpoints/{model_name}')
    keys = [obj['Key'] for obj in objects.get('Contents', [])]
    
    
    #get the key where 'best' is in the name
    p = [key for key in keys if 'best' in key]
    p = p[0] if len(p) > 0 else None
    object_name = p.split('/')[-1] if p is not None else ''
    
    
    ckpt_path = path + f'/checkpoint-step={model_step}-recent.ckpt' if model_step != 'best' else path + f'/{object_name}'

    # get the config from wandb
    api = wandb.Api()
    
    
    runs = api.runs(f'jul-guinot/DiffGAR-LDM')
    # get the config where names match the experiment name
    for run in runs:
        if run.name == experiment_name:
            config = run.config
            
    model_cls = eval(config['model'].get('class_path', 'LightningDiffGar').split('.')[-1])
    pprint(model_cls)

    if 'init_args' in config['model']:
        config['model'] = config['model']['init_args']
    
    # pprint(config['model'])
    channels = config['model']['unet_model_config']['in_channels'] if 'unet_model_config' in config['model'] else config['model']['mlp_model_config']['in_channels']

    training_encoder_pair = config['model']['encoder_pair']

    # TODO: Configure these paths according to your local setup
    encoder_pair_to_new_dir = {
        'song_describer': {
            'muleT5':   'PATH_TO_SONG_DESCRIBER_MULE_DATA',
            'clap':    'PATH_TO_SONG_DESCRIBER_CLAP_DATA',
            'music2latent' : f'PATH_TO_SONG_DESCRIBER_MUSIC2LATENT_DATA/{channels}/1hz',
            'MusCALL': 'PATH_TO_SONG_DESCRIBER_MUSCALL_DATA'
        },
        'musiccaps': {
            'muleT5':   'PATH_TO_MUSICCAPS_MULE_DATA',
            'clap':    'PATH_TO_MUSICCAPS_CLAP_DATA',
            'MusCALL': 'PATH_TO_MUSICCAPS_MUSCALL_DATA'
        },
        'upmm': {
            'clap' : None,
            'muleT5': None,
            'MusCALL': None
    }
    }
    
    # TODO: Configure these paths according to your local setup
    encoder_pair_to_old_dir = {
        'song_describer': 'PATH_TO_SONG_DESCRIBER_AUDIO_DATA',
        'musiccaps': 'PATH_TO_MUSICCAPS_AUDIO_DATA',
        'upmm': None
    }
    
    # TODO: Configure these paths according to your local setup
    task_to_task_kws = {
        'musiccaps': {
                'data_path': 'PATH_TO_MUSICCAPS_AUDIO_DATA',
                'csv_path': 'PATH_TO_MUSICCAPS_CSV'
        },
        'song_describer': {
                'data_path': 'PATH_TO_SONG_DESCRIBER_AUDIO_DATA',
                'csv_path': 'PATH_TO_SONG_DESCRIBER_CSV'
        },
        'upmm': {
            'data_path': 'PATH_TO_UPMM_DATA',
            'csv_path': 's3://maml-aimcdt/datasets/embeddings/GDR-MusCALL/upmm.csv'
        }
    }
    
    encoder_pair_to_stats_path = {
        'clap': 's3://maml-aimcdt/datasets/embeddings/spotify_most_popular/clap/1hz/statistics',
        'muleT5': 's3://maml-aimcdt/datasets/embeddings/spotify_most_popular/mule_512_norm/1hz/statistics'
    }
        

    
    model = model_cls.from_pretrained(config_path, ckpt_path, device=device)
    model.to(device)
    
    
    dm_config = {
        'tasks' : [
            {
                'task': task,
                'task_kwargs': task_to_task_kws[task],
                'split': 'val' if return_full_dataset else 'keep',
                'root_dir': encoder_pair_to_old_dir[task],
                'new_dir': encoder_pair_to_new_dir[task][training_encoder_pair],
            }
        ],
        'dataloader_kwargs': {
            'batch_size': 1,
            'preextracted_features': True,
            'truncate_preextracted': 64
        }
    }

    # latent_dm = TextAudioDataModule(
    #     task=task,
    #     task_kwargs=task_to_task_kws[task],
    #     batch_size=1,
    #     preextracted_features=True,
    #     truncate_preextracted=64,
    #     new_dir=encoder_pair_to_new_dir[task][training_encoder_pair],
    #     root_dir=encoder_pair_to_old_dir[task]
    # )
    
    latent_dm = TextAudioDataModule(**dm_config)
    
    # if return_full_dataset:
    #     latent_dm.test_annotations = latent_dm.annotations
    #     latent_dm.val_annotations = latent_dm.annotations
    
    latent_dm.setup('eval')
    
    
    if return_full_dataset:
        dataset = latent_dm.val_datasets[0]
    else:
        if task == 'song_describer':
            dataset = latent_dm.val_datasets[0]
        else:
            # might need to modify musiccaps so that it is all in the test set
            dataset = latent_dm.test_datasets[0]
    

    if return_stats_path:
        return model, dataset, experiment_name, config, encoder_pair_to_stats_path[training_encoder_pair]
    return model, dataset, experiment_name, config


