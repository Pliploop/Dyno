from gdr.dataloading.dataloaders import TextAudioDataModule
from gdr.models.ldm.diffusion import LightningDiffGar
from gdr.dataloading.dataloaders import TextAudioDataModule
from pytorch_lightning.cli import SaveConfigCallback, LightningCLI
import os
from jsonargparse import lazy_instance
from pytorch_lightning.strategies import DDPStrategy
import boto3
from botocore.exceptions import NoCredentialsError
from datetime import datetime

import logging

logger = logging.getLogger(__name__)

class LoggerSaveConfigCallback(SaveConfigCallback):
    def save_config(self) -> None:
        
            config = self.parser.dump(self.config, skip_none=False)
            
            #dump to config.py
            
            # with open(self.config_filename, "w") as config_file:
            #     config_file.write(config)
                
            
            
            
            

class MyLightningCLI(LightningCLI):
    
    trainer_defaults = {
        "strategy": lazy_instance(DDPStrategy, find_unused_parameters=False),
    }
    def add_arguments_to_parser(self, parser):
        parser.add_argument("--save_dir", default=None)
        parser.add_argument("--root_path", default=None)
        parser.add_argument("--extract_method", default='get_audio_embedding_from_data')
        parser.add_argument("--out_key", default='embedding_proj')
        parser.add_argument("--hop", default=48000)
        parser.add_argument("--limit_n", default=None)
        parser.add_argument("--save", default=False)
        parser.add_argument('--device', default='cuda:0')
        parser.add_argument('--extracted_at', default=None)
        


if __name__ == "__main__":

    #intercept the --config argument before it reaches the parser (for sagemaker)
    import sys


    # try:    

    cli = MyLightningCLI(model_class=LightningDiffGar, datamodule_class=TextAudioDataModule, seed_everything_default=123,
                        run=False, save_config_callback=LoggerSaveConfigCallback, save_config_kwargs={"overwrite": True},trainer_defaults=MyLightningCLI.trainer_defaults)

    logging.basicConfig(format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO)
    
    
    
    dm = cli.datamodule
    ldm = cli.model
    
    dm.setup(None)
    ldm.encoder_pair.to(cli.config['device'])
    
    cli.config['extracted_at'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    
    save_dir = cli.config.save_dir
    if 's3://' in save_dir:
        import boto3
        import io
        client = boto3.client('s3')
        bucket, key = save_dir.replace("s3://", "").split("/", 1)
        key = f"{key}/config.yaml"
        
        try:
            # client.upload_file(save_path, bucket, key)
            with open("config.yaml", "w") as config_file:
                config_file.write(cli.parser.dump(cli.config, skip_none=False))
            client.upload_file("config.yaml", bucket, key)
            os.remove("config.yaml")
        except NoCredentialsError:
            print("No AWS credentials found. Please set up your AWS credentials.")
    else:
        
        if save_dir is None:
            save_dir = os.path.dirname(dm.train_dataset.annotations[0]['file_path'])
        os.makedirs(save_dir, exist_ok=True)
        with open(os.path.join(save_dir, "config.yaml"), "w") as config_file:
            config_file.write(cli.parser.dump(cli.config, skip_none=False))

    save_dir, extract_method, out_key, hop, limit_n = cli.config['save_dir'], cli.config['extract_method'], cli.config['out_key'], cli.config['hop'], cli.config['limit_n']
    save = cli.config['save']
    root_path = cli.config['root_path']
    
    
    print(f"Extracting features with {extract_method} method, saving to {save_dir}, out_key: {out_key}, hop: {hop}, limit_n: {limit_n}, save: {save}")
    
    datasets = [dm.train_dataset] + dm.val_datasets + dm.test_datasets
    
    for dataset in datasets:
        if dataset is not None:
            dataset.extract_and_save_features(ldm.encoder_pair, save_dir = save_dir, extract_method=extract_method, out_key = out_key, hop = hop, limit_n = limit_n, save = save, verbose = False, root_path = root_path)
    