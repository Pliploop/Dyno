import sys

from dora.hydra import hydra_main
from omegaconf import DictConfig, OmegaConf
from sagemaker.inputs import TrainingInput
from sagemaker.estimator import Estimator

from dyno.utils.resolvers import register_resolvers
import boto3
import logging


register_resolvers()


@hydra_main(version_base="1.3", config_path="configs", config_name="launch.yaml")
def launch(cfg: DictConfig):
    # --- Collect all Hydra overrides except the launcher script itself ---
    # This replicates your CLI arguments to the training job.
    hydra_args = ["paths=aws"] + sys.argv[1:]  # everything after `python launch.py`
    

    print(OmegaConf.to_yaml(cfg, resolve=True))

    inputs = {
        cfg.data.name: TrainingInput(**cfg.sagemaker.training_input)
    }

    # --- Build the SageMaker Estimator ---
    cfg.sagemaker.estimator.tags = [  # more readable than defining in YAML
        {"Key": k, "Value": v} for k, v in cfg.sagemaker.estimator.tags.items()
    ]

    with open(".secrets/wandb_api_key") as f:
        wandb_api_key = f.read()

    # save the config to a yaml at cfg.sagemaker.estimator.output_path with boto3
    s3 = boto3.client('s3')
    output_bucket, output_key = cfg.sagemaker.estimator.output_path.replace('s3://', '').split('/', 1)
    s3.put_object(Bucket=output_bucket, Key = output_key + '/config.yaml', Body=OmegaConf.to_yaml(cfg, resolve=True))
    logging.info(f'Config saved to {output_bucket}/{output_key}/config.yaml')


    estimator = Estimator(
        **OmegaConf.to_container(cfg.sagemaker.estimator, resolve=True),
        container_arguments=hydra_args,
        environment={
            "S3_DATA_DIR": cfg.sagemaker.s3_data,
            "WANDB_API_KEY": wandb_api_key
        }
    )

    estimator.fit(inputs=inputs,  # magic trick to mount datasets in /opt/ml/data/<dataset_name>
                  logs='All',
                  job_name=cfg.sagemaker.job_name
                  )


if __name__ == "__main__":
    launch()
