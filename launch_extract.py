import sys

import hydra
from omegaconf import DictConfig, OmegaConf
from sagemaker.processing import ProcessingInput, ProcessingOutput, Processor

from gdr.utils.resolvers import register_resolvers
import boto3
from tqdm import tqdm



register_resolvers()


@hydra.main(version_base="1.3", config_path="configs", config_name="launch_extract.yaml")
def launch_extract(cfg: DictConfig):
    """Launch SageMaker processing job for feature extraction.
    
    :param cfg: DictConfig configuration composed by Hydra.
    """
    # --- Collect all Hydra overrides except the launcher script itself ---
    # This replicates your CLI arguments to the processing job.
    hydra_args = ["paths=aws"] + sys.argv[1:]  # everything after `python launch_extract.py`

    print(OmegaConf.to_yaml(cfg, resolve=True))

    # --- Build processing inputs ---
    inputs = []
    
    # Add data input if specified
    if cfg.sagemaker.get("processing_input"):
        inputs.append(
            ProcessingInput(
                source=cfg.sagemaker.processing_input.s3_data,
                destination=cfg.sagemaker.processing_input.destination,
                s3_data_type=cfg.sagemaker.processing_input.get("s3_data_type", "S3Prefix"),
                s3_input_mode=cfg.sagemaker.processing_input.get("s3_input_mode", "File"),
                s3_data_distribution_type=cfg.sagemaker.processing_input.get("s3_data_distribution_type", "FullyReplicated"),
            )
        )

    # --- Build processing outputs ---
    outputs = []
    
    if cfg.sagemaker.get("processing_output"):
        outputs.append(
            ProcessingOutput(
                source=cfg.sagemaker.processing_output.source,
                destination=cfg.sagemaker.processing_output.destination,
                s3_upload_mode=cfg.sagemaker.processing_output.get("s3_upload_mode", "EndOfJob"),
            )
        )

    # --- Build the SageMaker Processor ---
    cfg.sagemaker.processor.tags = [  # more readable than defining in YAML
        {"Key": k, "Value": v} for k, v in cfg.sagemaker.processor.tags.items()
    ]

    processor = Processor(**OmegaConf.to_container(cfg.sagemaker.processor, resolve=True))

    processor.run(
        inputs=inputs,
        outputs=outputs,
        arguments=hydra_args,
        logs=True,
        job_name=cfg.sagemaker.job_name,
    )


if __name__ == "__main__":
    launch_extract()

