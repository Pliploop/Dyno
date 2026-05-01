"""
Launch SageMaker Processing job for the Gradio retrieval app.

Builds the SageMaker Processor and uses sagemaker config (like launch.py).
The container runs gdr/gradio_retrieval_app.py with the passed Hydra arguments.

Usage:
    python launch_gradio.py sagemaker=gradio [overrides]
"""
import sys
import os

from dora.hydra import hydra_main
from omegaconf import DictConfig, OmegaConf
from sagemaker.processing import ProcessingInput, ProcessingOutput, Processor
from omegaconf import ListConfig

from dyno.utils.resolvers import register_resolvers

register_resolvers()


@hydra_main(version_base="1.3", config_path="configs", config_name="launch_gradio.yaml")
def launch(cfg: DictConfig):
    if not cfg.get("sagemaker") or not cfg.sagemaker.get("processor"):
        raise ValueError(
            "Run with sagemaker config: python launch_gradio.py sagemaker=gradio [overrides]"
        )

    # Replicate CLI arguments to the processing job (container runs gdr/gradio_retrieval_app.py with these)
    hydra_args = ["paths=gradio"] + sys.argv[1:]

    print(OmegaConf.to_yaml(cfg, resolve=True))

    inputs = []
    if cfg.sagemaker.get("processing_input"):
        if not (isinstance(cfg.sagemaker.processing_input, list) or isinstance(cfg.sagemaker.processing_input, ListConfig)):
            inputs.append(
                ProcessingInput(
                    source=cfg.sagemaker.processing_input.s3_data,
                    destination=cfg.sagemaker.processing_input.destination,
                    s3_data_type=cfg.sagemaker.processing_input.get("s3_data_type", "S3Prefix"),
                    s3_input_mode=cfg.sagemaker.processing_input.get("s3_input_mode", "File"),
                    s3_data_distribution_type=cfg.sagemaker.processing_input.get(
                        "s3_data_distribution_type", "FullyReplicated"
                    ),
                )
            )
        else:
            for input in cfg.sagemaker.processing_input:
                inputs.append(
                    ProcessingInput(
                        source=input.s3_data,
                        destination=input.destination,
                        s3_data_type=input.get("s3_data_type", "S3Prefix"),
                        s3_input_mode=input.get("s3_input_mode", "File"),
                        s3_data_distribution_type=input.get("s3_data_distribution_type", "FullyReplicated"),
                        input_name=input.get("input_name", None),
                    )
                )

    outputs = []
    if cfg.sagemaker.get("processing_output"):
        outputs.append(
            ProcessingOutput(
                source=cfg.sagemaker.processing_output.source,
                destination=cfg.sagemaker.processing_output.destination,
                s3_upload_mode=cfg.sagemaker.processing_output.get("s3_upload_mode", "EndOfJob"),
            )
        )

    cfg.sagemaker.processor.tags = [
        {"Key": k, "Value": str(v)} for k, v in cfg.sagemaker.processor.tags.items()
    ]

    env = dict(OmegaConf.to_container(cfg.sagemaker.processor.get("env", {}), resolve=True) or {})
    secrets_path = os.path.join(os.path.dirname(__file__) or ".", ".secrets", "wandb_api_key")
    if os.path.exists(secrets_path):
        with open(secrets_path) as f:
            env["WANDB_API_KEY"] = f.read().strip()

    processor_kw = OmegaConf.to_container(cfg.sagemaker.processor, resolve=True)
    processor_kw["env"] = env

    processor = Processor(**processor_kw)

    processor.run(
        inputs=inputs,
        outputs=outputs,
        arguments=hydra_args,
        logs=True,
        job_name=cfg.sagemaker.job_name,
    )


if __name__ == "__main__":
    launch()
