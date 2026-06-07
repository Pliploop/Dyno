import os
from typing import Any, Dict, List, Optional, Tuple

from dora import get_xp, hydra_main
import hydra

import lightning as L
import rootutils
import torch
from lightning import Callback, LightningDataModule, LightningModule, Trainer
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig, open_dict
import logging


rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
# ------------------------------------------------------------------------------------ #
# the setup_root above is equivalent to:
# - adding project root dir to PYTHONPATH
#       (so you don't need to force user to install project as a package)
#       (necessary before importing any local modules e.g. `from gdr import utils`)
# - setting up PROJECT_ROOT environment variable
#       (which is used as a base for paths in "configs/paths/default.yaml")
#       (this way all filepaths are the same no matter where you run the code)
# - loading environment variables from ".env" in root dir
#
# you can remove it if you:
# 1. either install project as a package or move entry files to project root dir
# 2. set `root_dir` to "." in "configs/paths/default.yaml"
#
# more info: https://github.com/ashleve/rootutils
# ------------------------------------------------------------------------------------ #

from dyno.utils import (
    RankedLogger,
    extras,
    get_metric_value,
    instantiate_callbacks,
    instantiate_loggers,
    log_hyperparameters,
    register_resolvers,
    task_wrapper,
)
from dyno.utils.experiment_registry import resolve_experiment_reference

log = RankedLogger(__name__, rank_zero_only=True)
register_resolvers()


def _cuda_capability() -> tuple[int, int] | None:
    if not torch.cuda.is_available():
        return None
    return torch.cuda.get_device_capability(0)


def _torch_cuda_arch_flags() -> set[int]:
    get_arch_flags = getattr(torch._C, "_cuda_getArchFlags", None)
    if get_arch_flags is None:
        return set()

    arch_flags = set()
    for flag in get_arch_flags().split():
        if flag.startswith("sm_") and flag[3:].isdigit():
            arch_flags.add(int(flag[3:]))
    return arch_flags


def _assert_torch_supports_current_gpu() -> None:
    capability = _cuda_capability()
    if capability is None:
        return

    arch_flags = _torch_cuda_arch_flags()
    if not arch_flags:
        return

    cc = capability[0] * 10 + capability[1]
    compatible_arches = {arch for arch in arch_flags if arch // 10 == cc // 10 and arch <= cc}
    if compatible_arches:
        return

    device_name = torch.cuda.get_device_name(0)
    supported = ", ".join(f"sm_{arch}" for arch in sorted(arch_flags))
    raise RuntimeError(
        f"Installed PyTorch {torch.__version__} was not compiled for {device_name} "
        f"(CUDA compute capability {capability[0]}.{capability[1]}, sm_{cc}). "
        f"This build supports: {supported}. Install a PyTorch build that includes sm_{cc}, "
        "or run on a newer GPU supported by this build."
    )


def _is_single_device(devices: Any) -> bool:
    if devices == 1 or devices == "1":
        return True
    if isinstance(devices, (list, tuple)) and len(devices) == 1:
        return True
    return False


def _patch_trainer_for_cuda_compat(cfg: DictConfig) -> None:
    trainer_cfg = cfg.get("trainer")
    if trainer_cfg is None:
        return

    capability = _cuda_capability()

    with open_dict(trainer_cfg):
        precision = str(trainer_cfg.get("precision", ""))
        if capability is not None and capability < (8, 0) and "bf16" in precision:
            device_name = torch.cuda.get_device_name(0)
            log.warning(
                f"{device_name} has CUDA capability {capability[0]}.{capability[1]} and does not support "
                "native bf16. Switching trainer.precision from bf16-mixed to 16-mixed."
            )
            trainer_cfg.precision = "16-mixed"

        strategy = trainer_cfg.get("strategy")
        if _is_single_device(trainer_cfg.get("devices")) and isinstance(strategy, str) and strategy.startswith("ddp"):
            log.warning(
                f"trainer.devices={trainer_cfg.devices} does not need DDP. Switching trainer.strategy "
                f"from {strategy} to auto."
            )
            trainer_cfg.strategy = "auto"


@task_wrapper
def train(cfg: DictConfig) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Trains the model. Can additionally evaluate on a testset, using best weights obtained during
    training.

    This method is wrapped in optional @task_wrapper decorator, that controls the behavior during
    failure. Useful for multiruns, saving info about the crash, etc.

    :param cfg: A DictConfig configuration composed by Hydra.
    :return: A tuple with metrics and dict with all instantiated objects.
    """
    if cfg.get("run_ref"):
        resolved = resolve_experiment_reference(
            cfg.run_ref,
            registry_path=cfg.get("experiment_registry", ".agents/EXPERIMENTS.md"),
            checkpoint_preference=cfg.get("checkpoint_preference", "best"),
        )
        with open_dict(cfg):
            cfg.ckpt_path = str(resolved.checkpoint)
        log.info(f"Resolved run reference {cfg.run_ref!r} to {cfg.ckpt_path}")

    # set seed for random number generators in pytorch, numpy and python.random
    if cfg.get("seed"):
        L.seed_everything(cfg.seed, workers=True)

    log.info(f"Instantiating datamodule <{cfg.data._target_}>")
    datamodule = hydra.utils.instantiate(cfg.data)

    log.info(f"Instantiating model <{cfg.model._target_}>")
    model: LightningModule = hydra.utils.instantiate(cfg.model)
    # model.xp = get_xp()

    log.info("Instantiating callbacks...")
    callbacks: List[Callback] = instantiate_callbacks(cfg.get("callbacks"))
    log.info(f"Callbacks: {callbacks}")

    log.info("Instantiating loggers...")
    logger: List[Logger] = instantiate_loggers(cfg.get("logger"))

    _patch_trainer_for_cuda_compat(cfg)

    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(cfg.trainer, logger=logger, callbacks=callbacks)

    object_dict = {
        "cfg": cfg,
        "datamodule": datamodule,
        "model": model,
        "callbacks": callbacks,
        "logger": logger,
        "trainer": trainer,
    }

    if logger:
        log.info("Logging hyperparameters!")
        log_hyperparameters(object_dict)

    # automatically resume from latest checkpoint if exists and ckpt_path not manually specified
    # TODO: discuss cfg.resume, this is anti-dora but maybe it's useful
    ckpt_path = cfg.get("ckpt_path")
    cfg.resume = cfg.resume or os.environ.get("USE_MPI")

    if '/opt/ml/' in cfg.paths.ckpt_dir:
        was_s3 = True
    else:
        was_s3 = False


    logging.info("="*100)
    logging.info(os.listdir('/opt/ml/input/data')) if os.path.exists('/opt/ml/input/data') else logging.info("No data found in /opt/ml/input/data")
    logging.info("="*100)


    if os.path.exists(cfg.paths.ckpt_dir) and cfg.resume:
        candidates = [os.path.join(cfg.paths.ckpt_dir, ckpt_file) for ckpt_file in os.listdir(cfg.paths.ckpt_dir) if ckpt_file.endswith(".ckpt")]
        if candidates:
            # get the last modified ckpt else get last.ckpt, reason is that s3 downloads are not in order of creation
            # ckpt_path = max(candidates, key=os.path.getmtime) if not was_s3 else 

            if was_s3:
                ckpt_path = os.path.join(cfg.paths.ckpt_dir, "last.ckpt")
                if "last.ckpt" not in os.listdir(cfg.paths.ckpt_dir):
                    log.warning("last.ckpt not found in s3 ckpt_dir. Training from scratch!")
                    ckpt_path = None
            else:
                ckpt_path = max(candidates, key=os.path.getmtime)
                log.info(f"Resuming from checkpoint {ckpt_path}...")

            # ckpt_path = os.path.join(cfg.paths.ckpt_dir, "last.ckpt") if "last.ckpt" in os.listdir(cfg.paths.ckpt_dir) else None
            log.info(f"Resuming from checkpoint {ckpt_path}...")
        else:
            log.info(ckpt_path, "is empty. Training from scratch!")

    if cfg.get("train"):
        log.info("Starting training!")
        capability = _cuda_capability()
        enable_flash = capability is None or capability >= (8, 0)
        with torch.backends.cuda.sdp_kernel(enable_flash=enable_flash, enable_math=True, enable_mem_efficient=True):
            trainer.fit(model=model, datamodule=datamodule, ckpt_path=ckpt_path)

    train_metrics = trainer.callback_metrics

    if cfg.get("test"):
        log.info("Starting testing!")
        if cfg.get("train"):
            checkpoint_callback = trainer.checkpoint_callback
            ckpt_path = checkpoint_callback.best_model_path if checkpoint_callback is not None else None
            if ckpt_path == "":
                log.warning("Best ckpt not found! Using current weights for testing...")
                ckpt_path = None
        trainer.test(model=model, datamodule=datamodule, ckpt_path=ckpt_path)
        log.info(f"Best ckpt path: {ckpt_path}")

    test_metrics = trainer.callback_metrics

    # merge train and test metrics
    metric_dict = {**train_metrics, **test_metrics}

    return metric_dict, object_dict

    return {}, object_dict


@hydra_main(version_base="1.3", config_path="../configs", config_name="train.yaml")
def main(cfg: DictConfig) -> Optional[float]:
    """Main entry point for training.

    :param cfg: DictConfig configuration composed by Hydra.
    :return: Optional[float] with optimized metric value.
    """
    # Stamp the Dora XP sig with a Unix timestamp so that two launches with the
    # same config produce distinct experiment folders, W&B runs, and save dirs.
    # The ${dora:xp.sig} / ${dora:xp.folder} resolvers are lazy, so patching
    # xp.sig here propagates to every config interpolation that uses them.
    import time as _time
    xp = get_xp()
    _ts = str(int(_time.time()))
    xp.sig = f"{xp.sig}_{_ts}"
    xp.delta = [("name", cfg.name), ("ts", _ts)]

    # handle A100 GPUs
    if torch.cuda.is_available() and ("A100" in torch.cuda.get_device_name() or "A5000" in torch.cuda.get_device_name()):
        torch.set_float32_matmul_precision("high")

    # avoid annoying multiprocessing errors
    torch.multiprocessing.set_sharing_strategy('file_system')

    # prevent annoying warning
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    _assert_torch_supports_current_gpu()

    # apply extra utilities
    # (e.g. ask for tags if none are provided in cfg, print cfg tree, etc.)
    extras(cfg)

    # train the model
    metric_dict, _ = train(cfg)

    # safely retrieve metric value for hydra-based hyperparameter optimization
    metric_value = get_metric_value(
        metric_dict=metric_dict, metric_name=cfg.get("optimized_metric")
    )

    # return optimized metric
    return metric_value


if __name__ == "__main__":
    main()
