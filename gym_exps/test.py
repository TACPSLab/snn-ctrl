import logging
import os

import ray
from hydra_zen import make_config, MISSING, launch

log = logging.getLogger(__name__)

TestConf = make_config(
    defaults=[
        "_self_",
        {"override /hydra/launcher": "ray"},
    ],
    task=MISSING,
)

def test(cfg):
    log.info(f"Task: {cfg.task}")
    log.info(f"ray.get_gpu_ids(): {ray.get_gpu_ids()}")
    log.info(f"CUDA_VISIBLE_DEVICES: {os.environ['CUDA_VISIBLE_DEVICES']}")

jobs = launch(
    TestConf,
    test,
    job_name="test",
    overrides=[
        # Configure Hydra
        "hydra.sweep.dir=${hydra.job.name}s/${now:%Y-%m-%d_%H-%M-%S}",
        "hydra.sweep.subdir=${hydra.job.override_dirname}",
        "hydra.job_logging.handlers.file.filename=${hydra.runtime.output_dir}/${hydra.job.name}.log",
        # Configure ray.init() and ray.remote()
        # "+hydra.launcher.ray.init.num_gpus=2",
        "+hydra.launcher.ray.remote.num_gpus=0.5",
        # Configure Experiments
        "+task=1,2,3,4,5,6,7",
    ],
    multirun=True,
    version_base=None,
    to_dictconfig=True,
)
