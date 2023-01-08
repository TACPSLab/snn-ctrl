import os

from configs import ExperimentConf
from experiment import experiment
from hydra_zen import launch

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="1,0"  # sets visible CUDA devices and their order

jobs = launch(
    ExperimentConf,
    experiment,
    job_name="experiment",
    overrides=[
        # Configure Hydra
        "hydra.sweep.dir=${hydra.job.name}s/${now:%Y-%m-%d_%H-%M-%S}",
        "hydra.sweep.subdir=${hydra.job.override_dirname}",
        "hydra.job_logging.handlers.file.filename=${hydra.runtime.output_dir}/${hydra.job.name}.log",
        # Configure ray.init() and ray.remote()
        "+hydra.launcher.ray.remote.num_gpus=0.2",  # https://docs.ray.io/en/latest/ray-core/tasks/using-ray-with-gpus.html#fractional-gpus
        "+hydra.launcher.ray.remote.max_calls=1",  # forces GPU tasks to release resources after finishing
        # Configure Experiments
        "+env=Ant,HalfCheetah,Hopper,Humanoid,Walker2d",
        "+algo=SAC,TD3",
    ],
    multirun=True,
    version_base=None,  # FIXME: The version_base parameter is not specified. Please specify a compatability version level, or None. Will assume defaults for version 1.1
    to_dictconfig=True,  # FIXME: UserWarning: Your dataclass-based config was mutated by this run. If you just executed with a `hydra/launcher` that utilizes cloudpickle (e.g., hydra-submitit-launcher), there is a known issue with dataclasses (see: https://github.com/cloudpipe/cloudpickle/issues/386). You will have to restart your interactive environment ro run `launch` again. To avoid this issue you can use the `launch` option: `to_dictconfig=True`.
)
