import os

from configs import ExperimentConf
from experiment import experiment_process
from hydra_zen import launch

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="1,0"  # sets visible CUDA devices and their order

jobs = launch(
    ExperimentConf,
    experiment_process,
    job_name="experiment",
    overrides=[
        # Configure ray.init() and ray.remote()
        "+hydra.launcher.ray.init.address=auto",
        "+hydra.launcher.ray.remote.num_gpus=0.2",  # https://docs.ray.io/en/latest/ray-core/tasks/using-ray-with-gpus.html#fractional-gpus
        "+hydra.launcher.ray.remote.max_calls=1",  # forces GPU tasks to release resources after finishing
        # Experiment sweeps
        "+seed=0",
        "+env=Ant,HalfCheetah,Hopper,Humanoid,Walker2d",
        "+algo=SAC-ANN,TD3-ANN",
    ],
    multirun=True,
    version_base=None,  # FIXME: The version_base parameter is not specified. Please specify a compatability version level, or None. Will assume defaults for version 1.1
)
