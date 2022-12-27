from hydra_zen import launch

from configs import ExperimentConf
from train import train

jobs = launch(
    ExperimentConf,
    train,
    overrides=[
        # "hydra.sweep.subdir=${hydra.job.override_dirname}", FIXME https://github.com/facebookresearch/hydra/issues/1832#issuecomment-1012066935
        "+env=Ant,HalfCheetah,Hopper,InvertedDoublePendulum",
        "+algo=TD3",
        "+algo/policy=ANN",
    ],
    multirun=True,
    version_base=None,
)
