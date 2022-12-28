from configs import ExperimentConf
from experiment import experiment
from hydra_zen import launch

jobs = launch(
    ExperimentConf,
    experiment,
    job_name="experiment",
    overrides=[
        # Configuring Hydra
        "hydra.sweep.dir=${hydra.job.name}s/${now:%Y-%m-%d_%H-%M-%S}",
        "hydra.sweep.subdir=${hydra.job.override_dirname}",
        "hydra.job_logging.handlers.file.filename=${hydra.runtime.output_dir}/${hydra.job.name}.log",
        # Configuring Experiments
        "+env=Ant,HalfCheetah,Hopper,InvertedDoublePendulum",
        "+algo=TD3",
        "+policy=ANN",
    ],
    multirun=True,
    version_base=None,
)
