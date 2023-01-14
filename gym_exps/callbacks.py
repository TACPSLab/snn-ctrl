import logging
from typing import Any
from omegaconf import DictConfig
from hydra.core.utils import JobReturn, JobStatus
from hydra.experimental.callback import Callback


class LogJobReturnCallback(Callback):
    """
    Log the job's return value or error upon job end

    Ray jobs exit without traceback info. This is a workaround mentioned in
    https://github.com/facebookresearch/hydra/issues/1698
    """

    def __init__(self) -> None:
        self.log = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    def on_job_end(
        self, config: DictConfig, job_return: JobReturn, **kwargs: Any
    ) -> None:
        if job_return.status == JobStatus.COMPLETED:
            self.log.info(f"Succeeded with return value: {job_return.return_value}")
        elif job_return.status == JobStatus.FAILED:
            self.log.error("", exc_info=job_return._return_value)
        else:
            self.log.error("Status unknown. This should never happen.")
