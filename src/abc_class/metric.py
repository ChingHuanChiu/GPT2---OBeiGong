from typing import Dict, Union
from abc import ABCMeta, abstractmethod


class AbcMetric(metaclass=ABCMeta):

    @abstractmethod
    def calculate_metric(self, **kwargs) -> None:
        raise NotImplemented("not implemented")


    @abstractmethod
    def reset(self):
        raise NotImplemented("not implemented")


    def get_result(self) -> Dict[str, Union[float, int]]:
        """
        return a dictionary of result of metrics
        """
        result = dict()
        for name, metric in self.__dict__.items():

            result[name] = metric

        return result

