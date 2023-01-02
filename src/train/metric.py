
from src.abc_class.metric import AbcMetric
import numpy as np

class NLGMetric(AbcMetric):
    
    def __init__(self):
        self.ppl = None

    def calculate_metric(self, ce_loss) -> None:
        self.ppl = np.exp(ce_loss)

    def reset(self):
        pass
        



    