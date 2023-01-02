
class EarlyStopping:
    
    def __init__(self, tolerance: int) -> None:
        self.tolerance = tolerance
        self.patient = 0
        self.pre_val_loss = 0

    def __call__(self, val_loss) -> None:

        if val_loss <= self.pre_val_loss:
            self.patient = 0
        
        else:
            self.patient += 1

        if self.patient == self.tolerance:
            return True
        return False