from typing import Dict, Union, List


class TensorBoard:

    def __init__(self, writer, model=None):
        self.model = model
        self.writer = writer

    def start_to_write(self, metrics_result: Dict[str, Union[int, float]], step, loss=None, histogram=False, optimizer=None):
        
        
        if loss is not None:

            self.writer.add_scalar(tag="loss",
                            scalar_value=loss, 
                            global_step=step
                            )

        if histogram is True:
            for name, param in self.model.named_parameters():
                self.writer.add_histogram(tag=name,
                                          values=param,
                                          global_step=step
                                        )

        for name, result in metrics_result.items():
            self.writer.add_scalar(tag=name,
                            scalar_value=result, 
                            global_step=step
                            )

        if optimizer is not None:

            for idx, lr in enumerate(self._get_lr(optimizer=optimizer), start=1):
                self.writer.add_scalar(tag='lr_' + str(idx),
                                        scalar_value=lr, 
                                        global_step=step
                                        )

        self.writer.close()


    def _get_lr(self, optimizer) -> List[float]:

        if hasattr(optimizer, 'param_groups'):
            lr = []
            for i in optimizer.param_groups:
                lr.append(i['lr'])


        else:
            lr = optimizer.get_last_lr()
            
        return lr
