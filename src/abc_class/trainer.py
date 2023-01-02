import os
import time
from abc import ABCMeta, abstractmethod
from typing import Callable, List, Optional, Dict, Callable
from tqdm import tqdm

import numpy as np
import dill
import torch
from torch.utils.tensorboard import SummaryWriter
import torch.distributed as dist



from src.abc_class.metric import AbcMetric
from src.train.tensorboard import TensorBoard 
from src.model.config import SEQ_MAX_LENGTH


class Trainer(metaclass=ABCMeta):

    def __init__(self) -> None:

        self.model = None

        self.metric = None

        self.optimizer = None
        self.lr_scheduler = None



    @abstractmethod
    def train_step(self, data_batch):
        """define the training step  per batch"""
        raise NotImplemented("not implemented")


    def restore_checkpoint(self) -> torch.nn:

        return None

    def validation_loop(self, epoch):
        return None

    def start_to_train(self, 
                       train_data_loader,
                       batch_size: int,
                       epochs: int,
                       checkpoint_path: str,
                       tensorboard_path: str,
                       earlystopping_tolerance: Optional[int] = None
                       ):

        self._count_total_parameters()


        writer = SummaryWriter(tensorboard_path)
        
        assert self.model != None, 'model is not set, such as self.model=YOUR MODEL'
        assert self.optimizer != None, 'optimizer is nit set, such as self.optimizer=YOUR OPTIMIZER'
        
        if earlystopping_tolerance is not None:
            earlystop = EarlyStopping(earlystopping_tolerance)
        else:
            earlystop = None
            print('Training without Earlystopping')

        # global_step = 0
        TRAINSTEP_PER_EPOCH = len(train_data_loader) 
        train_metric = self.metric
        for epoch in range(epochs):
            epoch_start = time.time()
            running_loss = 0
            print(f'EPOCH {epoch + 1} start')
            print('='*50)
            
            self.model.train()

            for X_batch, y_batch in tqdm(train_data_loader, total=TRAINSTEP_PER_EPOCH):


                loss = self.train_step(X_batch=X_batch,
                                       y_batch=y_batch, 
                                        )
                running_loss += loss.item()
                    

            epoch_loss = running_loss / TRAINSTEP_PER_EPOCH

            print(f"Epoch Loss :  {epoch_loss}")

            if train_metric is not None:
                train_metric.calculate_metric(epoch_loss)
                for name, result in train_metric.get_result().items():
                    print(f'Training {name} over epoch : {float(result)}')
                    print('=' *30)
                train_metric.reset()


            TrainTB = TensorBoard(writer=writer,
                                  model=self.model)
            
            metrics_result = train_metric if train_metric is None else train_metric.get_result()
            TrainTB.start_to_write(metrics_result=metrics_result,
                                   step=epoch,
                                   loss=epoch_loss,
                                   histogram=True,
                                   optimizer=self.optimizer)

                


            if self.validation_loop is not None:
                val_loss = self.validation_loop(batch_size, epoch)
                if earlystop is not None:
                    if earlystop(val_loss=val_loss):
                        print(f'EarlyStopping during epoch {epoch + 1}')
                        break


            print(f'saving model for epoch {epoch + 1}')
            model_path = checkpoint_path + f'model_epoch{epoch}'
            if not 'gcs' in checkpoint_path:
                if not os.path.exists(checkpoint_path):
                    os.makedirs(checkpoint_path)
            
            # https://zhuanlan.zhihu.com/p/136902153
            checkpoint = {#'model': self.model,
                         'model_state_dict': self.model.state_dict(),
                         'optimizer_state_dict': self.optimizer.state_dict(),
                         'epoch': epoch}
            if self.lr_scheduler is not None:
                checkpoint.update({'lr_scheduler': self.lr_scheduler})
            
            torch.save(checkpoint, f'{model_path}.pkl', pickle_module=dill)



            print('epoch {} finished'.format(epoch + 1))

            epoch_end = time.time()
            print(f'time for one epoch: {epoch_end - epoch_start} secconds')


        print('Finish training')

    def _count_total_parameters(self):
        num_parameters = 0
        parameters = self.model.parameters()
        for parameter in parameters:
            num_parameters += parameter.numel()
        print(f'number of parameters: {num_parameters}')
        



class DDPTrainer(metaclass=ABCMeta):

    def __init__(self) -> None:

        self.model = None

        self.metric = None

        self.optimizer = None
        self.lr_scheduler = None


        self.world_size = dist.get_world_size()



    @abstractmethod
    def train_step(self, data_batch):
        """define the training step  per batch"""
        raise NotImplemented("not implemented")


    def validation_loop(self, epoch):
        return None

    def start_to_train(self, 
                       train_data_loader,
                       epochs: int,
                       checkpoint_path: str,
                       tensorboard_path: str,
                       local_rank: int,
                       sampler,
                       earlystopping_tolerance: Optional[int] = None,
                       ):

        assert self.model != None, 'model is not set, such as self.model=YOUR MODEL'
        assert self.optimizer != None, 'optimizer is nit set, such as self.optimizer=YOUR OPTIMIZER'

        if dist.get_rank() == 0: # master rank
            self._count_total_parameters()
            writer = SummaryWriter(tensorboard_path)

        TRAINSTEP_PER_EPOCH = len(train_data_loader) 
        train_metric = self.metric
        for epoch in range(epochs):
            epoch_start = time.time()
            running_loss = 0
            print(f'Train_local_rank:{local_rank}, Train epoch:{epoch + 1} start training')
            print('='*60)

            sampler.set_epoch(epoch)
            self.model.train()
            
            for X_batch, y_batch in train_data_loader:

                loss, y_pred = self.train_step(X_batch=X_batch,
                                               y_batch=y_batch, 
                                              )
                running_loss += loss.item()


            rank_epoch_loss = running_loss / TRAINSTEP_PER_EPOCH
            print(f'===== rank : {dist.get_rank()}=====')
            print(f"rank_{dist.get_rank()} Epoch loss : {rank_epoch_loss}")
            
            
            if self.validation_loop is not None:
                val_loss = self.validation_loop(epoch)

            sum_rank_epoch_loss = self.reduce_sum_all(rank_epoch_loss)
            epoch_loss = sum_rank_epoch_loss.cpu().numpy() / self.world_size
            
            if dist.get_rank() == 0:
                
                if train_metric is not None:
                    
                    train_metric.calculate_metric(epoch_loss)
                    

               
                TrainTB = TensorBoard(writer=writer,
                                    model=self.model)
                metrics_result = train_metric if train_metric is None else train_metric.get_result()
                TrainTB.start_to_write(metrics_result=metrics_result,
                                   step=epoch,
                                   loss=epoch_loss,
                                   histogram=True,
                                   optimizer=self.optimizer)

                if train_metric is not None:
                    for name, result in train_metric.get_result().items():
                        if isinstance(result, (int, float)):
                            print(f'Training {name} over epoch on rank 0 : {float(result)}')
                            print('=' *50)
                    train_metric.reset()


                print(f'saving model for epoch {epoch + 1}')
                model_path = checkpoint_path + f'model_epoch{epoch + 1}'
                if not 'gcs' in checkpoint_path:
                    if not os.path.exists(checkpoint_path):
                        os.makedirs(checkpoint_path)
                
                # https://zhuanlan.zhihu.com/p/136902153
                checkpoint = {#'model': self.model,
                            'model_state_dict': self.model.state_dict(),
                            'optimizer_state_dict': self.optimizer.state_dict(),
                            'epoch': epoch}
                if self.lr_scheduler is not None:
                    checkpoint.update({'lr_scheduler': self.lr_scheduler})
                
                torch.save(checkpoint, f'{model_path}.pkl', pickle_module=dill)
                        
            dist.barrier()


            
            print(f'rank {dist.get_rank()} epoch {epoch + 1} finished')
            epoch_end = time.time()
            print(f'time for one rank{dist.get_rank()}_epoch: {epoch_end - epoch_start} secconds')


        print('Finish training')

                


    def _count_total_parameters(self):
        num_parameters = 0
        parameters = self.model.parameters()
        for parameter in parameters:
            num_parameters += parameter.numel()
        print(f'number of parameters: {num_parameters}')




    def gather_all(self, value_tensor):
        tensor_list = [torch.zeros_like(value_tensor).cuda() for _ in range(self.world_size)]
        dist.all_gather(tensor_list, value_tensor)
        return tensor_list


    def reduce_sum_all(self, value_tensor):
        if not isinstance(value_tensor, torch.Tensor):
            value_tensor = torch.tensor(value_tensor).to(dist.get_rank())
        
        dist.all_reduce(value_tensor, op=dist.ReduceOp.SUM)

        return value_tensor


