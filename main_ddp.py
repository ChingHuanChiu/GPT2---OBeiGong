import os
from typing import Callable, List, Dict, Callable
from tqdm import tqdm

import numpy as np
import pandas as pd

import torch
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2LMHeadModel, BertTokenizerFast, get_cosine_schedule_with_warmup
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist

from src.train.tensorboard import TensorBoard 
from src.model.config import SEQ_MAX_LENGTH
from src.model.gpt import GPT2
from src.train.metric import NLGMetric


from prepare_data import *
from src.data.preprocessor import IPreprocessor, GPT2Preprocessor
from src.abc_class.trainer import DDPTrainer


class GPT2Dataset(Dataset):
    def __init__(self, training: bool, dataframe: pd.DataFrame, need_column: List[str], preprocessor: IPreprocessor = None):
        super().__init__()
        self.training = training
        self.preprocessor = preprocessor
        self.df = dataframe[need_column]
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        
        x = self.df.iloc[idx, :]
        
        x, y = self.preprocessor.transform(x.values[0])
        x = {k:v.squeeze() for k, v in x.items()}
        y = {k:v.squeeze() for k, v in y.items()}
        
        if self.training:
            return x, y
        
        return x


class GPT2Trainer(DDPTrainer):
    def __init__(self, model, metric, initial_lr, num_training_steps, local_rank, warm_up_step, val_dataloader=None):
        super().__init__()

        self.device = torch.device("cuda", local_rank)
        print(f'using device: {self.device}')
        self.model = model.to(self.device)
        
        self.model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.model)
        self.model = torch.nn.parallel.DistributedDataParallel(self.model, 
                                                               device_ids=[local_rank],
                                                                output_device=local_rank)
        self.metric = metric
        # self.val_dataloader = val_dataloader
        
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=initial_lr)
        self.lr_scheduler = get_cosine_schedule_with_warmup(
                                                            optimizer=self.optimizer,
                                                            num_warmup_steps=warm_up_step,
                                                            num_training_steps=num_training_steps
                                                             )
        
       
        
        
    def train_step(self, X_batch, y_batch):
        X_batch = {k : v.to(self.device) for k, v in X_batch.items()}
        y_batch = {k : v.to(self.device) for k, v in y_batch.items()}
        logits = self.model.forward(X_batch)
        logits = logits.view(-1, logits.size(-1))
        target = y_batch['input_ids'].squeeze().view(-1)


        loss = torch.nn.CrossEntropyLoss()(logits, target)
        
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        self.lr_scheduler.step()

        
        return loss, None
    
        
#     def validation_loop(self, epoch):
#         VALSTEP_PER_EPOCH = len(self.val_dataloader)
        
        
#         val_metric = self.metric
#         self.model.eval()
#         val_running_loss = 0
#         with torch.no_grad():
#             for X_val, y_val in self.val_dataloader:
#                 X_val = {k : v.to(self.device) for k, v in X_val.items()}
#                 y_val = {k : v.to(self.device) for k, v in y_val.items()}
                
#                 logits = self.model.forward(X_val)
#                 val_logits = logits.view(-1, logits.size(-1))
#                 target = y_val['input_ids'].squeeze().view(-1)
                
#                 val_loss = torch.nn.CrossEntropyLoss()(val_logits, target)
#                 val_running_loss += val_loss.item()
                
#             rank_epoch_val_loss = val_running_loss / VALSTEP_PER_EPOCH
#             sum_rank_val_epoch_loss = self.reduce_sum_all(rank_epoch_val_loss)
#             epoch_val_loss = sum_rank_val_epoch_loss.cpu().numpy() / self.world_size
#             print("="*30)
#             print(f"Validation Loss : {epoch_val_loss}")
            
#             if dist.get_rank() == 0:
#                 val_writer = SummaryWriter('./storage/tensorboard/val/')
#                 ValTB = TensorBoard(val_writer)
#                 val_metric.calculate_metric(epoch_val_loss)
#                 for name, result in val_metric.get_result().items():
#                     print(f'Validation {name} over epoch : {float(result)}')
#                 print('='*30)
#                 ValTB.start_to_write(metrics_result=val_metric.get_result(),
#                                        step=epoch)
#                 val_metric.reset()
#             dist.barrier()

def main():
    
    train = pd.read_csv('./storage/data/train_data/chatbot_baike.csv')


    dist.init_process_group(backend='nccl')
    dist.barrier()

    EPOCHS = 50
    BS = 2
    initial_lr = 1e-5
    warm_up = 1000
    local_rank = int(os.environ["LOCAL_RANK"])

    tokenizer = BertTokenizerFast.from_pretrained('./save/')



    p = GPT2Preprocessor(tokenizer=tokenizer, eos_token='[EOS]', bos_token='')
    train_dataset = GPT2Dataset(training=True, dataframe=train, need_column=['context'], preprocessor=p)
    train_sampler = DistributedSampler(train_dataset)
    train_loader = DataLoader(train_dataset, batch_size=BS, sampler=train_sampler)

    # val_dataset = GPT2Dataset(training=True, dataframe=val, need_column=['context'], preprocessor=p)
    # val_sampler = DistributedSampler(val_dataset)
    # val_loader = DataLoader(val_dataset, batch_size=BS, sampler=val_sampler)

    NUM_TRAINING_STEPS = EPOCHS * len(train_loader)


    model = GPT2(tokenizer)
    mygpt = GPT2Trainer(model=model, 
                        metric=NLGMetric(), 
                        initial_lr=initial_lr, 
                        num_training_steps=NUM_TRAINING_STEPS, 
                        local_rank=local_rank, 
                        warm_up_step=warm_up)
    
    mygpt.start_to_train(
                    train_data_loader=train_loader,
                    epochs=EPOCHS,
                    checkpoint_path=f'./storage/ckpt_ep_{EPOCHS}_bs_{BS}/',
                    tensorboard_path=f'/gcs/pchome-hadoopincloud-hadoop/user/stevenchiou/tmp/test/tb/train/',
                    local_rank=local_rank,
                    sampler=train_sampler
                        )

if __name__ == '__main__':

    main()