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
from src.abc_class.trainer import DDPTrainer


class GPT2Dataset(Dataset):
    def __init__(self, training: bool, dataframe: pd.DataFrame, need_column: List[str], tokenizer):
        super().__init__()
        self.training = training
        self.df = dataframe[need_column]
        self.tokenizer = tokenizer
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        
        x = self.df.iloc[idx, :]
    
        
        x = self.tokenizer(x.values[0], 
                        padding='max_length', 
                        truncation=True, 
                        max_length= SEQ_MAX_LENGTH, 
                        return_tensors='pt')
        
        x = {k:v.squeeze() for k, v in x.items()}
        
        
        y = x['input_ids']
        shift_y = y[..., 1:].contiguous()
       

        if self.training:
            return x, shift_y

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
        
        self.optimizer = torch.optim.Adam(self.model.module.parameters(), lr=initial_lr)
        self.lr_scheduler = get_cosine_schedule_with_warmup(
                                                            optimizer=self.optimizer,
                                                            num_warmup_steps=warm_up_step,
                                                            num_training_steps=num_training_steps
                                                             )
        
       
        
        
    def train_step(self, X_batch, y_batch):
        X_batch = {k : v.to(self.device) for k, v in X_batch.items()}
        y_batch = y_batch.to(self.device)
        
        logits = self.model.forward(X_batch)
        shift_logits = logits[..., :-1, :].contiguous()
        shift_logits  = shift_logits.view(-1, shift_logits.size(-1))
        target = y_batch.view(-1)
        
        # ignore ['PAD']
        loss_fn = torch.nn.CrossEntropyLoss(ignore_index=0, reduction='sum')
 
        loss = loss_fn(shift_logits, target)
        
        not_ignore = target.ne(0)
        num_targets = not_ignore.long().sum().item()
        loss = loss / num_targets
    

        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        self.lr_scheduler.step()

        
        return loss, None
    

def main():
    train = pd.read_csv('./storage/data/chinese_chatbot/chatbot.csv')


    dist.init_process_group(backend='nccl')
    dist.barrier()

    EPOCHS = 30
    BS = 16
    initial_lr = 1e-5
    warm_up = 5
    local_rank = int(os.environ["LOCAL_RANK"])

    tokenizer = BertTokenizerFast.from_pretrained('./save/')




    train_dataset = GPT2Dataset(training=True, dataframe=train, need_column=['context'], tokenizer=tokenizer)
    train_sampler = DistributedSampler(train_dataset)
    train_loader = DataLoader(train_dataset, batch_size=BS, sampler=train_sampler)


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
                    checkpoint_path=f'/gcs/pchome-hadoopincloud-hadoop/user/stevenchiou/tmp/test/ckpt_ep_{EPOCHS}_bs_{BS}',
                    tensorboard_path='/gcs/pchome-hadoopincloud-hadoop/user/stevenchiou/tmp/test/tb/train/',
                    local_rank=local_rank,
                    sampler=train_sampler
                        )

if __name__ == '__main__':

    main()