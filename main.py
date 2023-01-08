from typing import Callable, List, Dict, Callable
from tqdm import tqdm

import numpy as np
import pandas as pd

import torch
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2LMHeadModel, BertTokenizerFast, get_cosine_schedule_with_warmup


from src.train.tensorboard import TensorBoard 
from src.model.config import SEQ_MAX_LENGTH
from src.model.gpt import GPT2
from src.train.metric import NLGMetric


from prepare_data import *
from src.data.preprocessor import IPreprocessor, GPT2Preprocessor
from src.abc_class.trainer import Trainer

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


class GPT2Trainer(Trainer):
    def __init__(self, model, metric, val_dataloader, num_training_steps):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(f'using device: {self.device}')
        super().__init__()
        self.model = model.to(self.device)
        self.metric = metric
        self.val_dataloader = val_dataloader
        
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        self.lr_scheduler = get_cosine_schedule_with_warmup(
                                                            optimizer=self.optimizer,
                                                            num_warmup_steps=3000,
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

        
        return loss
    
        
    def validation_loop(self, batch_size, epoch):
        VALSTEP_PER_EPOCH = len(self.val_dataloader) 
        val_writer = SummaryWriter('./storage/tensorboard/val/')
        ValTB = TensorBoard(val_writer)
        val_metric = self.metric
        self.model.eval()
        val_running_loss = 0
        with torch.no_grad():
            for X_val, y_val in self.val_dataloader:
                X_val = {k : v.to(self.device) for k, v in X_val.items()}
                y_val = {k : v.to(self.device) for k, v in y_val.items()}
                
                logits = self.model.forward(X_val)
                val_logits = logits.view(-1, logits.size(-1))
                target = y_val['input_ids'].squeeze().view(-1)
                
                val_loss = torch.nn.CrossEntropyLoss()(val_logits, target)
                val_running_loss += val_loss.item()
            epoch_val_loss = val_running_loss / VALSTEP_PER_EPOCH
            print("="*30)
            print(f"Validation Loss : {epoch_val_loss}")
            val_metric.calculate_metric(epoch_val_loss)
            for name, result in val_metric.get_result().items():
                print(f'Validation {name} over epoch : {float(result)}')
            print('='*30)
            ValTB.start_to_write(metrics_result=val_metric.get_result(),
                                   step=epoch)
            val_metric.reset()


            
            
df = pd.read_csv('./storage/data/ppt.csv')


tokenizer = BertTokenizerFast.from_pretrained('./save/')

split_point = int(len(df) * 0.9)
train, val = df.iloc[: split_point],  df.iloc[split_point:]

p = GPT2Preprocessor(tokenizer=tokenizer, eos_token='[EOS]', bos_token='[BOS]')
train_dataset = GPT2Dataset(training=True, dataframe=train, need_column=['context'], preprocessor=p)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

val_dataset = GPT2Dataset(training=True, dataframe=val, need_column=['context'], preprocessor=p)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=True)


EPOCHS = 10
NUM_TRAINING_STEPS = EPOCHS * len(train_loader)
model = GPT2(tokenizer)
mygpt = GPT2Trainer(model, NLGMetric(), val_loader, num_training_steps=NUM_TRAINING_STEPS)
mygpt.start_to_train(
                train_data_loader=train_loader,
                batch_size=64,
                epochs=EPOCHS,
                checkpoint_path='./storage/ckpt/',
                tensorboard_path='./storage/tensorboard/train/',
                    )
