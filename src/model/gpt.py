from typing import Optional

from transformers import GPT2LMHeadModel
import torch


from src.model.config import PRETRAIN



class GPT2(torch.nn.Module):
    
    def __init__(self, tokenizer) -> None:
        super().__init__()
        self.gpt2 = GPT2LMHeadModel.from_pretrained(PRETRAIN)
        self.gpt2.resize_token_embeddings(len(tokenizer))

        

    def forward(self, inputs):
        
        logits = self.gpt2(**inputs).logits


        return logits



