import torch
from transformers import BertTokenizerFast

from src.model.util import load_model_from_checkpoint
from src.model.gpt import GPT2
from generate import main


CKPT_PATH = './storage/ckpt/ckpt_ep_50_bs_16model_epoch2.pkl'

tokenizer = BertTokenizerFast.from_pretrained('./save/')
model=GPT2(tokenizer)


model = load_model_from_checkpoint(device='cuda:0', model_ckpt=CKPT_PATH, model=model, is_ddp_model=True)


print('嗨！你好')
while True:
    inp = input('Q: ')

    if inp == '掰掰':
        print('掰掰 !')
        break
    inp += '[SEP]'
    print('Model: ', main(inp, model))
    

        