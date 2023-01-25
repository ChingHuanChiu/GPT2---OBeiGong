#ref : https://github.com/Morizeyao/GPT2-Chinese/blob/old_gpt_2_chinese_before_2021_4_22/generate.py

import torch
import torch.nn.functional as F
import os
import argparse
from tqdm import trange
from transformers import BertTokenizer, GPT2LMHeadModel, TextGenerationPipeline, BertTokenizerFast

from src.model.util import load_model_from_checkpoint



def is_word(word):
    for item in list(word):
        if item not in 'qwertyuiopasdfghjklzxcvbnm':
            return False
    return True

def _is_chinese_char(char):
    """Checks whether CP is the codepoint of a CJK character."""
    # This defines a "chinese character" as anything in the CJK Unicode block:
    #   https://en.wikipedia.org/wiki/CJK_Unified_Ideographs_(Unicode_block)
    #
    # Note that the CJK Unicode block is NOT all Japanese and Korean characters,
    # despite its name. The modern Korean Hangul alphabet is a different block,
    # as is Japanese Hiragana and Katakana. Those alphabets are used to write
    # space-separated words, so they are not treated specially and handled
    # like the all of the other languages.
    cp = ord(char)
    if ((cp >= 0x4E00 and cp <= 0x9FFF) or  #
            (cp >= 0x3400 and cp <= 0x4DBF) or  #
            (cp >= 0x20000 and cp <= 0x2A6DF) or  #
            (cp >= 0x2A700 and cp <= 0x2B73F) or  #
            (cp >= 0x2B740 and cp <= 0x2B81F) or  #
            (cp >= 0x2B820 and cp <= 0x2CEAF) or
            (cp >= 0xF900 and cp <= 0xFAFF) or  #
            (cp >= 0x2F800 and cp <= 0x2FA1F)):  #
        return True

    return False


def top_k_top_p_filtering(logits, top_k=0, top_p=0.0, filter_value=-float('Inf')):
    """ Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
        Args:
            logits: logits distribution shape (vocabulary size)
            top_k > 0: keep only top k tokens with highest probability (top-k filtering).
            top_p > 0.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
                Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
        From: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
    """
    assert logits.dim() == 1  # batch size 1 for now - could be updated for more but the code would be less clear
    top_k = min(top_k, logits.size(-1))  # Safety check
    if top_k > 0:
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = filter_value
    return logits



def sample_sequence(tokenizer, model, inputs_dict, length, temperature=1, top_k=0, top_p=0.0, device='cpu'):
    
    # remove [SEP] token
    for k, v in inputs_dict.items():
        inputs_dict[k] = v[: -1]
        
    context = inputs_dict['input_ids']
    context = torch.tensor(context, dtype=torch.long, device=device)
    context = context.unsqueeze(0)
    generated = context
    with torch.no_grad():
        # for _ in trange(length):
        for _ in range(length):
            inputs = {'input_ids': generated}
            
            outputs = model(inputs)  # Note: we could also use 'past' with GPT-2/Transfo-XL/XLNet (cached hidden-states)

            next_token_logits = outputs[0, -1, :] / temperature
            filtered_logits = top_k_top_p_filtering(next_token_logits, top_k=top_k, top_p=top_p)
            next_token = torch.multinomial(F.softmax(filtered_logits, dim=-1), num_samples=1)
            generated = torch.cat((generated, next_token.unsqueeze(0)), dim=1)
    return generated.tolist()[0]




def generate(tokenizer, model, inputs_dict, length, device, temperature=1, top_k=0, top_p=0.0):
  

    return sample_sequence(tokenizer, model, inputs_dict, length, temperature=temperature, top_k=top_k, top_p=top_p, device=device)

    
def main(question: str, model):
    # os.environ["CUDA_VISIBLE_DEVICES"] = '0'  
    length = 50
    batch_size = 1
    nsamples = 1
    temperature = 0.8
    topk = 30
    topp = 0.95
    


    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = BertTokenizerFast.from_pretrained('./save/')
    model.to(device)

    if length == -1:
        # model.gpt2.config.n_ctx = 1024
        length = model.gpt2.config.n_ctx - len(question)
    elif length > model.gpt2.config.n_ctx - len(question):
        raise ValueError("Can't get samples longer than window size: %s" % model.config.n_ctx)
    
    
    while True:
        raw_text = question
        inputs_dict = tokenizer(question)
        generated = 0
        for _ in range(nsamples // batch_size):
            out = generate(
                tokenizer=tokenizer,
                model=model,
                inputs_dict=inputs_dict,
                length=length,
                temperature=temperature, top_k=topk, top_p=topp, device=device
            )
            for i in range(batch_size):
                generated += 1
                text = tokenizer.convert_ids_to_tokens(out)
                for i, item in enumerate(text[:-1]):  # 确保英文前后有空格
                    if is_word(item) and is_word(text[i + 1]):
                        text[i] = item + ' '
                for i, item in enumerate(text):
                    if item == '[MASK]' or item == '[UNK]':
                        text[i] = ''
                    if item == '[CLS]':
                        text[i] = '\n'
                info = "=" * 40 + " SAMPLE " + str(generated) + " " + "=" * 40 + "\n"
                
                res = []
                for t in text:
                    if t != '[SEP]': res.append(t)
                    else: break
#                 print(res)
#                 input_text = res[: res.index('[QA]') + 1]
#                 input_text= ''.join(input_text).replace('##', '').strip()
                
#                 response_text = res[res.index('[QA]') + 1: ]
                
                response_text = ''.join(res).replace('##', '').strip()
                

        print("=" * 80)
        if generated == nsamples :

            break
    
    return response_text[len(question): ]
    

