import glob
from typing import List


# from opencc import OpenCC
import pandas as pd
import yaml
from tqdm import tqdm





def make_drcd_data(data_json):
    """https://github.com/DRCKnowledgeTeam/DRCD
    """
    context_list, q_a_list = [], []
    for data in data_json['data']:
        paragraphs = data['paragraphs']
        for paragraphs_data in paragraphs:
            context_list.append(paragraphs_data['context'])
            qas = paragraphs_data['qas']
            for _qas in qas:

                # q_a_data = '[BOS]' + _qas['question'] + '[SEP]' + _qas['answers'][0]['text'] + '[EOS]'
                q_a_data = _qas['question'] + '[SEP]' + _qas['answers'][0]['text']
                q_a_list.append(q_a_data)
    return context_list, q_a_list



def make_chinese_chatbot_data() -> List[str]:
    context_with_Q_A = []
    cc = OpenCC('s2t')
    for yaml_path in glob.glob('./storage/data/chinese_chatbot/*.yml'):
        
        try:
            with open(yaml_path, "r") as stream:
                data = yaml.safe_load(stream)
                converstions: List[List[str]] = data['conversations']

                for i in converstions:
                    for idx in range(0,  len(i), 2):
                        try:
                            if i[idx][-1] != '?':
                                i[idx] += '?'
                            context_with_Q_A.append(cc.convert(i[idx]) + cc.convert(i[idx+1]))
                        except:
                            print('length of data is not even', len(i), i)


        except yaml.YAMLError as exc:
            print(exc)
    return context_with_Q_A


def truncate(context: str, max_length: int):
    """truncate the sentence which length is larger than max_length, 
    e.g. 'abc,def,ghi' -> 'abc,def,' and 'def,ghi'
    """
    
    res = []
    punctuation = ['!', '，', '。', '?']
    
    while True:
        segment_context = context[: max_length]
        
        max_index_punctuation = max([segment_context.rfind(punc) for punc in punctuation])
        if max_index_punctuation != -1: 
            segment_context_idx = max_index_punctuation + 1
        else:
            # max_index_punctuation = max_length
            segment_context_idx = max_length
         
        select_segment_context = segment_context[: segment_context_idx]

        res.append(select_segment_context)
        
        
        # find the second to last punctuation in segment_context  
        the_second_to_last_punctuation = max([select_segment_context[: -1].rfind(punc) for punc in punctuation])
        
        if the_second_to_last_punctuation != -1:
            remain_segement_context = segment_context[the_second_to_last_punctuation + 1: ]
        else:
            remain_segement_context = context[segment_context_idx: ]
            
            
            
        if len(remain_segement_context) <= max_length:

            res.append(remain_segement_context)
            break
            
        else:
            partial_res = truncate(remain_segement_context, max_length)
            res += partial_res
        
        break
            
            
        
    return res

def generate_data(context_list: List[str], max_length) -> List[str]:
    # minus 4 because of reserving the length for [SEP]、[CLS]、[BOS]、[EOS]
    res_list = []
    max_length = max_length - 4
    for i in tqdm(context_list, total=len(context_list)):
        if len(i) > max_length:
            res_list += truncate(i, max_length)
        else:
            res_list.append(i)
    return res_list
            
    
    