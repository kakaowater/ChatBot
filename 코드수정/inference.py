# -*- coding: utf-8 -*-

############################### TODO ##########################################
# 필요한 모듈 불러오기
###############################################################################

import argparse
import os
import pickle
import tensorflow as tf

from to_array.bert_to_array import BERTToArray #import tokenizationK
from models.bert_slot_model import BertSlotModel

# read command-line parameters
parser = argparse.ArgumentParser('Evaluating the BERT / ALBERT NLU model')
parser.add_argument('--model', '-m', help = 'Path to BERT / ALBERT NLU model', type = str, required = True)
parser.add_argument('--type', '-tp', help = 'bert or albert', type = str, default = 'bert', required = False)


VALID_TYPES = ['bert', 'albert']

args = parser.parse_args()
load_folder_path = args.model
type_ = args.type

# this line is to disable gpu
os.environ["CUDA_VISIBLE_DEVICES"]="-1"
                                      
config = tf.ConfigProto(intra_op_parallelism_threads=1,
                        inter_op_parallelism_threads=1,
                        allow_soft_placement=True,
                        device_count = {'CPU': 1})
sess = tf.compat.v1.Session(config=config)                        

if type_ == 'bert':
############################### TODO 경로 고치기 ##########################################
    bert_model_hub_path = '/content/drive/MyDrive/ChatBot/codes/bert-module'
###########################################################################################
    is_bert = True
elif type_ == 'albert':
    bert_model_hub_path = 'https://tfhub.dev/google/albert_base/1'
    is_bert = False
else:
    raise ValueError('type must be one of these values: %s' % str(VALID_TYPES))


############################### TODO ##########################################
# 모델과 벡터라이저 불러오기
###############################################################################
bert_vocab_path = os.path.join(bert_model_hub_path, '/content/drive/ChatBot\codes\bert-module\assets\vocab.korean.rawtext.list')
bert_to_array = BERTToArray(is_bert, bert_vocab_path)

'''model = BertSlotModel(load_folder_path, sess)
tags_to_array_path = os.path.join(load_folder_path, 'tags_to_array.pkl')
with open(tags_to_array_path, 'rb') as handle:
  tags_to_array = pickle.load(handle)
  slots_num = len(tags_to_array.label_encoder.classes_)
while True:
    print('\nEnter your sentence: ')
    try:
        input_text = input().strip()
        #input_text_arr = input_text.splitlines()
        input_text_arr = BERTToArray.__to_array(input_text)
        input_ids, input_mask, segment_ids = BERTToArray.transform([' '.join(input_text_arr)])
        inferred_tags, slot_score = model.predict_slots([input_ids, input_mask, segment_ids], tags_to_array)
        print(input_text_arr)
        print(inferred_tags)
        print(slot_score)
    except:
        continue
    if input_text == 'quit':
        break'''

while True:
    print('\nEnter your sentence: ')
    try:
        input_text = input().strip()
    except:
        continue
        
    if input_text == 'quit':
        break

############################### TODO ##########################################
# 사용자가 입력한 한 문장을 슬롯태깅 모델에 넣어서 결과 뽑아내기
###############################################################################

tf.compat.v1.reset_default_graph()

