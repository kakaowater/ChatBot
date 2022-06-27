# by_dajeong

# -*- coding: utf-8 -*-
from flask import Flask, render_template, request
import os, pickle, re, sys
import tensorflow as tf
import datetime

sys.path.append( '/content/drive/MyDrive/bert/codes' )
from to_array.tokenizationK import FullTokenizer
from models.bert_slot_model import BertSlotModel
from to_array.bert_to_array import BERTToArray

# -----------------------------------------------------------------


# 슬롯태깅 모델과 벡터라이저 불러오기

bert_model_hub_path = '/content/drive/MyDrive/bert-module'
# pretrained BERT 모델을 모듈로 export - ETRI에서 사전훈련한 BERT의 체크포인트를 가지고 만든 BERT 모듈
is_bert = True

vocab_file = os.path.join(bert_model_hub_path, "assets/vocab.korean.rawtext.list")
# 토큰화된 단어들에 숫자 매겨놓은 것
bert_to_array = BERTToArray(is_bert, vocab_file)
# 벡터라이저
tokenizer = FullTokenizer(vocab_file=vocab_file)
# 보캡 파일로 토크나이징 하는 도구?

load_folder_path = '/content/drive/MyDrive/finetuned_epoch128_sample2'
# loading models
print('Loading models ...')
if not os.path.exists(load_folder_path):
    print('Folder `%s` not exist' % load_folder_path)

with open(os.path.join(load_folder_path, 'tags_to_array.pkl'), 'rb') as handle:
    tags_to_array = pickle.load(handle)
    slots_num = len(tags_to_array.label_encoder.classes_)
 
# this line is to disable gpu
os.environ["CUDA_VISIBLE_DEVICES"]="-1"
config = tf.ConfigProto(intra_op_parallelism_threads=1,
                        inter_op_parallelism_threads=1,
                        allow_soft_placement=True,
                        device_count = {'CPU': 1})
sess = tf.compat.v1.Session(config=config)
graph = tf.compat.v1.get_default_graph()

model = BertSlotModel.load(load_folder_path, sess)
# finetuned_epoch128로 훈련시킨 슬롯태깅모델 (?)


types = ['에일', 'IPA', '라거', '바이젠', '흑맥주']
abv = ['3도', '4도', '5도', '6도', '7도', '8도',
            '3도이상', '4도이상', '5도이상', '6도이상', '7도이상',
            '3도 이상', '4도 이상', '5도 이상', '6도 이상', '7도 이상', 
            '4도이하', '5도이하', '6도이하', '7도이하', '8도이하',
            '4도 이하', '5도 이하', '6도 이하', '7도 이하', '8도 이하']
flavor = ['과일', '홉', '꽃', '상큼한', '커피', '스모키한']
taste = ['단', '달달한', '달콤한', '안단', '안 단', 
              '달지 않은', '달지않은', '쓴', '씁쓸한',
              '쌉쌀한', '달콤씁쓸한', '안쓴', '안 쓴', '쓰지 않은',         
              '신', '상큼한', '새콤달콤한', '시지 않은', '시지않은',
              '쓰지않은/','안신', '안 신', '과일', '고소한', '구수한']

options = {'types':'종류', 'abv':'도수', 'flavor':'향', 'taste':'맛'}
dic = {i:globals()[i] for i in options}
# globals()[원하는 변수 이름] = 변수에 할당할 값 : 변수 여러개 동시 생성
# dic = {'types': types, 'abv': abv, 'flavor': flavor, 'taste': taste}

cmds = {'명령어':[],
        '종류':types,
        '도수':abv,
        '향':flavor,
        '맛':taste}
cmds["명령어"] = [k for k in cmds]
# cmds["명령어"] = ['명령어', '종류', '도수', '향', '맛']


app = Flask(__name__)
app.static_folder = 'static'

@app.route("/")
def home():

############################### TODO ##########################################
# 슬롯 사전 만들기
  app.slot_dict = {'types':[], 'abv':[], 'flavor':[], 'taste':[]}
  app.score_limit = 0.8
###############################################################################

  return render_template("index.html")


@app.route("/get")
def get_bot_response():
    userText = request.args.get('msg').strip() # 사용자가 입력한 문장
    
    if userText[0] == "!":
      try:
        li = cmds[userText[1:]]
        message = "<br />\n".join(li)
      except:
        message = "입력한 명령어가 존재하지 않습니다."

      return message

    text_arr = tokenizer.tokenize(userText)
    input_ids, input_mask, segment_ids = bert_to_array.transform([" ".join(text_arr)])

    # 예측
    with graph.as_default():
      with sess.as_default():
        inferred_tags, slots_score = model.predict_slots(        
          [input_ids, input_mask, segment_ids], tags_to_array
        )

    # 결과 체크
    print("text_arr:", text_arr) 
    print("inferred_tags:", inferred_tags[0])
    print("slots_score:", slots_score[0])   

    # 슬롯에 해당하는 텍스트를 담을 변수 설정
    #slot_text = {k: "" for k in app.slot_dict}
    #app.slot_dict = {'types':[], 'abv':[], 'flavor':[], 'taste':[]}
    slot_text = {'abv': '', 'flavor': '', 'taste': '', 'types': ''}

    # 슬롯태깅 실시
    for i in range(0, len(inferred_tags[0])):           
      #if slots_score[i] >= app.score_limit:      
      if slots_score[0][i] >= app.score_limit:
        if not inferred_tags[0][i] == "O":
          word_piece = re.sub("_", "", text_arr[i])       
          slot_text[inferred_tags[0][i]] += word_piece
        #catch_slot(i, inferred_tags, text_arr, slot_text)
        #태그가 '0'가 아니면 text_arr에서 _를 지우고  slot_text에서 해당하는 태그에 단어를 담는다. 
        ##slot_text = {'abv': '', 'flavor': '', 'taste': '', 'types': ''}
      else:
        print("something went wrong!")
    print("slot_text:", slot_text)

    # 메뉴판의 이름과 일치하는지 검증
    for k in app.slot_dict:
    # k : 'types','abv','flavor','taste' 
        for x in dic[k]:
        # {'types': [types], 'abv': [abv], 'flavor': [flavor], 'taste': [taste]}  
          x = x.lower().replace(" ", "\s*")
          m = re.search(x, slot_text[k])
          if m:
            app.slot_dict[k].append(m.group())
    print(app.slot_dict)           

    #options = {'beer_types':'종류', 'beer_abv':'도수', 'beer_flavor':'향', 'beer_taste':'맛'}
    empty_slot = [options[k] for k in app.slot_dict if not app.slot_dict[k]]
    filled_slot = [options[k] for k in app.slot_dict if app.slot_dict[k]]    
    print(empty_slot)
    print(filled_slot)

    if ('종류' in empty_slot and '도수' in empty_slot and '향' in empty_slot and '맛' in empty_slot):
      message = '맥주의 종류, 도수, 향, 맛을 넣어서 다시 입력해주세요'
    elif ('종류' in empty_slot and '도수' in empty_slot and '향' in empty_slot) :
      message = '종류와 도수, 향은요?'
    elif ('종류' in empty_slot and '도수' in empty_slot and '맛' in empty_slot) :
      message = '종류와 도수, 맛은요?'  
    elif ('도수' in empty_slot and '향' in empty_slot and '맛' in empty_slot) :
      message = '도수와 향, 맛은요?'
    elif ('종류' in empty_slot and '도수' in empty_slot) :
      message = '종류와 도수는요?'  
    elif ('종류' in empty_slot and '향' in empty_slot) :
      message = '종류와 향은요?'       
    elif ('종류' in empty_slot and '맛' in empty_slot) :
      message = '종류와 맛은요?'                
    elif ('도수' in empty_slot and '향' in empty_slot) :
      message = '도수와 향은요?' 
    elif ('도수' in empty_slot and '맛' in empty_slot) :
      message = '도수와 맛은요?' 
    elif ('향' in empty_slot and '맛' in empty_slot) :
      message = '향과 맛은요?' 
    elif ('종류' in empty_slot) :
      message = '종류는요?' 
    elif ('도수' in empty_slot) :
      message = '도수는요?' 
    elif ('향' in empty_slot) :
      message = '향은요?' 
    elif ('맛' in empty_slot) :
      message = '맛은요?'


    return message  



    #app.slot_dict['a_slot'] = ''
    #print(app.slot_dict)

    #return 'hi'
    
def catch_slot(i, inferred_tags, text_arr, slot_text):
  if not inferred_tags[0][i] == "O":
    word_piece = re.sub("_", " ", text_arr[i])
    slot_text[inferred_tags[0][i]] += word_piece
# inffered_tags = ['O', 'abv', 'abv', 'O', 'type', 'type', 'type', 'O', 'O', 'O', 'O', 'O', 'flavor', 'flavor', 'flavor', 'flavor', 'O', 'O']
#text_arr = ['나는_', '7', '도_', '넘는_', '흑', '맥', '주로_', '주', '문', '하고_', '싶', '어_', '스', '모', '키', '한_', '걸', '로_']
#slot_text = {'beer_abv': '7도', 'beer_flavor': '', 'beer_taste': '', 'beer_types': '흑맥주'}

def init_app(app):
    app.slot_dict = {
        'types': [],
        'abv':[],
        'flavor':[],
        'taste':[],
        }

############################### TODO ##########################################
# 1. 사용자가 입력한 한 문장을 슬롯태깅 모델에 넣어서 결과 뽑아내기


# 2. 추출된 슬롯 정보를 가지고 더 필요한 정보 물어보는 규칙 만들기 (if문)
    #app.slot_dict['a_slot'] = ''
    #print(app.slot_dict)

    #return 'hi' # 챗봇이 이용자에게 하는 말을 return
###############################################################################








