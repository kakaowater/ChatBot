# 버트 미세조정 - 슬롯 태깅

# 복잡한 코드를 마주치면?:
 1. readme를 찾아라
 2. 인풋, 아웃풋을 파악하는 것이 우선. 
 --> 이번 과제는 인풋, 아웃풋만 잘 파악해서 어떻게든 훈련을 끝내보자는 과제다.
 3. 파이썬 파일명을 잘 읽자. 가장 중요한 함수를 찾아보자.
 (보통 core, main, train 들어가는 것이 가장 중요한 함수)
 (++if __name__ == "__main__": 이것도 찾아보자)


''' codes 하위 폴더
1. 004_bert_eojeol_tensorflow : etri BERT 사전훈련 체크포인트 폴더
체크포인트란?: 훈련 결과를 컴퓨터가 이해할 수 있는 형태로 만든 파일 (워드 임베딩 등의 정보)
--> "xxx.pt" 또는 "model.ckpt-56000" 이러한 파일명
--> 보통 딥러닝에서 많이 씀
2. export_kobert : bert_to_module.py: 사전훈련 체크포인트를 모듈로 exprot하는 과정
3. prepare_data.py: 우리가 했던 거

# 주요 훈련 파일: 4, 5
4. train_bert_finetuning.py: train 파일
여기서 train이라함은, 우리가 받은 pre-train BERT 체크포인트 ++ fine-tuning data(seq.in / seq.out)
5. eval_bert_finetuned.py : evaluation 파일 (f1 score 등)    ---> 훈련 끝
6. inference.py : 훈련 결과 확인 위해, 아무 문장이나 인풋으로 넣어서 확인해볼 수 있다.
'''
  
1. pretrained BERT 모델을 모듈로 export  
    - ETRI에서 사전훈련한 BERT의 체크포인트를 가지고 BERT 모듈을 만드는 과정.
    - `python export_korbert/bert_to_module.py -i {체크포인트 디렉토리} -o {output 디렉토리}`
    - 예시: `python export_korbert/bert_to_module.py -i /content/drive/MyDrive/004_bert_eojeol_tensorflow -o /content/drive/MyDrive/bert-module`  

2. 데이터 준비
    - 모델을 훈련하기 위해 필요한 seq.in, seq.out이라는 2가지 파일을 만드는 과정.  
    - `python prepare_data.py -i {input파일} -o {output 디렉토리}`   
    - 예시: `python prepare_data.py -i /content/drive/MyDrive/data/sample_data.txt -o /content/drive/MyDrive/data/sample/`  
  
3. Fine-tuing 훈련  
    - TODO - 위의 내용처럼 어떻게 하면 `train_bert_finetuning.py` 코드를 실행할 수 있는지 코드 내부의 parser을 참조하여 작성하세요.  
  
4. 모델 평가  
    - TODO - 위의 내용처럼 어떻게 하면 `eval_bert_finetuned.py` 코드를 실행할 수 있는지 코드 내부의 parser을 참조하여 작성하세요.  
    - 테스트의 결과는 --model에 넣어준 모델 경로 아래의 `test_results`에 저장된다.  
  
5. Inference (임의의 문장을 모델에 넣어보기)  
    - TODO - `eval_bert_finetuned.py`를 참고하여 한 문장씩 넣어서 모델이 내뱉는 결과물을 볼 수 있도록 inference.py 코드를 완성하세요.  
    - `python inference.py --model {훈련된 모델이 저장된 경로}`   
    - 예시: `python inference.py --model saved_model/`   
    - 모델 자체가 용량이 커서 불러오는 데까지 시간이 걸림  
    - "Enter your sentence:"라는 문구가 나오면 모델에 넣어보고 싶은 문장을 넣어 주면 됨  
    - quit라는 입력을 넣어 주면 종료  
