# KPF-BERT-NER

## 모델 소개

### KPF-BERT-NER


본 과제에서는 한국언론진흥재단이 개발한 kpf-BERT 모델을 기반으로 NER(Named Entity Recognition) task를 수행할 수 있는 kpf-BERT-ner 모델을 설계 및 개발한다. NER은 이름을 가진 객체를 인식하는 것을 의미한다. 한국정보통신기술협회가 제공하는 정보통신용어사전에 따르면 NER은 다음과 같다. 
“NER은 미리 정의해둔 사람, 회사, 장소, 시간, 단위 등에 해당하는 단어(개체명)를 문서에서 인식하여 추출 분류하는 기법. 추출된 개체명은 인명(person), 지명(location), 기관명(organization), 시간(time) 등으로 분류된다. 개체명 인식은 정보 추출을 목적으로 시작되어 자연어 처리, 정보 검색 등에 사용된다.” 
실무적으로 표현하면 ‘문자열을 입력으로 받아 단어별로 해당하는 태그를 출력하게 하는 multi-class 분류 작업’이다. 본 과제에서는 kpf-BERT-ner 모델을 설계 및 개발하고 언론 기사를 학습하여 150개 클래스를 분류한다.

한국언론진흥재단이 개발한 kpf-BERT를 기반으로 classification layer를 추가하여 kpf-BERT-ner 모델을 개발한다. 
BERT는 대량의 데이터를 사전학습에 사용한다. 
kpf-BERT는 신문기사에 특화된 BERT 모델로 언론, 방송 매체에 강인한 모델이다.
![img](https://user-images.githubusercontent.com/87846939/221454339-b893a2d1-5c04-4307-bcf8-90a966604050.png)

BERT 모델의 학습을 위해서는 문장에서 토큰을 추출하는 과정이 필요하다. 
이는 kpf-BERT에서 제공하는 토크나이저를 사용한다. 
kpf-BERT 토크나이저는 문장을 토큰화해서 전체 문장벡터를 만든다. 
이후 문장의 시작과 끝 그 외 몇가지 특수 토큰을 추가한다. 
이 과정에서 문장별로 구별하는 세그먼트 토큰, 각 토큰의 위치를 표시하는 포지션 토큰 등을 생성한다. 
![img_1](https://user-images.githubusercontent.com/87846939/221454345-fa5a9fdf-45a9-4ecf-8900-64f5ae56a74f.png)

NER 모델 개발을 위해서는 추가로 토큰이 어떤 클래스를 가졌는지에 대한 정보가 필요하다. 
본 과제에서는 토크나이저를 사용하여 문장을 토큰으로 분류한 이후에 해당 토큰별로 NER 태깅을 진행한다. 
추가로 BIO(Begin-Inside-Outside) 표기법을 사용하여 정확도를 높인다. 
B는 개체명이 시작되는 부분, I는 개체명의 내부 부분, O는 개체명이 아닌 부분으로 구분한다
![img_2](https://user-images.githubusercontent.com/87846939/221454349-53690fe7-96b8-4dcb-aad9-5db1c0bb8994.png)

---
## 실행하기 위한 환경

1. 라이브러리

    ```
    python==3.7
    PyTorch==1.7.1
    tqdm==4.64.0
    transformers==4.24.0
    seqeval==0.0.12
    
    그 외 pytorch 등 머신러닝을 위한 기본적인 패키지(소스 내 import 참조)
    ```
    
---
## Usage

1. BERT

    kpfBERT 모델을 다운받아 아래의 디렉토리에 넣어주고 fine-tuning을 진행한다.

    - `kpfbert` 다운받은 kpfBERT 화일들이 위치할 곳
 

2. Data Processing

    모델 전처리 과정은 학습 데이터(json 파일)를 불러오고 해당 파일 중에 필요한 정보로 파싱한다. 
    해당 정보들은 토크나이저를 통해 문장 내에서의 토큰의 위치, 세그먼트, ID 등으로 다시 구분하며 토큰별로 BIO 표기법 형태에 맞춰 표기한다. 그 후 학습을 위해 tensor 형태로 변환하여 저장한다. 
    데이터셋 관리를 위해 torch의 Dataset 라이브러리를 사용해서 데이터를 불러올 때마다 위의 내용을 반복한다.    

    `Dataset.py`에서 데이터 전처리 과정을 진행한다.
    - NerDataset : ner dataset 클래스. (torch의 dataset 라이브러리 사용, 해당 문서 참고)
    - load_data  : 말뭉치 데이터(dict)에 맞게 필요한 정보를 추출하는 함수. 정보를 추출하고 BIO 표기법으로 분류 후 모델의 input 형태로 변형함.
    - collate_fn : 학습에 사용할 수 있도록 torch 라이브러리를 사용하여 타입을 변형시키고 매칭시켜줌.
    
    말뭉치 데이터를 받아 학습에 필요한 input 형태로 변환.
    ```
    sentence : 문장 (ex. "아디다스의 대표 운동화 '스탠스미스'가 연간 800만 켤레 팔리는 것과 비교하면 놀랄 만한 실적이다")
    token_label : 토큰의 클래스 (ex. ['B-OGG_ECONOMY', 'I-OGG_ECONOMY', 'I-OGG_ECONOMY', 'O', 'O', 'O', 'O', 'B-AFW_OTHER_PRODUCTS', 'I-AFW_OTHER_PRODUCTS',
                                    'I-AFW_OTHER_PRODUCTS', 'O', 'O', 'O', 'B-QT_COUNT', 'I-QT_COUNT', 'I-QT_COUNT', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O'])
    char_label : 단어의 클래스 <BIO표기법> (ex. ['B-OGG_ECONOMY', 'I-OGG_ECONOMY', 'I-OGG_ECONOMY', 'I-OGG_ECONOMY', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 
                                                'B-AFW_OTHER_PRODUCTS', 'I-AFW_OTHER_PRODUCTS', 'I-AFW_OTHER_PRODUCTS', 'I-AFW_OTHER_PRODUCTS', 'I-AFW_OTHER_PRODUCTS',
                                                'O', 'O', 'O', 'O', 'O', 'O', 'B-QT_COUNT', 'I-QT_COUNT', 'I-QT_COUNT', 'I-QT_COUNT', 'I-QT_COUNT', 'I-QT_COUNT', 'I-QT_COUNT',
                                                'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O'])
    offset_mapping : 토큰의 위치정보 (ex. [(0, 0), (0, 2), (2, 3), (3, 4), (4, 5), (6, 8), (9, 12), (13, 14), (14, 16), (16, 17), (17, 19), (19, 20), (20, 21), (22, 24), (25, 28),
                                    (28, 29), (30, 32), (33, 35), (35, 36), (37, 38), (38, 39), (40, 42), (42, 44), (45, 47), (48, 50), (51, 53), (53, 54), (54, 55), (55, 56), (0, 0)])
    ```
   

3. Train

    ```
    python train.py -s TRAIN_FILE -o MODEL_NAME
    (ex. python train.py -s dataset/NXEL2102203310.json -o kpfbert-ner)
    ```
    - `kpfbert-ner` : KPF-BERT-NER 모델의 저장 위치
    - `train.py` : 학습 관련 코드. 
    - 실행에 필요한 파일 : label.py, config.py, Dataset.py, kpfbert 폴더가 있어야함.
    - input : dataset, model name
    - output : trained model
   
   ![img_3](https://user-images.githubusercontent.com/87846939/221454354-0f49973e-cf42-4a5d-ac79-173a0e4ff165.png)
   
   학습데이터의 입력값과 출력값
      

4. Predictions

   `ner_module.py`의 `ner_predict(text)`를 사용하여 `text`에 기사 전문을 넣어.
