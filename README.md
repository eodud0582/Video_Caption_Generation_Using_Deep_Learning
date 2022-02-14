# Deep Learning Based Video Caption Generator (배너 삽입)

# 딥러닝 기반 동영상 캡션 자동 생성
### 한국어 데이터 및 딥러닝을 활용한 동영상 캡션 자동 생성 모델 개발

---
## 프로젝트 목적

- 캡셔닝(captioning)이란 이미지나 영상이 주어졌을 때, 해당 이미지나 영상에 대한 설명을 문장 형식으로 생성하는 기술입니다. 
- 한국어 멀티모달 데이터를 활용하여, 이용자가 동영상을 입력하면 동영상의 상황/내용을 묘사하는 캡션을 생성해 주는 **CNN+LSTM 기반의 동영상 캡셔닝 모델을 개발**하였습니다.

---
## 프로젝트 결과

<div align=center> 

![captioned_office](https://user-images.githubusercontent.com/38115693/153170767-f47adbe6-db3f-4bae-abc0-58c47bdac226.gif)![captioned_people_walking](https://user-images.githubusercontent.com/38115693/153171395-11b209c1-f0b5-4075-8c43-759b69a278a5.gif)
	
![captioned_cats](https://user-images.githubusercontent.com/38115693/153171356-24403b58-fa3c-482b-b9c7-b558b45ca465.gif)![captioned_black_cat](https://user-images.githubusercontent.com/38115693/153180977-65572efb-2083-4981-a3ed-12c9ebbbbb00.gif)

![captioned_seoul_night_city](https://user-images.githubusercontent.com/38115693/153171118-a93533c4-4e47-408a-ba8d-50f0597c0adb.gif)![captioned_seoul_road_street](https://user-images.githubusercontent.com/38115693/153171144-62f41be7-4bad-45bd-a2c7-0b786e4661a1.gif)

![captioned_KETI_MULTIMODAL_0000000695](https://user-images.githubusercontent.com/38115693/153171433-5aca8f3d-5832-4004-a7db-b63d7bc3a371.gif)![captioned_KETI_MULTIMODAL_0000000215](https://user-images.githubusercontent.com/38115693/153171459-394a5fcc-deba-45e5-845a-aa05a327a0e9.gif)

<img src="https://user-images.githubusercontent.com/38115693/153171626-08746848-62f4-479c-960c-18478380bf33.gif" width="256"><img src="https://user-images.githubusercontent.com/38115693/153171646-18e6adac-ed1f-4f6f-9bf2-1a8a1d045300.gif" width="256"><img src="https://user-images.githubusercontent.com/38115693/153171658-c2d88d7b-4d85-4cde-a52d-425a7b948c36.gif" width="256">

</div>

---
## 프로젝트 기간

![image](https://user-images.githubusercontent.com/38115693/153365286-ecd7ed33-79d3-4bdf-90e8-9f39f92172b6.png)

- 프로젝트 기간: 2022.01.11~2022.02.02 (3.5주)

---
## 프로젝트 배경

**멀티모달(Multimodality)**
- '멀티모달'이란 언어뿐만 아니라 이미지, 영상에 이르기까지 사람의 의사소통과 관련된 다양한 정보를 습득하고 다룰 수 있는 기술을 의미합니다. 딥러닝에 대한 관심이 증가하면서 컴퓨터가 시각적 이미지를 인식하여 사람처럼 문장으로 표현하는 것에 대한 연구 또한 활발히 진행되어 왔습니다. 이런 멀티모달 기술을 사용하여, **이미지나 영상에 대한 설명글(캡션)이 자동으로 생성된다면 일상생활 속에서 다양한 서비스를 제공** 할 수 있을 것이라 생각하여 프로젝트를 진행했습니다. 

**한국어 데이터 기반**
- 한국어 데이터의 부족으로 한국어를 대상으로 한 동영상/이미지 캡션 연구는 활발히 이루어지지 못했습니다. **기존 동영상/이미지 캡션 연구와 개발도 대다수 영어로 공개된 데이터셋이 이용**되었으며, 한글 캡션을 생성하기 위해서는 **영어 데이터를 번역하여 사용하거나 캡션 결과를 번역**해야 했습니다. 하지만 국내에서도 한국어 캡션 데이터셋를 최근에서야 제공하고 있습니다. 이를 활용하여, **국내 환경과 한국어에 맞는 동영상/이미지 캡션 생성 모델 개발이 가능**해 졌습니다. 

---
## 데이터

![image](https://user-images.githubusercontent.com/38115693/153358951-a01619cf-5801-42df-aab8-880ac0e4f9ca.png)

- **AI Hub '멀티모달' 데이터셋**
	- 여러 TV 드라마에 대한 감정, 사용자 의도 등 다양한 관점의 멀티모달 데이터와 영상/음성/텍스트 정보가 있는 멀티모달 원시 데이터로 구성되어 있습니다.
	- 총 11개의 zip파일(약 300GB) 중 4개 zip파일을 사용했습니다.
	- 동영상, 샷 구간 이미지, 각 이미지별 5개 상황 묘사 텍스트 데이터 사용했습니다.
- **AI Hub '한국어 이미지 설명' 데이터셋**
	- MS COCO 캡셔닝 데이터를 한국어로 번역한 데이터입니다.
	- 총 123,287개 이미지와 각 이미지에 대한 묘사 5개 텍스트 데이터(영어/한국어)로 구성되어 있습니다.
	- 전체 이미지와 한국어 묘사 텍스트 데이터를 사용했습니다.
---
## 모델 구조

### 동영상 캡셔닝 모델

<div align=center><img src="https://user-images.githubusercontent.com/38115693/153815473-2fff29db-1349-4cae-8f2e-092341d32f2e.png" width="600"></div>
<div align=center> Video Captioning Model </div>

**동영상 캡셔닝 알고리즘**
- 입력받은 동영상을 여러 **프레임**으로 나누고, **이미지 캡션 생성 모델**을 통해 각 프레임 이미지에 대한 캡션을 생성합니다.
- 각 프레임 이미지에 대해 생성된 캡션을 해당 프레임 이미지에 출력하고, 프레임 이미지들을 **다시 동영상으로 변환**합니다.

**이미지 유사도 분석**
- 동영상의 캡션이 시시각각 변하는 구간들이 있어 캡션을 보는 것에 불편함이 있었습니다. 이에 대해, **이미지 유사도 분석을 통해 비슷한 프레임/장면에서는 동일한 캡션을 출력**하게 만들어 동영상 속 출력된 캡션이 부드럽게 전환되고, 보기 편하게 만들었으며, 캡셔닝 처리에 걸리는 시간도 단축시켰습니다.
- 이미지 유사도 측정은 **MSE**(Mean Squared Error)와 **SSIM**(Structured Similarity Image Matching)을 사용하였으며, 최종적으로 모델에는 SSIM을 적용하였습니다.

### 이미지 캡셔닝 모델

**Merge Encoder-Decoder Model**

<br>
<div align=center><img src="https://user-images.githubusercontent.com/38115693/153910950-13bdabce-df27-4e8a-a64c-827fcf8d42a6.png" width="300"></div>
<div align=center> Merge Architecture for Encoder-Decoder Model in Caption Generation </div>
<br>

- **Encoder-Decoder** architecture 기반
	- Encoder: 이미지와 텍스트를 읽어 고정된 길이의 벡터로 인코딩하는 network model
	- Decoder: 인코딩된 이미지와 텍스트를 이용해 텍스트 설명을 생성하는 network model
- Encoder-Decoder architecture 구현을 위해 *Marc Tanti, et al.* (*2017*)가 제시한 **Merge** 모델을 사용
	- Merge architecture에서는 **이미지와 언어/텍스트 정보가 별도로 인코딩** 되며, 이후 **multimodal layer** architecture의 Feedforward Network(FF)에서 병합(merge)되어 함께 처리됩니다.
	- CNN을 encoder로, RNN을 decoder로 사용한 기존의 Inject architecture와 비교하여, Merge 모델은 RNN을 텍스트 데이터에 대해서만 인코딩하고 해석하는 데 온전히 사용 할 수 있으며, 인코딩에 GloVe, FastText와 같은 pre-trained language model을 사용 할 수 있다는 장점이 있습니다. 또한, Merge 모델이 더 작은 layers로 더 나은 캡셔닝 성능을 보인다 알려져 있습니다.

**CNN(Convolutional Neural Networks) + LSTM(Long Short-Term Memory)**

<br>
<div align=center><img src="https://user-images.githubusercontent.com/38115693/153630047-befa082e-c486-45ea-ab70-2aabad793d2a.png" width="500"></div>
<div align=center> RNN as Language Model </div>
<br>

- **CNN**을 이미지 데이터 인코딩을 위한 '**이미지 모델**'로, **RNN/LSTM**을 텍스트 시퀀스 데이터를 인코딩하는 '**언어 모델**'로 사용
	- 이미지 인코딩: ImageNet 데이터셋으로 pre-trained 된 CNN 모델을 사용하는데, 다른 pre-trained 모델에 비해 상대적으로 training parameters가 더 적으면서도 더 우수한 성능을 가진 **InceptionV3**를 사용하여 전이학습(transfer learning) 합니다. 여기서 추출된 이미지 특성들은 캡셔닝 모델의 input으로 사용됩니다.
	- 텍스트 인코딩: 토큰화 된 정수 형태의 텍스트 시퀀스 데이터를 input으로 받고, pre-trained language model인 **FastText**를 사용하여 embedding layer에서 모든 단어를 200차원 벡터로 매핑합니다. 뒤이어 LSTM layer에서 벡터 시퀀스를 처리합니다.
- **Decoder 모델**
	- Decoder 모델은 각각 따로 처리된 이미지와 텍스트 **두 입력 모델의 인코딩 결과/벡터를 병합**하고 Dense layer을 통해 **시퀀스의 '다음 단어'를 생성**합니다.
	- Dense layer는 softmax에 의해 **모든 단어에 대한 확률분포**를 구하여 시퀀스의 다음 단어를 생성하게 됩니다.
<br>
<div align=center><img src="https://user-images.githubusercontent.com/38115693/153908121-a0cb87fc-0517-4551-8721-cc7c0bbe72dd.png" width="1000"></div>
<div align=center> Defined Model Structure </div>

---
## 캡션 생성

**학습한 모델을 사용한 캡션 생성 과정**

<div align=center><img src="https://user-images.githubusercontent.com/38115693/153921982-d1373341-5fba-444a-87a2-d801ee68e28a.png" width="600"></div>
<div align=center> Caption Generation Process </div>
<br>

- 이미지에 대한 캡션을 생성할 때, 학습한 모델은 시퀀스의 다음 단어를 예측(단어들의 연속을 예측) 하는데, **모든 단어에 대한 확률분포**를 구하여 예측/선택 힙니다. 다시 말해, 단어들의 후보 시퀀스들의 **우도(likelihood)**에 따라 점수화 되어 선택 됩니다.
- 이미지를 입력으로 받으면, **시퀀스의 시작을 의미하는 토큰인 'startseq'를 전달**하여 단어 하나를 생성한 다음, 다시 모델을 호출하고 생성된 단어까지를 연결하여/합쳐서 input으로 넘겨 그 다음 단어를 생성하게 됩니다.
- 이렇게 **다음 단어를 생성하고, 지금까지 생덩된 단어들을 다시 모델에 input으로 넘기고, 또 다음 단어를 생성하는 과정을 재귀적으로 반복**합니다. 그러다 아래 조건에 도달하게 되면 반복을 종료하고, 최종적으로 이미지에 대한 캡션이 만들어지게 됩니다.
	- (1) **시퀀스의 끝을 의미하는 토큰인 'endseq'가 생성**되거나,
	- (2) **최대 캡션 길이에 도달**할 때까지 반복합니다.
- 다음 단어를 예측하는 건 일반적으로 사용되는 Greedy Search와 Beam Search 두 기법을 사용했습니다.

**Greedy Search**
- Greedy search는 전체 단어에서 각 단어에 대한 확률 분포를 예측하여 다음 단어를 선택하는데 **각 스텝에서 가장 가능성이/확률이 높은 단어를 선택**합니다.
- 빠른 속도로 탐색 및 예측 과정이 완료되나, 하나의 예측만을 고려하기 때문에 minor한 변화에 영향을 받을 수 있어 최적의 예측을 하지 못활 위험이 있습니다. 쉽게 말해, 한 번이라도 잘못된 단어를 예측하게 되면 뒤이어 다 잘못된 예측이 될 수도 있다는 뜻입니다.

**Beam Search**
- Greedy Search의 단점을 보완하여 확장한 기법이 Beam Search 입니다.
- Beam Search에선 각 후보 시퀀스가 모든 가능한 다음 스텝들로 확장됩니다. 쉽게 말해, **가능한 모든 다음 단어/시퀀스를 예측** 합니다. 그렇게 예측된 각 후보는 확률을 곱하여 점수가 매겨지고, **가장 확률이 높은 k개(beam size) 시퀀스가 선택**되며, 다른 모든 후보들은 제거됩니다. 이 과정을 시퀀스가 끝날때까지 반복합니다.
	- 예를 들어, 만약 지정한 beam size가 2라면, 각 step에서 가장 확률 높은 2개를 선택합니다.
- Greedy Search의 경우는 beam size가 1인 것과 같다고 보면 됩니다. 
- Beam의 수는 일반적으로 5 또는 10을 사용하고, beam size가 클수록 타겟 시퀀스가 맞을 확률이 높지만 디코딩 속도가 떨어지게 됩니다.

---
## 모델 평가

**BLEU(Bilingual Evaluation Understudy) Score**

- 캡션 생성 평가 지표로는 기존의 이미지 캡션 논문들에서 사용되는 **BLEU 스코어(BLEU-1, BLEU-2, BLEU-3, BLEU-4)**를 사용했습니다.
- BLEU 스코어는 기계번역의 결과와 사람이 직접 번역한 결과가 얼마나 유사한지 비교하여 번역에 대한 성능을 측정하는 평가 지표로, 데이터의 X가 순서정보를 가진 단어들(문장)로 이루어져 있고, y 또한 단어들의 시리즈(문장)로 이루어진 경우에 사용됩니다.
- BLEU의 3가지 요소/기능입니다:
	- n-gram을 통한 순서쌍들이 얼마나 겹치는지 측정(precision)
	- 문장길이에 대한 과적합 보정(Brevity Penalty)
	- 같은 단어가 연속적으로 나올때 과적합 되는 것을 보정(Clipping)
- 측정 기준은 n-gram에 기반하는데, **예측 문장과 정답 문장의 n-gram들이 서로 얼마나 겹치는지 비교하여 정확도를 측정**하기 때문에 이미지 캡션 평가에서도 보편적으로 사용됩니다.
	- 점수를 매길 최대 길이를 n-gram 길이라고 할 때, n-gram이라고 하는 연속된 n개의 단어를 기준으로 실제 캡션(ground truth)과 얼마나 공통적인 단어가 나왔는지/겹치는지를 판단해서 BLEU 점수를 계산합니다.
	- k를 n-gram이라고 할 때, 만약 k=4라면, 길이가 4이하인 n-gram에 대해서만 고려하게 되며, 더 큰 길이의 n-gram은 무시합니다.
	- 보통 1~4의 크기의 n-gram을 측정 지표로 사용합니다.
- 점수는 0.0~1.0 사이에서 나타내는데, **1.0에 가까울 수록, 높을수록 좋은 점수를 의미**합니다.

---
## 모델링 과정

1. Using pretrained CNN to extract image features. A pretrained InceptionV3 CNN will be used to extract image features which will be merged with the RNN output
2. Prepare training data. The training captions will be tokenized and embedded using the GLOVE/FastText word embeddings. The embeddings will be fed into the RNN.
3. Model definition
4. Training the model
5. Generating novel image captions using the trained model. Test images and images from the internet will be used as input to the trained model to generate captions. The captions will be examined to determine the weaknesses of the model and suggest improvements.
6. Beam search. We will use beam search to generate better captions using the model.
7. Model evaluation. The model will be evaluated using the BLEU and ROUGE metric.

## 모델링 세부 과정

Trial 1
Trial 2
...
마지막 trial에 대해,
그리고 틀린, 동일한 캡션을 출력하는 구간이 있어.. (이런 식으로)
cross validation도 진행하여 overfitting을 확인했다. 그리고 validation loss가 더 이상 낮아지지 않는 지점의 모델을 최종 모델로 선택

---
## 활용방안

캡션 생성은 이미지나 영상을 설명하는 캡션을 자동으로 생성하는 기술로 이미지/시각 처리 기술과 자연어 처리 기술이 합쳐진 어려운 기술이지만, 이미지 검색, 유아 교육, 청각 장애인들을 위한 캡셔닝 서비스와 같은 응용에 사용될 수 있는 중요한 기술입니다.

**1. 사진이나 동영상에 대한 검색/SEO 개선과 컨텐츠 접근성 향상**
- 사진이나 동영상에 캡션을 입히는 것을 통해, 동영상 플랫폼이나 컨텐츠 제작자는 영상에 대한 검색엔진최적화, 즉 SEO를 개선할 수 있습니다. 구글 등 검색 엔진에선 이미지나 영상 검색에 대한 검색의 질도 향상 시킬 수 있습니다.
- 또한, 일상적으로 동영상을 시청할 때 소리 없이 습관적으로 시청하거나 해야하는 경우가 종종 있습니다. 이러한 경우, 시청자 입장에선 동영상 속 상황이나 내용을 더 쉽게 이해할 수 있을 것이며, 더 다양한 컨텐츠에도 접근 할 수 있게 될 것입니다.

**2. 시각 또는 청각이 불편한 사람들에게 사진/동영상을 설명**
- 캡셔닝 기술을 전자고글이나 카메라 등 디바이스에 접목시킨다면 시각 장애를 가진 사람들에게 영상에 대한 설명을 음성으로 제공하거나 길안내 등에 사용 될 수 있으며, 청각이 불편한 사람들에겐 영상에 캡션이 함께 제공되어 영상 속 상황이나 행동을 더 쉽게 이해할 수 있어 미디어/컨텐츠에 대한 접근성/접근 환경을 개선 할 수 있습니다.

**3. 자율 주행 차량에 활용**
- 차량 주변 장면/환경에 대해 캡션을 적절하게 생성 할 수 있게 되면 자율 주행 시스템을 더욱 고도화 할 수 있습니다.

**4. CCTV 감시 장비에 활용**
- CCTV에 캡셔닝 기술을 적용하여 위험한 무기나 상황을 탐지하여 알람을 보내는 등 위험 사전 방지와 치안유지에 도움이 될 것입니다.

**5. 미술 심리 치료**
- 입력된 그림을 기록하여 전문 상담사들이 그림에 대한 질문을 제시하여 내면의 심리를 상세하게 기록할 수 있도록 돕는 등 미술 심리치료에도 활용될 수 있습니다.

**6. 언어 교육**
- 이미지나 동영상에 대한 한국어 설명글을 통해 아동이나 외국인에게 언어 교육도 제공 할 수 있을 것입니다. 

---
## 마무리

(한계점, 보완점, 향후 과제 등)

**모델의 캡션 예측/생성 성능 향상을 위한 시도**
- 더 많은 데이터를 사용하여 학습한다면, 캡션 예측/생성 성능이 더 좋아질 것으로 생각되기 때문에, 데이터를 더 확보하여 시도
	- AI허브 MSCOCO와 멀티모달 두 데이터를 합쳐 모델 학습을 진행해 보는 것도 고려
- AI허브 멀티모달 영상 데이터에 대해 pre-trained CNN 모델로 transfer learning시 fine tuning을 더 시도하거나(trainable layer을 증가?), 자체 CNN 모델을 설계하여 처음부터 학습하여 특성 추출
- 모델 아키텍쳐를 변경 (e.g. Bidirectional RNNs/LSTMs 사용, Attention 메커니즘 기법 사용, Injecting methodology 사용)
- more 하이퍼파라미터 튜닝 (e.g. learning rate, batch size, embedding dimension 300, number of layers, number of units, dropout rate, batch normalization 등 조정)
- 영상을 표현하는 시각 특징 외에, 정적 그리고 동적 의미 특징들도 이용

**출력된 캡션에 대한 추가적인 처리**
- 문장이 완전하지 않은 형태로 출력 되는 경우가 있습니다. 예를 들어, "...에 서있다"가 맞는 형태이지만, "...에서 있다"로 출력이 되는 경우입니다. 더 고도화한 문장 생성 및 출력을 위해 형태소 분석이나 관련 기능을 조사하여 적용이 필요합니다.

---
## 참고 자료



