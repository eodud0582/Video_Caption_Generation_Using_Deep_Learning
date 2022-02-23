<img width="100%" alt="banner_2" src="https://user-images.githubusercontent.com/38115693/154050968-88be5579-22ba-4437-9baa-72d59f4c171e.png">

# 딥러닝 기반 동영상 캡션 생성
#### 한국어 데이터 및 딥러닝을 활용한 동영상 캡션 생성 모델 개발

---
## :question: 프로젝트 목적

```
# 프로젝트에 대한 자세한 overview는 첨부된 Project_Overview.md 파일을 참고 바랍니다.
```

- 캡셔닝(captioning)이란 이미지나 영상이 주어졌을 때, 해당 이미지나 영상에 대한 설명을 문장 형식으로 생성하는 기술입니다. 
- 한국어 멀티모달 데이터를 활용하여, 이용자가 동영상을 입력하면 동영상의 상황/내용을 묘사하는 캡션을 생성해 주는 **CNN+LSTM 기반의 동영상 캡셔닝 모델을 개발**하였습니다.

---
## :boom: 프로젝트 결과

<div align=center> 

![captioned_office](https://user-images.githubusercontent.com/38115693/153170767-f47adbe6-db3f-4bae-abc0-58c47bdac226.gif)![captioned_people_walking](https://user-images.githubusercontent.com/38115693/153171395-11b209c1-f0b5-4075-8c43-759b69a278a5.gif)
	
![captioned_cats](https://user-images.githubusercontent.com/38115693/153171356-24403b58-fa3c-482b-b9c7-b558b45ca465.gif)![captioned_black_cat](https://user-images.githubusercontent.com/38115693/153180977-65572efb-2083-4981-a3ed-12c9ebbbbb00.gif)

![captioned_seoul_night_city](https://user-images.githubusercontent.com/38115693/153171118-a93533c4-4e47-408a-ba8d-50f0597c0adb.gif)![captioned_seoul_road_street](https://user-images.githubusercontent.com/38115693/153171144-62f41be7-4bad-45bd-a2c7-0b786e4661a1.gif)

![captioned_KETI_MULTIMODAL_0000000695](https://user-images.githubusercontent.com/38115693/153171433-5aca8f3d-5832-4004-a7db-b63d7bc3a371.gif)![captioned_KETI_MULTIMODAL_0000000215](https://user-images.githubusercontent.com/38115693/153171459-394a5fcc-deba-45e5-845a-aa05a327a0e9.gif)

<img src="https://user-images.githubusercontent.com/38115693/153171626-08746848-62f4-479c-960c-18478380bf33.gif" width="256"><img src="https://user-images.githubusercontent.com/38115693/153171646-18e6adac-ed1f-4f6f-9bf2-1a8a1d045300.gif" width="256"><img src="https://user-images.githubusercontent.com/38115693/153171658-c2d88d7b-4d85-4cde-a52d-425a7b948c36.gif" width="256">

</div>

---
## :calendar: 프로젝트 기간

![image](https://user-images.githubusercontent.com/38115693/153365286-ecd7ed33-79d3-4bdf-90e8-9f39f92172b6.png)

- 프로젝트 기간: 2022.01.11~2022.02.02 (3.5주)

---
## :bulb: 프로젝트 배경

**멀티모달(Multimodality)**
- 멀티모달 기술을 사용하여, **이미지나 영상에 대한 설명글(캡션)이 자동으로 생성된다면 일상생활 속에서 다양한 서비스를 제공** 할 수 있을 것이라 생각하여 프로젝트를 진행했습니다. 

**한국어 데이터 기반**
- **기존 동영상/이미지 캡션 연구와 개발은 대다수 영어로 된 데이터셋이 이용**되었습니다.
- 최근 국내에서 공개된 한국어 캡션 데이터셋을 활용하여, **국내 환경과 한국어에 맞는 동영상/이미지 캡션 생성 모델 개발이 가능**해 졌습니다. 

---
## :floppy_disk: 데이터

- **AI Hub '멀티모달' 데이터셋**
	- 여러 TV 드라마에 대한 감정, 사용자 의도 등 다양한 관점의 멀티모달 데이터와 영상/음성/텍스트 정보가 있는 멀티모달 원시 데이터로 구성되어 있습니다.
	- 동영상, 샷 구간 이미지, 각 이미지별 5개 상황 묘사 텍스트 데이터를 사용했습니다.
- **AI Hub '한국어 이미지 설명' 데이터셋**
	- MS COCO 캡셔닝 데이터를 한국어로 번역한 데이터로 총 123,287개 이미지와 각 이미지에 대한 묘사 5개 텍스트 데이터(영어/한국어)로 구성되어 있습니다.
	- 전체 이미지와 한국어 묘사 텍스트 데이터를 사용했습니다.

<div align=center><img src="https://user-images.githubusercontent.com/38115693/154023368-08583ffd-a8f0-4f60-97fe-dab56d4f2c62.png" width="600"></div>
<div align=center> AI Hub MSCOCO Image Caption Dataset </div>

---
## :mag_right: 모델 구조

### :clapper: 동영상 캡셔닝 모델

<div align=center><img src="https://user-images.githubusercontent.com/38115693/153815473-2fff29db-1349-4cae-8f2e-092341d32f2e.png" width="600"></div>
<div align=center> Video Captioning Model </div>

**동영상 캡셔닝 알고리즘**
- 입력받은 동영상을 여러 **프레임**으로 나누고, **이미지 캡셔닝 모델**을 통해 각 프레임 이미지에 대한 캡션을 생성하고 **다시 동영상으로 변환**합니다.
- **이미지 유사도 분석을 통해 비슷한 프레임/장면에서는 동일한 캡션을 출력**합니다.
- 형태소 분석기를 활용하여 생성된 캡션이 어절 단위로 구분 된 문장 형태로 출력되도록 처리했습니다.

### :camera: 이미지 캡셔닝 모델

**Merge Encoder-Decoder Model**

<br>
<div align=center><img src="https://user-images.githubusercontent.com/38115693/153910950-13bdabce-df27-4e8a-a64c-827fcf8d42a6.png" width="300"></div>
<div align=center> Merge Architecture for Encoder-Decoder Model in Caption Generation </div>
<br>

- **Encoder-Decoder** architecture 기반
- *Marc Tanti, et al.* (*2017*)가 제시한 **Merge** 모델 사용
	- Merge architecture에서는 **이미지와 언어/텍스트 정보가 별도로 인코딩** 되며, 이후 **multimodal layer** architecture의 Feedforward Network(FF)에서 병합(merge)되어 함께 처리됩니다.

**CNN(Convolutional Neural Networks) + LSTM(Long Short-Term Memory)**

<br>
<div align=center><img src="https://user-images.githubusercontent.com/38115693/153630047-befa082e-c486-45ea-ab70-2aabad793d2a.png" width="500"></div>
<div align=center> RNN as Language Model </div>
<br>

- **Pre-trained CNN 모델 InceptionV3**을 이미지 데이터 인코딩을 위한 '**이미지 모델**'로, **RNN/LSTM**을 텍스트 시퀀스 데이터를 인코딩하는 '**언어 모델**'로 사용
- **Decoder 모델**은 각각 따로 처리된 이미지와 텍스트 **두 입력 모델의 인코딩 결과/벡터를 병합**하고 Dense layer을 통해 **시퀀스의 '다음 단어'를 생성**합니다.

<br>
<div align=center><img src="https://user-images.githubusercontent.com/38115693/153908121-a0cb87fc-0517-4551-8721-cc7c0bbe72dd.png" width="1000"></div>
<div align=center> Defined Model Structure </div>

### :muscle: 모델 학습 과정

- 모델 학습에는 다음의 input과 output이 필요합니다.
	- **Input** : (1) InceptionV3으로 추출한 이미지 특성 벡터, (2) 인코딩 된 캡션 텍스트 시퀀스
	- **Output**: one-hot 인코딩 된 '다음 단어'
- Data generator를 통해 모델에 하나의 **이미지 특성(X1)과 text sequence(X2)가 input으로 주어지면 다음 단어(y)가 학습** 됩니다. 이 생성된 **다음 단어는 다시 input으로 주어져 그 다음 단어(y)에 대한 학습**이 이루어집니다.
- 예를 들어, "two boys are playing baseball in the ground"라는 문장이 있다면, 이 하나의 문장과 해당 이미지에 대해 아래와 같이 처리되고 학습됩니다.

<div align=center><img src="https://user-images.githubusercontent.com/38115693/154433865-80921d06-15bc-420a-a487-169ceae72132.png" width="500"></div>
<div align=center> Input Data Structure </div>

---
## :cool: 캡션 생성 과정

**학습한 모델을 사용한 캡션 생성 과정**

<div align=center><img src="https://user-images.githubusercontent.com/38115693/153921982-d1373341-5fba-444a-87a2-d801ee68e28a.png" width="600"></div>
<div align=center> Caption Generation Process </div>
<br>

- 이미지에 대한 캡션을 생성할 때, 학습한 모델은 **시퀀스의 다음 단어를 예측**(단어들의 연속을 예측) 하는데, **모든 단어에 대한 확률분포**를 구하여 예측 힙니다.
- 캡션을 생성하는 기법은 일반적으로 사용되는 아래 두 기법을 사용했습니다.
	- **Greedy Search** : 전체 단어에서 각 단어에 대한 확률 분포를 예측하여 다음 단어를 선택하는데 **각 스텝에서 가장 가능성이/확률이 높은 단어를 선택**합니다.
	- **Beam Search** : Greedy Search의 단점을 보완하여 확장한 기법 입니다.

---
## :bar_chart: 캡셔닝 모델 평가 방법

**BLEU(Bilingual Evaluation Understudy) Score**

- 캡션 생성 평가 지표로는 기존의 이미지 캡션 논문들에서 사용되는 **BLEU 스코어**(BLEU-1, BLEU-2, BLEU-3, BLEU-4)를 사용했습니다.
- 연속된 n개의 단어를 기준으로 실제 캡션(ground truth)과 얼마나 공통적인 단어가 나왔는지/겹치는지를 판단해서 BLEU 점수를 계산합니다.
- 점수는 0.0~1.0 사이에서 나타내는데, **1.0에 가까울 수록, 높을수록 좋은 점수이며 뛰어난 캡셔닝 성능을 의미**합니다.

---
## :musical_keyboard: 모델링 과정

1. 캡션 텍스트 시퀀스 데이터에 대해 전처리와 토큰화를 합니다. 그리도 embedding vector를 준비합니다 (텍스트 시퀀스 데이터는 나중에 임베딩 되고 LSTM에서 처리됩니다).
2. 사전 훈련된 CNN 모델 InceptionV3를 사용하여 이미지 특성들을 추출합니다 (추출된 이미지 특성들은 나중에 LSTM 결과/출력과 병합됩니다).
3. 이미지 특성 데이터와 텍스트 시퀀스 데이터를 Train/Test 데이터로 나눕니다. 그리고 Train 텍스트 데이터를 기준으로 단어-인덱스 사전을 만들고 전체 단어 개수와 최대 캡션 길이를 구합니다. 이 때, 최소 빈도수 threshold를 설정해 학습에 사용할 단어 수를 줄이기도 합니다.
4. 모델을 구현(define)합니다. 이 때, 준비한 embedding vector를 사용해 모든 단어에 대한 embedding matrix를 만들어 모델의 embedding layer에 적용합니다.
5. 모델을 학습시킵니다.
6. 훈련된 모델을 사용해 이미지 캡션을 생성해 모델에 대한 정성적 평가를 합니다. Greedy Search와 Beam Search 모두를 사용합니다.
7. BLEU 평가를 통해 모델에 대한 정량적 평가를 합니다.

---
## :wrench: 모델링 세부 과정

<div align=center><img src="https://user-images.githubusercontent.com/38115693/155276817-554732e2-f039-492a-b503-4f6d6acce03c.png" width="500"></div>
<div align=center> Model Architecture of the Trials </div>
<br>

- 모델링은 아래의 요소들을 변경해가며 진행했습니다.
	- 데이터셋, 임베딩 방법, 텍스트 전처리 및 토큰화 방법, 단어 최소 빈도수 threshold, 에포크(epoch), 배치사이즈(batch size), 학습률(learning rate), 교차검증(cross validation)
- 이미지 특성 추출은 모든 과정에서 동일하게 pre-trained model인 InceptionV3를 사용했습니다.

|Options|Trial 1-1|Trial 1-2|Trial 2|Trial 3|Trial 4|Trial 5|Trial 6|
|---|---|---|---|---|---|---|---|
|Dataset|AI Hub 멀티모달 데이터셋|AI Hub 멀티모달 데이터셋|AI Hub MSCOCO 이미지 설명 데이터셋|AI Hub MSCOCO 이미지 설명 데이터셋|AI Hub MSCOCO 이미지 설명 데이터셋|AI Hub MSCOCO 이미지 설명 데이터셋|AI Hub MSCOCO 이미지 설명 데이터셋|
|Data type/size|약 26,000개 고유한 캡션 및 이미지|약 160,000개 연속된 이미지 및 캡션|약 120,000개 개별 이미지 및 캡션|약 120,000개 개별 이미지 및 캡션|약 120,000개 개별 이미지 및 캡션|약 120,000개 개별 이미지 및 캡션|약 120,000개 개별 이미지 및 캡션|
|Data splitting|Train(70%), Test(30%)|Train(70%), Test(30%)|Train(70%), Test(30%)|Train(70%), Test(30%)|Train(70%), Test(30%)|Train(70%), Test(30%)|Train(70%), Valid(15%), Test(15%)|
|Embedding|Glove로 학습 및 생성한 임베딩 matrix 사용|사전훈련된 FastText 임베딩 모델 사용|사전훈련된 FastText 임베딩 모델 사용|사전훈련된 FastText 임베딩 모델 사용|사전훈련된 FastText 임베딩 모델 사용|사전훈련된 FastText 임베딩 모델 사용|사전훈련된 FastText 임베딩 모델 사용|
|Text tokenization|어절(띄어쓰기) 단위|의미형태소 단위 및 어간 추출|의미형태소 단위 및 어간 추출|의미형태소 단위 및 어간 추출|의미형태소 단위 및 어간 추출|의미형태소+기능형태소 단위|의미형태소+기능형태소 단위|
|Word minimum frequency threshold|최소 빈도수 3 미만 단어 제외|최소 빈도수 threshold 미설정|최소 빈도수 10 미만 단어 제외|최소 빈도수 10 미만 단어 제외|최소 빈도수 10 미만 단어 제외|최소 빈도수 4 미만 단어 제외|최소 빈도수 4 미만 단어 제외|
|Training|Epoch 40, Batch Size 3, Learning Rate 0.001|Epoch 10, Batch Size 3, Learning Rate 0.001|Epoch 30, Batch Size 8, Learning Rate 0.001|Epoch 30, Batch Size 16, Learning Rate 0.001|Epoch 20, Batch Size 16, Learning Rate 0.001 + Epoch 10, Batch Size 32, Learning Rate 0.0001|Epoch 20, Batch Size 16, Learning Rate 0.001 + Epoch 10, Batch Size 32, Learning Rate 0.0001|Epoch 20, Batch Size 16, Learning Rate 0.001 + Epoch 10, Batch Size 32, Learning Rate 0.0001|

## :chart_with_upwards_trend: 모델 BLEU 평가 결과

<div align=center><img src="https://user-images.githubusercontent.com/38115693/154012751-6d4e564f-0b2f-404f-9408-e8a6084a65bd.png" width="500"></div>
<div align=center> BLEU Scores </div>
<br>

- 실험을 거치면서 BLEU 스코어가 지속적으로 증가했습니다.
- 결론적으로 마지막 실험 모델이 가장 높은 성능을 보였습니다.

<div align=center><img src="https://user-images.githubusercontent.com/38115693/154014256-c2a3248b-f0e9-41cb-a0ee-bd19170c3b8b.png" width="500"></div>
<div align=center> BLEU Scores Chart </div>

---
## :warning: 향후 과제

**모델의 캡션 예측/생성 성능 향상을 위한 시도**
1. 새로운 데이터를 더 확보하거나, AI Hub MSCOCO와 멀티모달 두 데이터를 합쳐 모델 학습 진행
3. 모델 아키텍쳐를 변경하여 모델링 (e.g. Bidirectional RNNs/LSTMs 사용, Attention 메커니즘 기법 사용 사용)
4. Pre-trained CNN 모델로 transfer learning시 여러 fine tuning 시도 (e.g. trainable layer 증가)
5. AI Hub 멀티모달 데이터셋을 이용한 모델링시 사전학습된 모델이 아닌 자체 CNN 모델을 설계하여 처음부터 학습하여 모델링
6. 여러 하이퍼파라미터 튜닝 시도 (e.g. learning rate, batch size, embedding dimension 300, number of layers, number of units, dropout rate, batch normalization 등 조정)
7. 영상을 표현하는 시각 특징 외에, 정적 그리고 동적 의미 특징들도 이용

**출력된 캡션에 대한 추가적인 처리**
1. 문장이 완전하지 않은 형태로 출력 되는 경우 존해
	- 예를 들어, "...에 서있다"가 맞는 형태이지만, "...에서 있다"로 출력이 되는 경우
	- 더 고도화한 문장 생성 및 출력을 위해 형태소 분석이나 관련 기능을 조사하여 적용 예정

---
## :game_die: 활용방안

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
## :books: 참조 문헌

- Vinyals, O., Toshev, A., Bengio, S., & Erhan, D. (2015). Show and tell: A neural image caption generator. In Proceedings of the IEEE conference on computer vision and pattern recognition (pp. 3156-3164).
- Masters, D., & Luschi, C. (2018). Revisiting small batch training for deep neural networks. arXiv preprint arXiv:1804.07612.
- Smith, S. L., Kindermans, P. J., Ying, C., & Le, Q. V. (2017). Don't decay the learning rate, increase the batch size. arXiv preprint arXiv:1711.00489.
- Kandel, I., & Castelli, M. (2020). The effect of batch size on the generalizability of the convolutional neural networks on a histopathology dataset. ICT Express, 6(4), 312–315. https://doi.org/10.1016/j.icte.2020.04.010
- Tanti, M., Gatt, A., & Camilleri, K. P. (2017). What is the role of recurrent neural networks (rnns) in an image caption generator?. arXiv preprint arXiv:1708.02043.
- Tanti, M., Gatt, A., & Camilleri, K. P. (2018). Where to put the image in an image caption generator. Natural Language Engineering, 24(3), 467-489.
- Mao, J., Xu, W., Yang, Y., Wang, J., Huang, Z., & Yuille, A. (2014). Deep captioning with multimodal recurrent neural networks (m-rnn). arXiv preprint arXiv:1412.6632.
- Park, Seong-Jae & Cha, Jeong-Won. (2017). Generate Korean image captions using LSTM. Proceedings of the Korean Society for Language and Information Conference, (), 82-84.
- Lamba, H. (2019, February 17). Image Captioning with Keras. Medium. https://towardsdatascience.com/image-captioning-with-keras-teaching-computers-to-describe-pictures-c88a46a311b8
- Brownlee, J. (2020, December 22). How to Develop a Deep Learning Photo Caption Generator from Scratch. Machine Learning Mastery. https://machinelearningmastery.com/develop-a-deep-learning-caption-generation-model-in-python/
- Data-Stats. (n.d.). Image-Captioning-using-Keras-and-Tensorflow. GitHub. https://github.com/data-stats/Image-Captioning-using-Keras-and-Tensorflow
- Tang. (n.d.). Image Caption using Neural Networks. GitHub. https://xiangyutang2.github.io/image-captioning/
- Captioning Leaderboard. (n.d.). COCO - Common Objects in Context. https://cocodataset.org/#captions-leaderboard
