# Deep Learning Based Video Caption Generator (배너 삽입)

# 딥러닝 기반 동영상 캡션 자동 생성
### 한국어 데이터 및 딥러닝을 활용한 동영상 캡션 자동 생성 모델 개발

---
## 프로젝트 목적

- 한국어 멀티모달 데이터를 활용하여, 이용자가 동영상을 입력하면 동영상의 상황을 묘사하는 캡션을 만들어주는 **CNN+LSTM 기반의 동영상 캡셔닝 모델을 개발**하였습니다.

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

---
## 프로젝트 배경

**멀티모달(Multimodality)**
- '멀티모달'이란 언어뿐만 아니라 이미지, 영상에 이르기까지 사람의 의사소통과 관련된 다양한 정보를 습득하고 다룰 수 있는 기술을 의미합니다. 딥러닝에 대한 관심이 증가하면서 컴퓨터가 시각적 이미지를 인식하여 사람처럼 문장으로 표현하는 것에 대한 연구 또한 활발히 진행되어 왔습니다. 이런 멀티모달 기술을 사용하여, **이미지나 영상에 대한 설명글(캡션)이 자동으로 생성된다면 일상생활 속에서 다양한 서비스를 제공** 할 수 있을 것이라 생각하여 프로젝트를 진행했습니다. 

**한국어 데이터 기반**
- 한국어 데이터의 부족으로 한국어를 대상으로 한 동영상/이미지 캡션 연구는 활발히 이루어지지 못했습니다. **기존 동영상/이미지 캡션 연구와 개발도 대다수 영어로 공개된 데이터셋이 이용**되었으며, 한글 캡션을 생성하기 위해서는 **영어 데이터를 번역하여 사용하거나 캡션 결과를 번역**해야 했습니다. 하지만 국내에서도 한국어 캡션 데이터셋를 최근에서야 제공하고 있습니다. 이를 활용하여, **국내 환경과 한국어에 맞는 동영상/이미지 캡션 생성 모델 개발이 가능**해 졌습니다. 

---
구체적인 설명(모델 구조, 학습 원리, 캡션 생성 원리, 평가 방법, 데이터 소개, 데이터 정리/전처리, 모델링 등)

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

### 동영상 캡셔닝 모델 구조

<div align=center><img src="https://user-images.githubusercontent.com/38115693/153815473-2fff29db-1349-4cae-8f2e-092341d32f2e.png" width="600"></div>

- 입력받은 동영상을 여러 **프레임**으로 나누고, **이미지 캡션 생성 모델**을 통해 각 프레임 이미지에 대한 캡션을 생성합니다.
- 각 프레임 이미지에 대해 생성된 캡션을 해당 프레임 이미지에 출력하고, 프레임 이미지들을 다시 동영상으로 변환합니다.
- 동영상이 플레이 될 때 캡션이 시시각각 변하는 경우들로 인해 캡션을 보는 것에 불편함이 있었습니다. 이에 대해, **SSIM 이미지 유사도 분석**을 통해 비슷한 프레임/장면에서는 동일한 캡션을 출력하게 만들어 동영상 속 출력된 캡션이 부드럽게 전환되고, 보기 편하게 만들었으며, 캡셔닝 처리에 걸리는 시간도 단축시켰습니다.

### 이미지 캡셔닝 모델 구조

<div align=center><img src="https://user-images.githubusercontent.com/38115693/153812190-f106a7e1-416e-45ff-80fd-16dcf1262722.png" width="500"></div>
<div align=center> Merge Architecture for Encoder-Decoder Model in Caption Generation </div>
<br>

- 이미지 캡셔닝 딥러닝 모델은 **Encoder-Decoder** architecture를 기반으로 만들었습니다.
	- Encoder: 이미지와 텍스트를 읽어 고정된 길이의 벡터로 인코딩하는 network model
	- Decoder: 인코딩된 이미지와 텍스트를 이용해 텍스트 설명을 생성하는 network model
- 그리고 Encoder-Decoder architecture 구현을 위해 Marc Tanti, et al.(2017)가 제시한 **Merge** 모델을 사용하였습니다. Merge architecture에서는 **이미지와 언어 정보가 별도로 인코딩** 되며, 이후 **multimodal layer** architecture의 Feedforward Network(FF)에서 병합(merge)되어 함께 처리됩니다.
	- CNN을 encoder로, RNN을 decoder로 사용한 기존의 Inject architecture와 비교하여, Merge 모델을 사용하면 텍스트 데이터를 인코딩하고 이해하는 데에만 RNN을 사용 할 수 있고, 인코딩에 pretrained language model을 사용 할 수 있는 등의 장점이 있고, 또한 몇몇 연구에선 Merge 모델이 더 나은 캡션 생성 성능을 보였습니다.

<br>
<div align=center><img src="https://user-images.githubusercontent.com/38115693/153630047-befa082e-c486-45ea-ab70-2aabad793d2a.png" width="500"></div>
<div align=center> RNN as Language Model </div>
<br>

- CNN을 이미지 인코딩을 위한 '이미지 모델'로, RNN/LSTM을 text sequence를 인코딩하는 '언어 모델'로 사용하였습니다. (CNN-LSTM)
	- 이미지 인코딩을 위해 ImageNet 데이터셋으로 pre-trained 된 모델을 사용하였으며, 다른 pre-trained 모델에 비해 상대적으로 training parameters가 더 적으면서도 더 우수한 성능을 가진 Inception V3를 사용해 전이학습(transfer learning) 했습니다. 그리고 과적합을 줄이기 위해 Dropout을 적용했습니다.
	- Text sequence 인코딩을 위해 pre-trained model인 FastText를 최종적으로 사용하여 embedding layer에서 모든 단어를 200차원 벡터로 매핑하였습니다. 뒤이어 RNN layer인 LSTM을 사용했습니다. 
- Decoder 모델에서 이렇게 각각 따로 처리된 이미지와 텍스트 두 입력 모델의 인코딩 결과/벡터를 병합(merge)하고, Dense layer을 통해 시퀀스에서 다음 단어를 예측해 가며 캡션을 생성합니다.

<div align=center><img src="https://user-images.githubusercontent.com/38115693/153851485-611ab3db-81cd-4649-a5fa-9658d997eca0.png" width="600"></div>

Merge Architecture 장점:
- The merging of image features with text encodings to a later stage in the architecture is advantageous and can generate better quality captions with smaller layers than the traditional inject architecture (CNN as encoder and RNN as a decoder).
- Several studies have also proven that merging architectures works better than injecting architectures for some cases.

---




Used an Encoder-Decoder model
- encoder model merges both the encoded form of the image and the encoded form of the text caption. The combination of these two encoded inputs is then used by a very simple decoder model to generate the next word in the sequence.
- model will treat CNN as the ‘image model’ and the RNN/LSTM as the ‘language model’ to encode the text sequences of varying length. The vectors resulting from both the encodings are then merged and processed by a Dense layer to make a final prediction.

- The merge model combines both the encoded form of the image input with the encoded form of the text description generated so far.
The combination of these two encoded inputs is then used by a very simple decoder model to generate the next word in the sequence.
The approach uses the recurrent neural network only to encode the text generated so far.


"In the case of ‘merge’ architectures, the image is left out of the RNN subnetwork, such that the RNN handles only the caption prefix, that is, handles only purely linguistic information. After the prefix has been vectorised, the image vector is then merged with the prefix vector in a separate ‘multimodal layer’ which comes after the RNN subnetwork"
— Where to put the Image in an Image Caption Generator, 2017.


CNN-LSTM

The main approach to this image captioning is in three parts:

- Photo Feature Extractor. This is a 16-layer VGG model pre-trained on the ImageNet dataset. We have pre-processed the photos with the VGG model (without the output layer) and will use the extracted features predicted by this model as input.
- Sequence Processor. This is a word embedding layer for handling the text input, followed by a Long Short-Term Memory (LSTM) recurrent neural network layer.
- Decoder (for lack of a better name). Both the feature extractor and sequence processor output a fixed-length vector. These are merged together and processed by a Dense layer to make a final prediction.

- The Photo Feature Extractor model expects input photo features to be a vector of 4,096 elements. These are processed by a Dense layer to produce a 256 element representation of the photo.
- The Sequence Processor model expects input sequences with a pre-defined length (34 words) which are fed into an Embedding layer that uses a mask to ignore padded values. This is followed by an LSTM layer with 256 memory units.
- Both the input models produce a 256 element vector. Further, both input models use regularization in the form of 50% dropout. This is to reduce overfitting the training dataset, as this model configuration learns very fast.
- The Decoder model merges the vectors from both input models using an addition operation. This is then fed to a Dense 256 neuron layer and then to a final output Dense layer that makes a softmax prediction over the entire output vocabulary for the next word in the sequence.

---
when the model is used to generate descriptions, the generated words will be concatenated and recursively provided as input to generate a caption for an image.

The function below named create_sequences(), given the tokenizer, a maximum sequence length, and the dictionary of all descriptions and photos, will transform the data into input-output pairs of data for training the model. There are two input arrays to the model: one for photo features and one for the encoded text. There is one output for the model which is the encoded next word in the text sequence.

The input text is encoded as integers, which will be fed to a word embedding layer.
The photo features will be fed directly to another part of the model.
The model will output a prediction, which will be a probability distribution over all words in the vocabulary.

---

캡션 생성

we need to be able to generate a description for a photo using a trained model.

This involves passing in the start description token ‘startseq‘, generating one word, then calling the model recursively with generated words as input until the end of sequence token is reached ‘endseq‘ or the maximum description length is reached.

The function below named generate_desc() implements this behavior and generates a textual description given a trained model, and a given prepared photo as input. It calls the function word_for_id() in order to map an integer prediction back to a word.

To generate the caption we will be using two popular methods which are Greedy Search and Beam Search. These methods will help us in picking the best words to accurately define the image.

---
## 모델링 과정

proceeded with the following steps.

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



