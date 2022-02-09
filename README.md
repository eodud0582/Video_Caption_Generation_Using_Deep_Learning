# Deep Learning Based Video Caption Generator (배너 삽입)

# 딥러닝 기반 동영상 캡션 자동 생성
## 한국어 데이터 기반 동영상 캡션 자동 생성 모델 개발
---

## 프로젝트 목적

한국어 멀티모달 데이터를 활용하여, 이용자가 동영상을 입력하면 동영상의 상황을 묘사하는 캡션을 만들어주는 **CNN+LSTM 기반의 동영상 캡셔닝 모델을 개발**하였습니다.

## 프로젝트 결과

![captioned_office](https://user-images.githubusercontent.com/38115693/153170767-f47adbe6-db3f-4bae-abc0-58c47bdac226.gif)![captioned_people_walking](https://user-images.githubusercontent.com/38115693/153171395-11b209c1-f0b5-4075-8c43-759b69a278a5.gif)

![captioned_cats](https://user-images.githubusercontent.com/38115693/153171356-24403b58-fa3c-482b-b9c7-b558b45ca465.gif)![captioned_black_cat](https://user-images.githubusercontent.com/38115693/153180977-65572efb-2083-4981-a3ed-12c9ebbbbb00.gif)

![captioned_seoul_night_city](https://user-images.githubusercontent.com/38115693/153171118-a93533c4-4e47-408a-ba8d-50f0597c0adb.gif)![captioned_seoul_road_street](https://user-images.githubusercontent.com/38115693/153171144-62f41be7-4bad-45bd-a2c7-0b786e4661a1.gif)

![captioned_KETI_MULTIMODAL_0000000695](https://user-images.githubusercontent.com/38115693/153171433-5aca8f3d-5832-4004-a7db-b63d7bc3a371.gif)![captioned_KETI_MULTIMODAL_0000000215](https://user-images.githubusercontent.com/38115693/153171459-394a5fcc-deba-45e5-845a-aa05a327a0e9.gif)

<img src="https://user-images.githubusercontent.com/38115693/153171626-08746848-62f4-479c-960c-18478380bf33.gif" width="256"><img src="https://user-images.githubusercontent.com/38115693/153171646-18e6adac-ed1f-4f6f-9bf2-1a8a1d045300.gif" width="256"><img src="https://user-images.githubusercontent.com/38115693/153171658-c2d88d7b-4d85-4cde-a52d-425a7b948c36.gif" width="256">

---
## 프로젝트 배경

**멀티모달(Multimodality)**
- '멀티모달'이란 언어뿐만 아니라 이미지, 영상에 이르기까지 사람의 의사소통과 관련된 다양한 정보를 습득하고 다룰 수 있는 기술을 의미합니다. 딥러닝에 대한 관심이 증가하면서 컴퓨터가 시각적 이미지를 인식하여 사람처럼 문장으로 표현하는 것에 대한 연구 또한 활발히 진행되어 왔습니다. 이런 멀티모달 기술을 사용하여, **이미지나 영상에 대한 설명글(캡션)이 자동으로 생성된다면 일상생활 속에서 다양한 서비스를 제공** 할 수 있을 것이라 생각하여 프로젝트를 진행했습니다. 

**한국어 데이터 기반**
- 한국어 데이터의 부족으로 한국어를 대상으로 한 동영상/이미지 캡션 연구는 활발히 이루어지지 못했습니다. **기존 동영상/이미지 캡션 연구와 개발도 대다수 영어로 공개된 데이터셋이 이용**되었으며, 한글 캡션을 생성하기 위해서는 **영어 데이터를 번역하여 사용하거나 캡션 결과를 번역**해야 했습니다. 하지만 국내에서도 한국어 캡션 데이터셋를 최근에서야 제공하고 있습니다. 이를 활용하여, **국내 환경과 한국어에 맞는 동영상/이미지 캡션 생성 모델 개발이 가능**해 졌습니다. 

---
## 프로젝트 기간

(타임테이블)

---

구체적인 설명(모델 구조, 학습 원리, 캡션 생성 원리, 평가 방법, 데이터 소개, 데이터 정리/전처리, 모델링 등)


---
## 활용방안

캡션 생성은 이미지나 영상을 설명하는 캡션을 자동으로 생성하는 기술로 이미지/시각 처리 기술과 자연어 처리 기술이 합쳐진 어려운 기술이지만, 이미지 검색, 유아 교육, 청각 장애인들을 위한 캡셔닝 서비스와 같은 응용에 사용될 수 있는 중요한 기술입니다.

**1. 사진이나 동영상에 대한 검색/SEO 개선과 컨텐츠 접근성 향상**
- 시청자들은 동영상 속 상황이나 내용을 더 쉽게 이해할 수 있으며, 플랫폼이나 컨텐츠 제작자는 영상에 대한 검색엔진최적화, 즉 SEO를 개선할 수 있습니다. 구글링을 통한 이미지나 영상 검색에 대한 검색의 질도 향상 시킬 수 있을 것입니다.

**2. 시각 또는 청각이 불편한 사람들에게 사진/동영상을 설명**
- 캡셔닝 기술을 전자고글이나 카메라 같은 디바이스에 접목시킨다면 시각 장애를 가진 사람들에게 영상에 대한 설명을 음성으로 제공하거나 길안내 등에 사용 될 수 있으며, 청각이 불편한 사람들에겐 영상에 캡션이 함께 제공되어 영상 속 상황이나 행동을 더 쉽게 이해할 수 있어 미디어/컨텐츠에 대한 접근성/접근 환경을 개선 할 수 있습니다.

**3. 미술 심리 치료**
- 입력된 그림을 기록하여 전문 상담사들이 그림에 대한 질문을 제시하여 내면의 심리를 상세하게 기록할 수 있도록 돕는 등 미술 심리치료에도 활용될 수 있습니다.

**4. 언어 교육**
- 이미지나 동영상에 대한 한국어 설명글을 통해 아동이나 외국인에게 언어 교육도 제공 할 수 있을 것입니다. 

---
## 마무리

(한계점, 보완점, 향후 과제 등)

모델의 캡션 예측/생성 성능 향상을 위한 시도
- 더 많은 데이터를 사용하여 학습한다면, 캡션 예측/생성 성능이 더 좋아질 것으로 생각합니다. 데이터를 더 확보하거나 AI허브 MSCOCO와 멀티모달 두 데이터를 합쳐 모델 학습을 진행해 보겠습니다.
- AI허브 멀티모달 영상 데이터에 대해 pre-trained CNN 모델이 아닌, 기본 CNN 모델을 사용하여 학습 및 특성 추출
- 모델 아키텍쳐를 변경 (e.g. Attention 메커니즘 기법 적용)
- 하이퍼파라미터 튜닝 (e.g. learning rate, batch size, number of layers, number of units, dropout rate, batch normalization 등 조정)
- 영상을 표현하는 시각 특징 외에, 정적 그리고 동적 의미 특징들도 이용

출력된 캡션에 대한 추가적인 처리
- 문장이 완전하지 않은 형태로 출력 되는 경우가 있습니다. 예를 들어, "...에 서있다"가 맞는 형태이지만, "...에서 있다"로 출력이 되는 경우입니다. 더 고도화한 문장 생성 및 출력을 위해 형태소 분석이나 관련 기능을 조사하여 적용이 필요합니다.



