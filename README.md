# video_caption_generator

프로젝트 목적

멀티모달 데이터를 활용하여 텍스트와 이미지를 동시에 이해하는 AI 모델을 개발하여, 이용자가 영상을 입력하면 영상의 상황을 묘사하는 캡션을 만들어주는 영상 캡셔닝 모델이다. 

프로젝트 상세 설명
1. 프로젝트 배경

<img src="https://user-images.githubusercontent.com/38115693/153191920-329cb4ab-5a6c-487c-8419-9f0bed6873f5.png" width="400"><img src="https://user-images.githubusercontent.com/38115693/153191324-276f0b4f-6a31-4f0e-974d-b4c30d91a0f3.png" width="400">

멀티모달에 대한 관심 증가
- '멀티모달'이란 언어뿐만 아니라 이미지, 영상에 이르기까지 사람의 의사소통과 관련된 다양한 정보를 습득하고 다룰 수 있는 기술을 의미합니다.
- 딥러닝에 대한 관심이 증가하면서 컴퓨터가 시각적 이미지를 인식하여 문장으로 표현하는 것에 대한 연구도 활발히 진행되어 왔습니다. 그리고 최근에는 여러 IT 기업들이 멀티모달 AI를 개발하고 활용하는 등 다양한 형태의 데이터를 학습하는 멀티모달에 대한 관심이 증가하고 있습니다. 멀티모달 기반으로, 영상에 대한 설명 글이 자동으로 생성된다면 일상생활 속에서 다양한 서비스를 제공 할 수 있지 않을까 생각하여 프로젝트를 시작하게 되었습니다.

한국어 캡셔닝 모델 개발의 필요성
- 한국어 데이터는 부족했기 때문에 한국어를 대상으로 한 이미지나 영상 캡션 연구는 활발히 이루어지지 못했으며, 기존 이미지 캡션 연구나 개발도 대다수 영어로 공개된 데이터 셋이 이용되었습니다. 또, 한글 캡션을 생성하기 위해서는 영어 데이터를 번역하여 사용하거나 캡션 결과를 번역해야 했습니다.
- 하지만 국내에서도 한국어 이미지 캡션 데이터를 최근에서야 제공하고 있습니다. 이를 잘 활용하여, 국내 환경과 한국어에 맞는 영상/이미지 캡션 생성 모델 개발이 가능할 것입니다.


하지만 국내에서도 한국어 이미지 캡션 데이터를 최근에서야 제공
이를 활용하여 국내 환경과 한국어에 맞는 영상/이미지 캡션 모델 개발이 가능함
![image](https://user-images.githubusercontent.com/38115693/153189495-209e7e60-f511-48a3-97f9-900405f792ef.png)



3. 설명(모델 구조, 학습 원리, 캡션 생성 원리, 평가 방법, 데이터 소개, 데이터 정리/전처리, 모델링 등)
4. 프로젝트 기간(타임테이블)
5. 프로젝트 결과

![captioned_office](https://user-images.githubusercontent.com/38115693/153170767-f47adbe6-db3f-4bae-abc0-58c47bdac226.gif)![captioned_people_walking](https://user-images.githubusercontent.com/38115693/153171395-11b209c1-f0b5-4075-8c43-759b69a278a5.gif)

![captioned_cats](https://user-images.githubusercontent.com/38115693/153171356-24403b58-fa3c-482b-b9c7-b558b45ca465.gif)![captioned_black_cat](https://user-images.githubusercontent.com/38115693/153180977-65572efb-2083-4981-a3ed-12c9ebbbbb00.gif)

![captioned_seoul_night_city](https://user-images.githubusercontent.com/38115693/153171118-a93533c4-4e47-408a-ba8d-50f0597c0adb.gif)![captioned_seoul_road_street](https://user-images.githubusercontent.com/38115693/153171144-62f41be7-4bad-45bd-a2c7-0b786e4661a1.gif)

![captioned_KETI_MULTIMODAL_0000000695](https://user-images.githubusercontent.com/38115693/153171433-5aca8f3d-5832-4004-a7db-b63d7bc3a371.gif)![captioned_KETI_MULTIMODAL_0000000215](https://user-images.githubusercontent.com/38115693/153171459-394a5fcc-deba-45e5-845a-aa05a327a0e9.gif)

<img src="https://user-images.githubusercontent.com/38115693/153171626-08746848-62f4-479c-960c-18478380bf33.gif" width="256"><img src="https://user-images.githubusercontent.com/38115693/153171646-18e6adac-ed1f-4f6f-9bf2-1a8a1d045300.gif" width="256"><img src="https://user-images.githubusercontent.com/38115693/153171658-c2d88d7b-4d85-4cde-a52d-425a7b948c36.gif" width="256">

6. 활용방안
7. 마무리(한계점, 보완점, 향후 과제 등)




