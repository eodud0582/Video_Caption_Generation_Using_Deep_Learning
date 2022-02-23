# Required installments
# ==============================================================================
'''
-------- 사전 설치 필요 --------
# 필요한 라이브러리/패키지의 원활한 설치를 위해 Google Colab 사용 권장

- Tensorflow: pip install --upgrade tensorflow
- OpenCV: pip install opencv-python
- Pillow: pip install --upgrade Pillow
- KoNLPy: pip install konlpy
- scikit-image: pip install -U scikit-image

- 한글 폰트 (원하는 한글 폰트 .ttf 파일을 직접 다운로드하여 사용 가능)
sudo apt-get install -y fonts-nanum
sudo fc-cache -fv

'''
# ==============================================================================
# Import required libraries/packages
import os
import sys
import time
import pickle
from time import time
from tqdm import tqdm

import cv2
import numpy as np
from PIL import Image, ImageFont, ImageDraw

from konlpy.tag import Kkma
from skimage.metrics import structural_similarity as ssim

import tensorflow
from tensorflow.keras.models import Model
import tensorflow.keras.preprocessing.image
import tensorflow.keras.applications.inception_v3
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.preprocessing.sequence import pad_sequences

# ==============================================================================
# 캡션 생성을 위한 준비/세팅
# wordtoidx, idxtoword, vocab_size, max_length 구하기
'''
- 캡션 생성에 필요한 데이터 불러오기
- 추론에 필요한 정보/값 구하기 (wordtoidx, idxtoword, vocab_size, max_length)
- 학습된 모델 다시 불러오기
'''
# train descriptions 불러오기
import pickle
with open("train_descriptions.pkl", "rb") as f:
    train_descriptions = pickle.load(f)

# 모든 (train용) descriptions(captions)을 리스트로 저장
# train data만 진행
all_train_captions = []
for key, val in train_descriptions.items():
    for cap in val:
        all_train_captions.append(cap)
print('all train descriptions:', len(all_train_captions))

# 빈도수가 지정한 숫자보다 적은 단어 제외
# 빈도수가 너무 적은 단어까지 포함시켜 학습을 하게 되면 시간이 많이 소요될 뿐만 아니라 정확도가 낮아질 수 있음
word_count_threshold = 4
word_counts = {}
nsents = 0
for sent in all_train_captions:
    nsents += 1
    for w in sent.split(' '):
        word_counts[w] = word_counts.get(w, 0) + 1
vocab = [w for w in word_counts if word_counts[w] >= word_count_threshold]
print('preprocessed words %d ==> %d' % (len(word_counts), len(vocab)))

# 모든 {단어 : 인덱스 번호} 구하기
# wordtoidx / idxtoword 딕셔너리 생성
idxtoword = {} # {인덱스 : 단어}
wordtoidx = {} # {단어 : 인덱스}
ix = 1
for w in vocab:
    wordtoidx[w] = ix
    idxtoword[ix] = w
    ix += 1

# vocab size 구하기
# we append 1 to our vocabulary since we append 0’s to make all captions of equal length
vocab_size = len(idxtoword) + 1 
print('vocab size:', vocab_size)

# max_length 구하기
max_length = max(len(d.split()) for d in all_train_captions)
print('description max length: %d' % max_length)

# pre-trained Inception model 불러오기 및 학습
encode_model = InceptionV3(weights='imagenet') #InceptionV3 사용
encode_model = Model(encode_model.input, encode_model.layers[-2].output)
WIDTH = 299
HEIGHT = 299
OUTPUT_DIM = 2048 #output : 2048 / OPUTPUT_DIM은 어떤 pre-trained CNN 모델을 쓰느냐에 따라 다르다
preprocess_input = tensorflow.keras.applications.inception_v3.preprocess_input

# 이미지 전처리 및 인코딩 함수
# 이미지 전처리, 파라미터 값 지정, 그리고 벡터로 변환
def encodeImage(img):
    # 이미지 사이즈를 표준크기로 재조정
    img = img.resize((WIDTH, HEIGHT), Image.ANTIALIAS) # eshape the images to (299 x 299) since we are using InceptionV3
    #  PIL 이미지를 numpy array로 변경; Convert PIL image to numpy array of 3-dimensions
    x = tensorflow.keras.preprocessing.image.img_to_array(img)
    # 2D array로 확장; Add one more dimension
    x = np.expand_dims(x, axis=0)
    # InceptionV3의 인풋을 위한 전처리; preprocess images using preprocess_input() from inception module
    x = preprocess_input(x)
    # 인코딩 벡터 반환
    x = encode_model.predict(x)
    # LSTM을 입력을 위한 shape 조정
    x = np.reshape(x, OUTPUT_DIM) # reshape from (1, 2048) to (2048, ); np.reshape(x, x.shape[1])
    return x

# 학습된 모델 불러오기
from tensorflow import keras
model = keras.models.load_model('caption_generation_model.h5')

# ==============================================================================
# 필요한 함수들 생성
'''
캡션 출력 함수
- Greedy Search
- Beam Search

이미지 유사도 측정 함수
- MSE
- SSIM
'''
# Greedy Search prediction
def greedy_search_prediction(photo):
    # seed the generation process
    in_text = 'startseq'
    # iterate over the whole length of the sequence
    for i in range(max_length):
        sequence = [wordtoidx[w] for w in in_text.split() if w in wordtoidx] # encode the input sequence to integer
        sequence = pad_sequences([sequence], maxlen=max_length) # pad the input; padding
        yhat = model.predict([photo, sequence], verbose=0) # predict next word
        yhat = np.argmax(yhat) # convert probability to integer
        word = idxtoword[yhat] # map/convert integer to word/text
        # stop if we cannot map the word
        if word is None:
            break
        # append as another input for generating the next word
        in_text += ' ' + word
        # stop if we predict the end of the sequence
        if word == 'endseq':
            break
    final = in_text.split()
    final = final[1:-1]
    final = ' '.join(final)
    return final

# Beam Search prediction
def beam_search_prediction(image, beam_index = 3):
    start = [wordtoidx["startseq"]]
    start_word = [[start, 0.0]]
    while len(start_word[0][0]) < max_length:
        temp = []
        for s in start_word:
            par_caps = pad_sequences([s[0]], maxlen=max_length, padding='post')
            preds = model.predict([image,par_caps], verbose=0)
            word_preds = np.argsort(preds[0])[-beam_index:]
            # Getting the top <beam_index>(n) predictions and creating a 
            # new list so as to put them via the model again
            for w in word_preds:
                next_cap, prob = s[0][:], s[1]
                next_cap.append(w)
                prob += preds[0][w]
                temp.append([next_cap, prob])   
        start_word = temp
        # Sorting according to the probabilities
        start_word = sorted(start_word, reverse=False, key=lambda l: l[1])
        # Getting the top words
        start_word = start_word[-beam_index:]
    start_word = start_word[-1][0]
    intermediate_caption = [idxtoword[i] for i in start_word]
    final_caption = []
    for i in intermediate_caption:
        if i != 'endseq':
            final_caption.append(i)
        else:
            break
    final_caption = ' '.join(final_caption[1:])
    return final_caption

# 이미지 유사도 측정 - mse
def mse(imageA, imageB):
    # the 'Mean Squared Error' between the two images is the sum of the squared difference between the two images;
    # NOTE: the two images must have the same dimension
    err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
    err /= float(imageA.shape[0] * imageA.shape[1])
    # return the MSE, the lower the error, the more "similar" the two images are
    return err

# 이미지 유사도 측정 - ssim
def compare_images(imageA, imageB):
    # compute the mean squared error and structural similarity index for the images
    #m = mse(imageA, imageB)
    s = ssim(imageA, imageB)
    return s

# ==============================================================================
# 동영상 캡션 생성
'''
- 영상을 입력 받으면 frame 이미지들로 나눠서 captioning 처리
- frame 이미지들은 별도로 폴더에 저장하지 않고 바로 처리
- 처리가 완료 되면 caption generate 된 동영상 파일을 결과로 반환
'''
start_time = time.time()

# read video file
video_name = 'your_video_file_name' # 동영상 파일 이름
base_path = '/content/drive/MyDrive/Colab Notebooks' # 기본 경로 입력

video_id = video_name + '.mp4' # .avi 등 읽어들일 동영상 파일 형식에 맞게 입력
video_path = os.path.join(base_path, video_id) # 동영상 파일 경로
cap = cv2.VideoCapture(video_path)

# video에서 frame image 추출
print('extracting video frame images...')
count = 0
image_dict = {}
while cap.isOpened():
    ret, frame = cap.read()
    if ret is False:
        break
    image_id = 'frame_{}.jpg'.format(count)
    image_dict[image_id] = frame
    count += 1
cap.release()
cv2.destroyAllWindows()
print('total video frame images:', count)

# frame 이미지들 특성 추출
print('extracting features from video frame images...')
test_features = {}
for i in tqdm(range(len(image_dict))):
    image_id = 'frame_{}.jpg'.format(i)
    # 이미지 배열을 pillow format에 적합하게 변환 
    image_arrays = cv2.cvtColor(image_dict[image_id], cv2.COLOR_BGR2RGB)
    # load and change numpy array of image into image in Pillow format, PIL Image instance
    img = tensorflow.keras.preprocessing.image.array_to_img(image_arrays)
    # frame 이미지별 특성 저장
    test_features[image_id] = encodeImage(img)
print('total extracted image features:', len(test_features))

# caption generation 및 video 생성
print('generating captions...')
kkma = Kkma()
img_array = []
prev_img = []
for i in tqdm(range(len(image_dict))): 
    # 이미지 id
    image_id = 'frame_{}.jpg'.format(i)
    
    # 이미지 load
    img = image_dict[image_id] # .astype(np.uint8)
    if img is None:
        print('Image load failed!')
        sys.exit()
    
    # 입력받은 이미지의 특성 불러오기
    image_feature = test_features[image_id].reshape((1,OUTPUT_DIM)) # OUTPUT_DIM = 2048
    
    # 이미지 width, height 저장
    height, width, layers = img.shape
    size = (width, height)
    
    # 이미지 유사도 분석을 위한 gray image
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 한글 캡션 출력을 위해 Pillow 사용 -- img 배열을 Pillow가 처리 가능하게 변환 (OpenCV에선 한글 깨짐)
    img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    
    # 읽어들인 이미지에 caption 삽입 시작
    draw = ImageDraw.Draw(img) # PIL ImageDraw 객체 생성
    fontsize = round(height * 0.02) # 폰트 크기를 이미지 크기에 비례하게 설정
    font = ImageFont.truetype('./fonts/NanumGothic.ttf', fontsize) # 사용할 한글 폰트 및 글자 크기 지정

    # pick greedy search (True) / beam search (False)
    use_greedy = True
    
    # beam search 사용시 사용할 beam_index 설정
    if use_greedy == False:
        beam_index = 3

    if not len(prev_img):
        prev_img = img_gray

        if use_greedy == True:
            greedy_caption = greedy_search_prediction(image_feature)
            greedy_caption = ''.join(greedy_caption.split(' '))
            greedy_caption = ' '.join(kkma.sentences(greedy_caption)) # sentence detection
            prev_greedy_caption = greedy_caption
        else:
            beam_caption = beam_search_prediction(image_feature, beam_index=beam_index)
            beam_caption = ''.join(beam_caption.split(' '))
            beam_caption = ' '.join(kkma.sentences(beam_caption)) # sentence detection
            prev_beam_caption = beam_caption
    
    else:
        ssim_result = compare_images(prev_img, img_gray) # 이미지 유사도 측정
        prev_img = img_gray
        
        if ssim_result >= 0.80: # 이미지 유사도 threshold
            if use_greedy == True:
                greedy_caption = prev_greedy_caption
            else:
                beam_caption = prev_beam_caption
        else:
            if use_greedy == True:
                greedy_caption = greedy_search_prediction(image_feature)
                greedy_caption = ''.join(greedy_caption.split(' '))
                greedy_caption = ' '.join(kkma.sentences(greedy_caption))
                prev_greedy_caption = greedy_caption
            else:
                beam_caption = beam_search_prediction(image_feature, beam_index=beam_index)
                beam_caption = ''.join(beam_caption.split(' '))
                beam_caption = ' '.join(kkma.sentences(beam_caption))
                prev_beam_caption = beam_caption

    if use_greedy == True:
        caption = greedy_caption
    else:
        caption = beam_caption
    
    text_w, text_h = draw.textsize(caption, font) # 캡션 글자의 width, height
    org = ((width-text_w)/2, (height*0.9)) # 캡션 위치 설정
    draw.text(org, caption, font=font, fill=(255,255,255)) # (255,255,255): 글자 색 흰색으로 설정

    # 다시 OpenCV가 처리가능하게 numpy 배열 및 BGR로 변환
    img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

    # img_array에 append
    img_array.append(img)

# 동영상 writer 객체 생성
print('creating captioned video...')
captioned_video_name = 'captioned_' + video_name + '.avi' # 동영상 파일 이름 / 저장 경로
out = cv2.VideoWriter(captioned_video_name, cv2.VideoWriter_fourcc(*'DIVX'), 30, size)
'''
참고로 mp4가 더 확장성이 있으며 모든 기기와 호환 가능하니, 필요시 변경
'''

# 동영상 생성
for i in range(len(img_array)):
    out.write(img_array[i]) # out.write(frame)을 호출하면 현재 frame이 저장
out.release()

print('video captioning completed')
print("--- %s seconds ---" % (time.time() - start_time))