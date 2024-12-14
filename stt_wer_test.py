import pyttsx3
import speech_recognition as sr
import os
import requests
import json
import base64
from pydub import AudioSegment
from pydub.playback import play
import librosa
import numpy as np
from scipy.spatial.distance import cosine
from dtw import accelerated_dtw
from jiwer import wer
import re


# ETRI API 키 (공백 삭제 후 사용)
API_KEY = "067ea6f9-1715-43ab-814f-e23876886b9b"

engine = pyttsx3.init()

# TTS 설정 (예: 한국어)
engine.setProperty('rate', 150)  # 속도 설정
engine.setProperty('volume', 1)  # 볼륨 설정
engine.setProperty('language', 'ko')

# 표준 발음으로 변환할 텍스트
text = "안녕하세요반갑습니다"

# 표준 발음 파일 경로 설정
REFERENCE_AUDIO_PATH = os.path.join(os.getcwd(), 'standard_pronunciation.wav')

# TTS로 음성 파일 저장
engine.save_to_file(text, REFERENCE_AUDIO_PATH)
engine.runAndWait()  # 음성 파일 저장 후 대기

if os.path.exists(REFERENCE_AUDIO_PATH):
    print(f"표준 발음 파일이 저장되었습니다: {REFERENCE_AUDIO_PATH}")
else:
    print("표준 발음 파일이 저장되지 않았습니다.")


# 음성 인식 설정
recognizer = sr.Recognizer()
microphone = sr.Microphone()

#음성 인식
def recognize_speech_from_mic():
    """실시간으로 마이크에서 음성을 받아 텍스트로 변환 및 저장"""
    with microphone as source:
        print("말씀해주세요...")
        recognizer.adjust_for_ambient_noise(source)  # 주변 소음 조정
        audio = recognizer.listen(source)  # 음성을 들음

    try:
        print("음성을 녹음 중...")
        # 녹음된 오디오를 파일로 저장
        audio_path = "user_audio.wav"
        with open(audio_path, "wb") as f:
            f.write(audio.get_wav_data())
        return audio_path
    except Exception as e:
        print(f"오디오 저장 중 오류 발생: {e}")
        return None

# 표준 발음 재생
def play_standard_pronunciation():
    if os.path.exists(REFERENCE_AUDIO_PATH):
        print("표준 발음을 재생합니다...")
        audio = AudioSegment.from_file(REFERENCE_AUDIO_PATH)
        play(audio)
    else:
        print("표준 발음 파일이 없습니다. 생성 후 재생하세요.")


# MFCC 추출 함수
def extract_mfcc(audio_path, sr=16000):
    y, _ = librosa.load(audio_path, sr=sr)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    # MFCC의 평균 및 표준편차로 벡터화
    mfcc_mean = np.mean(mfcc, axis=1)
    mfcc_std = np.std(mfcc, axis=1)

    # MFCC를 벡터화한 결과
    mfcc_vector = np.concatenate((mfcc_mean, mfcc_std))
    
    return mfcc_vector

'''
# MFCC 비교 함수
def compare_mfcc(reference_mfcc, user_mfcc):
    similarity = 1 - cosine(reference_mfcc, user_mfcc)
    return similarity

# DTW(Dynamic Time Warping)
def compare_mfcc_with_dtw(reference_mfcc, user_mfcc):
    dist, _, _, _ = accelerated_dtw(reference_mfcc.T, user_mfcc.T, dist='euclidean')
    return dist
'''
def remove_spaces_and_punctuation(text):
    """텍스트에서 공백과 마침표, 쉼표 등 특수문자 제거"""
    # 공백 및 특수문자 제거 (마침표, 쉼표, 느낌표 등)
    cleaned_text = re.sub(r'[^\w\s]', '', text)  # 알파벳, 숫자, 공백 외의 문자 제거
    cleaned_text = cleaned_text.replace(" ", "")  # 공백 제거
    return cleaned_text

#API로 텍스트 발음교정
def transcribe_with_etri(audio_path, script=""):
    # ETRI API URL (한국어 발음 평가 API)
    url = "http://aiopen.etri.re.kr:8000/WiseASR/PronunciationKor"
    
    # 헤더 설정 (Authorization에 accessKey 포함)
    headers = {
        "Content-Type": "application/json; charset=UTF-8",
        "Authorization": API_KEY
    }
    
    # 오디오 파일을 Base64로 인코딩
    with open(audio_path, "rb") as audio_file:
        audio_contents = base64.b64encode(audio_file.read()).decode("utf-8")
    
    data = {
        "argument": {
            "language_code": "korean",
            "audio": audio_contents,
        }
    }
    # POST 요청
    response = requests.post(url, headers=headers, data=json.dumps(data))

    if response.status_code == 200:
        try:
            result = response.json()
            print(f"ETRI API Response: {result}")
            if result.get('result') == 0:
                return result['return_object']['recognized']
            else:
                print(f"ETRI API Error: {result.get('reason', 'Unknown error')}")
                return None
        except json.JSONDecodeError as e:
            print(f"JSON 디코딩 실패: {e}")
            print(f"응답 본문: {response.text}")
            return None
    else:
        print(f"HTTP 요청 실패: 상태 코드 {response.status_code}")
        print(f"응답 본문: {response.text}")
        return None
    
# 사용자 발음 평가
def evaluate_pronunciation():
    
    # 사용자 발음 녹음
    
    recognize_speech_from_mic()
    input_audio_path = "user_audio.wav"
    processed_audio_path = "user_audio_processed.wav"
    audio = AudioSegment.from_file(input_audio_path)
    audio = audio.set_frame_rate(16000).set_channels(1).set_sample_width(2)
    audio.export(processed_audio_path, format="wav")
    
    if processed_audio_path:
        # ETRI API를 통한 음성 텍스트 변환
        recognized_text = transcribe_with_etri(processed_audio_path,script="")
       
        recognized_text=remove_spaces_and_punctuation(recognized_text)
        error_rate = wer(text, recognized_text)
        if recognized_text:
            print(f"인식된 텍스트: {recognized_text}")

            # 표준 발음(text)과 비교하는 발음 평가 로직
            if recognized_text == text:
                print("발음이 정확합니다!")
                print("error rate:",error_rate)
            else:
                print(f"발음에 차이가 있습니다. 표준 발음: {text}, 사용자 발음: {recognized_text}")
                print("error rate:",error_rate)
        else:
            print("발음 교정을 위한 텍스트 변환에 실패했습니다.")
    else:
        print("사용자 음성 녹음에 실패했습니다.")
    play_standard_pronunciation()
'''
     # 표준 발음과 사용자 발음의 MFCC 비교
    ref_mfcc = extract_mfcc(REFERENCE_AUDIO_PATH)
    user_mfcc = extract_mfcc(processed_audio_path)
    similarity = compare_mfcc(ref_mfcc, user_mfcc)
    dtw_distance = compare_mfcc_with_dtw(ref_mfcc, user_mfcc)

    print(f"MFCC 벡터 간 유사도: {similarity * 100:.2f}%")
    print(f"발음 유사도 (DTW 기반 거리): {dtw_distance:.2f}")
'''    
    

# 발음 교정 실행
evaluate_pronunciation()
