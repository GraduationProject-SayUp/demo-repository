import os
import whisper
import pyttsx3
import speech_recognition as sr
import librosa
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
import platform
from fastdtw import fastdtw
from difflib import SequenceMatcher


# 한글 폰트 설정
if platform.system() == "Windows":
    plt.rc("font", family="Malgun Gothic")  # Windows
elif platform.system() == "Darwin":
    plt.rc("font", family="AppleGothic")  # Mac
else:
    plt.rc("font", family="NanumGothic")  # Linux (Ubuntu 등)

# 마이너 경고 제거
plt.rcParams["axes.unicode_minus"] = False

# Whisper 모델 로드
model = whisper.load_model("small")

# TTS 엔진 초기화
engine = pyttsx3.init()

# TTS 설정 (예: 한국어)
engine.setProperty('language', 'ko')

# 표준 발음으로 변환할 텍스트
text = "안녕하세요, 반갑습니다."

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

def extract_mfcc(audio_path, sr=16000, n_mfcc=13):
    """MFCC 특징 추출 함수"""
    y, sr = librosa.load(audio_path, sr=sr)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    return mfcc

def calculate_similarity(mfcc1, mfcc2):
    """DTW를 사용해 두 MFCC 간 유사도 계산"""
    distance, _ = fastdtw(mfcc1.T, mfcc2.T, dist=lambda x, y: np.linalg.norm(x - y, ord=1))
    return distance

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

def evaluate_pronunciation(user_audio_path):
    """사용자 발음 평가"""
    print("표준 발음과 사용자 발음 비교를 시작합니다...")
    if not user_audio_path.endswith('.wav'):
        print(f"지원되지 않는 파일 형식입니다: {user_audio_path}")
        return None, None
    try:
        # Whisper를 사용한 텍스트 변환
        print("Whisper를 사용하여 음성 텍스트 변환 중...")
        whisper_result = model.transcribe("user_audio.wav", language="ko")
        detected_text = whisper_result["text"]
        print(f"Whisper 인식 텍스트: {detected_text}")

        # 기준 텍스트와 유사도 비교
        text_similarity = SequenceMatcher(None, text, detected_text).ratio()
        print(f"텍스트 유사도: {text_similarity * 100:.2f}%")

        # 표준 발음 MFCC 추출
        ref_mfcc = extract_mfcc(REFERENCE_AUDIO_PATH)
        # 사용자 발음 MFCC 추출
        user_mfcc = extract_mfcc(user_audio_path)

        # DTW를 사용한 음성 신호 유사도 계산
        distance = calculate_similarity(ref_mfcc, user_mfcc)
        print(f"발음 유사도 거리 (DTW): {distance}")

        # 발음 정확도 평가
        pronunciation_accuracy = max(0, 100 - distance * 10)  # 거리 기반으로 정확도 계산 (예: 100점 만점)
        print(f"발음 정확도: {pronunciation_accuracy:.2f}%")

        # MFCC 시각화
        plt.figure(figsize=(10, 4))
        plt.subplot(2, 1, 1)
        librosa.display.specshow(ref_mfcc, x_axis="time")
        plt.colorbar()
        plt.title("표준 발음 MFCC")

        plt.subplot(2, 1, 2)
        librosa.display.specshow(user_mfcc, x_axis="time")
        plt.colorbar()
        plt.title("사용자 발음 MFCC")

        plt.tight_layout()
        plt.show()

        return text_similarity, pronunciation_accuracy

    except Exception as e:
        print(f"발음 평가 중 오류 발생: {e}")
        return None, None

def main():
    while True:
        print("\n--- 실시간 발음 평가 ---")
        # 사용자 음성 녹음
        user_audio_path = recognize_speech_from_mic()

        if user_audio_path:
            # 발음 평가
            text_similarity, pronunciation_accuracy = evaluate_pronunciation(user_audio_path)

            if text_similarity is not None:
                print(f"\n--- 평가 결과 ---")
                print(f"텍스트 유사도: {text_similarity * 100:.2f}%")
                print(f"발음 정확도: {pronunciation_accuracy:.2f}%")
            else:
                print("발음 평가를 할 수 없습니다.")
        else:
            print("음성을 다시 시도해주세요.")

if __name__ == "__main__":
    main()
