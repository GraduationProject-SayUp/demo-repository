import pyttsx3
import speech_recognition as sr
import os
import requests
import json
import base64
from pydub import AudioSegment
import librosa
import numpy as np
from scipy.spatial.distance import cosine
from dtw import accelerated_dtw
from jiwer import wer
import re
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
import asyncio
from pydantic import BaseModel
import httpx  # 비동기 HTTP 요청을 위해 httpx 사용
import hgtk  # 한글 초성, 중성, 종성 분해용
from difflib import SequenceMatcher
from g2pk import G2p

app = FastAPI()

# ETRI API 키 (공백 삭제 후 사용)
API_KEY = "067ea6f9-1715-43ab-814f-e23876886b9b"

engine = pyttsx3.init()

# TTS 설정 (예: 한국어)
engine.setProperty('rate', 150)  # 속도 설정
engine.setProperty('volume', 1)  # 볼륨 설정
engine.setProperty('language', 'ko')

# 표준 발음 파일 경로 설정
REFERENCE_AUDIO_PATH = os.path.join(os.getcwd(), 'standard_pronunciation.wav')

# 음성 인식 설정
recognizer = sr.Recognizer()
microphone = sr.Microphone()
g2p = G2p()

class Model(BaseModel):
    file_path: str
    text: str  # text를 프론트에서 입력받을 수 있도록 수정

    @staticmethod
    def get_g2p_pronunciation(text):
        """주어진 텍스트의 표준 발음을 G2P 변환하여 반환"""
        return g2p(text)  # "안녕하세요" → "안녕하세오" (연음, 두음법칙 적용 가능)

    @staticmethod
    def compare_pronunciation(reference_pronunciation, user_pronunciation):
        """G2P 변환된 발음과 사용자의 발음을 비교"""
        ref_jamo = Model.split_to_jamo(reference_pronunciation)
        user_jamo = Model.split_to_jamo(user_pronunciation)

        feedback = []
        for i, (ref, user) in enumerate(zip(ref_jamo, user_jamo)):
            if ref != user:
                feedback.append(f"{i+1}번째 발음 차이: 표준 '{ref}' vs 사용 '{user}'")

        return feedback
    @staticmethod
    def remove_spaces_and_punctuation(text):
        """텍스트에서 공백과 마침표, 쉼표 등 특수문자 제거"""
        cleaned_text = re.sub(r'[^\w\s]', '', text)  # 알파벳, 숫자, 공백 외의 문자 제거
        cleaned_text = cleaned_text.replace(" ", "")  # 공백 제거
        return cleaned_text
    
    @staticmethod
    def split_text_to_characters(text):
        """텍스트를 한 글자 단위로 분할"""
        return list(text)
    
    @staticmethod
    def compare_text_by_characters(reference_text, user_text):
        """한 글자 단위로 비교"""
        reference_chars = Model.split_text_to_characters(reference_text)
        user_chars = Model.split_text_to_characters(user_text)
        
        # 글자 단위 비교
        match_count = sum(1 for r, u in zip(reference_chars, user_chars) if r == u)
        total_chars = max(len(reference_chars), len(user_chars))
        accuracy = match_count / total_chars * 100
        return accuracy
    
    @staticmethod
    def compare_text_by_syllables(reference_text, user_text):
        """음절 단위 비교"""
        matcher = SequenceMatcher(None, reference_text, user_text)
        ratio = matcher.ratio() * 100  # 유사도 비율
        return ratio
    
    @staticmethod
    def split_to_jamo(text):
        """한글 텍스트를 초성, 중성, 종성으로 분해"""
        jamo_list = []
        for char in text:
            if hgtk.checker.is_hangul(char):  # 한글인지 확인
                jamo_list.extend(hgtk.letter.decompose(char))
            else:
                jamo_list.append(char)  # 한글이 아니면 그대로 추가
        return jamo_list
    
    @staticmethod
    def jamo_distance(jamo1, jamo2):
        """자음/모음 간의 거리: 같은 경우 0, 비슷한 경우 0.5, 다른 경우 1"""
        similar_vowels = [
            ("ㅓ", "ㅔ"), ("ㅐ", "ㅔ"), ("ㅗ", "ㅜ"), ("ㅑ", "ㅕ"), ("ㅛ", "ㅠ")
        ]
        similar_consonants = [
            ("ㄱ", "ㅋ"), ("ㄷ", "ㅌ"), ("ㅂ", "ㅍ"), ("ㅅ", "ㅆ"), ("ㅈ", "ㅊ")
        ]
        if jamo1 == jamo2:
            return 0
        if (jamo1, jamo2) in similar_vowels or (jamo2, jamo1) in similar_vowels:
            return 0.5
        if (jamo1, jamo2) in similar_consonants or (jamo2, jamo1) in similar_consonants:
            return 0.5
        return 1
    
    @staticmethod
    def compare_jamo_dtw(reference_text, user_text):
        """DTW를 사용해 자음/모음 비교"""
        ref_jamo = Model.split_to_jamo(reference_text)
        user_jamo = Model.split_to_jamo(user_text)

        ref_jamo_np = np.array(ref_jamo).reshape(-1, 1)
        user_jamo_np = np.array(user_jamo).reshape(-1, 1)

        dist, _, _, _ = accelerated_dtw(ref_jamo_np, user_jamo_np, dist=lambda x, y: Model.jamo_distance(x[0], y[0]))
        return dist
    
    @staticmethod
    def compare_syllables(reference_text, user_text):
        """음절 단위로 비교하여 누락된 부분 확인"""
        ref_syllables = list(reference_text)
        user_syllables = list(user_text)


        # 길이가 같으면 누락 없음 처리
        if len(ref_syllables) == len(user_syllables):
            return 0
        
        # LCS로 최대 공통 부분 수열 계산
        lcs_length = Model.lcs(ref_syllables, user_syllables)
        
        # 누락된 음절 수 = 원래 음절 수 - LCS 길이
        missing_syllables = len(ref_syllables) - lcs_length
        
        return missing_syllables
    @staticmethod
    def lcs(X, Y):
        """LCS (최대 공통 부분 수열)을 구하는 함수"""
        m = len(X)
        n = len(Y)
        dp = [[0] * (n + 1) for _ in range(m + 1)]

        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if X[i - 1] == Y[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1] + 1
                else:
                    dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
        
        return dp[m][n]
    @staticmethod
    def compare_jamo_feedback(reference_text, user_text):
        """초성, 중성, 종성을 비교하여 잘못 발음된 음절 피드백 제공"""
        ref_jamo = Model.split_to_jamo(reference_text)
        user_jamo = Model.split_to_jamo(user_text)
        
        feedback = []
        for i, (ref, user) in enumerate(zip(ref_jamo, user_jamo)):
            if ref != user:
                # 유사 발음 체크
                similar_vowels = [("ㅓ", "ㅔ"), ("ㅐ", "ㅔ"), ("ㅗ", "ㅜ"), ("ㅑ", "ㅕ"), ("ㅛ", "ㅠ")]
                similar_consonants = [("ㄱ", "ㅋ"), ("ㄷ", "ㅌ"), ("ㅂ", "ㅍ"), ("ㅅ", "ㅆ"), ("ㅈ", "ㅊ")]

                if (ref, user) in similar_vowels or (user, ref) in similar_vowels:
                    feedback.append(f"{i+1}번째 음절: '{ref}' 발음을 '{user}'로 잘못 발음 (유사 모음)")
                elif (ref, user) in similar_consonants or (user, ref) in similar_consonants:
                    feedback.append(f"{i+1}번째 음절: '{ref}' 발음을 '{user}'로 잘못 발음 (유사 자음)")
                else:
                    feedback.append(f"{i+1}번째 음절: '{ref}' 발음을 '{user}'로 잘못 발음")
        return feedback

    @staticmethod
    def calculate_score(reference_text, user_text):
        """전체 점수를 계산"""
        # 각 방법에 대한 점수 계산
        syllable_accuracy = Model.compare_text_by_syllables(reference_text, user_text)
        character_accuracy = Model.compare_text_by_characters(reference_text, user_text)
        jamo_dtw = Model.compare_jamo_dtw(reference_text, user_text)
        missing_syllables = Model.compare_syllables(reference_text, user_text)
        lcs_score = Model.lcs(reference_text, user_text)
       
        # 가중치 설정
        syllable_weight = 0.1
        character_weight = 0.1
        jamo_weight = 0.45
        missing_weight = 0.05
        lcs_weight = 0.30

        # 점수를 100점 만점으로 정규화하여 계산
        syllable_score = syllable_accuracy  # 음절 정확도는 이미 100점 만점으로 계산됨
        character_score = character_accuracy  # 문자 정확도도 마찬가지
        jamo_score = 100 - jamo_dtw  # DTW의 결과는 낮을수록 정확도 높으므로 100에서 빼기
        missing_score = max(0, 100 - (missing_syllables * 10))  # 음절 누락은 누락된 수에 비례하여 점수 차감 (누락이 많을수록 점수 낮아짐)
        lcs_score = (lcs_score / len(reference_text)) * 100  # LCS 길이를 기준으로 비율로 계산
        # 최종 점수 계산
        total_score = (syllable_score * syllable_weight) + \
                    (character_score * character_weight) + \
                    (jamo_score * jamo_weight) + \
                    (missing_score * missing_weight) + \
                    (lcs_score * lcs_weight)
        return total_score
    # API로 텍스트 발음교정
    @staticmethod
    async def transcribe_with_etri(file_path, script=""):
        # ETRI API URL (한국어 발음 평가 API)
        url = "http://aiopen.etri.re.kr:8000/WiseASR/PronunciationKor"
        
        # 헤더 설정 (Authorization에 accessKey 포함)
        headers = {
            "Content-Type": "application/json; charset=UTF-8",
            "Authorization": API_KEY
        }
        
        # 오디오 파일을 Base64로 인코딩
        with open(file_path, "rb") as file:
            audio_contents = base64.b64encode(file.read()).decode("utf-8")
        
        data = {
            "argument": {
                "language_code": "korean",
                "audio": audio_contents,
            }
        }
        
        async with httpx.AsyncClient() as client:
            response = await client.post(url, headers=headers, json=data)

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

@app.post("/evaluate-pronunciation")
async def evaluate_pronunciation(file: UploadFile = File(...), text: str = Form(...)):
    try:
        # 업로드된 파일 저장
        file_path = os.path.join(os.getcwd(), f"temp_{file.filename}")
        with open(file_path, "wb") as f:
            f.write(await file.read())
        processed_audio_path = os.path.join(os.getcwd(), "user_audio_processed.wav")

        # 오디오 파일 처리 (샘플링 레이트 16000Hz로 설정)
        audio = AudioSegment.from_file(file_path)
        audio = audio.set_frame_rate(16000).set_channels(1).set_sample_width(2)
        audio.export(processed_audio_path, format="wav")
    except Exception as e:
        return JSONResponse(status_code=500, content={"message": f"파일 저장 또는 처리 실패: {str(e)}"})

    # ETRI API로 음성 텍스트 변환
    recognized_text = await Model.transcribe_with_etri(processed_audio_path)

    if recognized_text:
        cleaned_recognized_text = Model.remove_spaces_and_punctuation(recognized_text)
        text = Model.remove_spaces_and_punctuation(text)

        # **G2P 변환 적용**
        reference_pronunciation = Model.get_g2p_pronunciation(text)
        user_pronunciation = Model.get_g2p_pronunciation(cleaned_recognized_text)

        g2p_feedback = Model.compare_pronunciation(reference_pronunciation, user_pronunciation)

        total_score = Model.calculate_score(g2p(text), g2p(cleaned_recognized_text))

        # 피드백 생성
        missing_syllables = Model.compare_syllables(text, cleaned_recognized_text)
        missing_feedback = (
            f"누락된 음절: {missing_syllables}개" if missing_syllables > 0 else "음절 누락이 없습니다."
        )

        # 모음/자음 오류 피드백
        jamo_feedback = Model.compare_jamo_feedback(text, cleaned_recognized_text)
        jamo_feedback_str = "\n".join(jamo_feedback) if jamo_feedback else "모음/자음 발음 오류가 없습니다."

        print(jamo_feedback)
        feedback_message = {
            "original_text": text,
            "recognized_text": cleaned_recognized_text,
            "missing_feedback": missing_feedback,
            "score": total_score,
            "g2p_feedback": g2p_feedback,
            #"jamo_feedback": jamo_feedback_str,
           
        }

        # 점수와 피드백 반환
        response_content = {
            "message": "발음 평가 결과입니다.",
            "score": total_score,
            "feedback": feedback_message
        }

        return JSONResponse(content=response_content)
    else:
        os.remove(file_path)
        os.remove(processed_audio_path)
        return JSONResponse(content={"message": "발음 교정을 위한 텍스트 변환에 실패했습니다."})    

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
