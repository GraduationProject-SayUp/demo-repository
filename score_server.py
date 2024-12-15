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
<<<<<<< HEAD
from fastapi import FastAPI, File, UploadFile, Form
=======
from fastapi import FastAPI, File, UploadFile
>>>>>>> 324dab53e4b2d158a509b5c5d616c780a248b4c0
from fastapi.responses import JSONResponse
import asyncio
from pydantic import BaseModel
import httpx  # 비동기 HTTP 요청을 위해 httpx 사용

app = FastAPI()

# ETRI API 키 (공백 삭제 후 사용)
API_KEY = "067ea6f9-1715-43ab-814f-e23876886b9b"

engine = pyttsx3.init()

# TTS 설정 (예: 한국어)
engine.setProperty('rate', 150)  # 속도 설정
engine.setProperty('volume', 1)  # 볼륨 설정
engine.setProperty('language', 'ko')

<<<<<<< HEAD
# 표준 발음 파일 경로 설정
REFERENCE_AUDIO_PATH = os.path.join(os.getcwd(), 'standard_pronunciation.wav')

=======
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

>>>>>>> 324dab53e4b2d158a509b5c5d616c780a248b4c0
# 음성 인식 설정
recognizer = sr.Recognizer()
microphone = sr.Microphone()

class Model(BaseModel):
    file_path: str
<<<<<<< HEAD
    text: str  # text를 프론트에서 입력받을 수 있도록 수정

=======
    
>>>>>>> 324dab53e4b2d158a509b5c5d616c780a248b4c0
    @staticmethod
    def remove_spaces_and_punctuation(text):
        """텍스트에서 공백과 마침표, 쉼표 등 특수문자 제거"""
        cleaned_text = re.sub(r'[^\w\s]', '', text)  # 알파벳, 숫자, 공백 외의 문자 제거
        cleaned_text = cleaned_text.replace(" ", "")  # 공백 제거
        return cleaned_text

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
<<<<<<< HEAD
async def evaluate_pronunciation(file: UploadFile = File(...), text: str = Form(...)):
=======
async def evaluate_pronunciation(file: UploadFile = File(...)):
>>>>>>> 324dab53e4b2d158a509b5c5d616c780a248b4c0
    # 업로드된 파일을 서버에 임시 저장
    try:
        file_path = f"temp_{file.filename}"
        with open(file_path, "wb") as f:
            f.write(await file.read())
    except Exception as e:
        return JSONResponse(status_code=500, content={"message": f"파일 저장에 실패했습니다. {str(e)}"})
    
    # 오디오 파일 처리 (샘플링 레이트 16000Hz로 설정)
    try:
        processed_audio_path = "user_audio_processed.wav"
        audio = AudioSegment.from_file(file_path)
        audio = audio.set_frame_rate(16000).set_channels(1).set_sample_width(2)
        audio.export(processed_audio_path, format="wav")
    except Exception as e:
        return JSONResponse(status_code=500, content={"message": f"오디오 파일 처리에 실패했습니다. {str(e)}"})

    if processed_audio_path:
        # ETRI API를 통한 음성 텍스트 변환
        recognized_text = await Model.transcribe_with_etri(processed_audio_path)
        
        if recognized_text:
            cleaned_recognized_text = Model.remove_spaces_and_punctuation(recognized_text)
            error_rate = wer(text, cleaned_recognized_text)
            print(f"인식된 텍스트: {cleaned_recognized_text}")

            if cleaned_recognized_text == text:
                # 발음이 정확한 경우
                return JSONResponse(content={"message": f"발음이 정확합니다. error rate: {error_rate}"})
            else:
                # 발음에 차이가 있는 경우
                return JSONResponse(content={"message": f"발음에 차이가 있습니다. 표준 발음: {text}, 사용자 발음: {cleaned_recognized_text}, error rate: {error_rate}"})
        else:
            return JSONResponse(content={"message": "발음 교정을 위한 텍스트 변환에 실패했습니다."})
    else:
        print("")

    os.remove(file_path)  # 처리 후 임시 파일 삭제
    os.remove(processed_audio_path)  # 처리 후 임시 파일 삭제

    return {"message": "발음 평가 완료!"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
