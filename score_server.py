
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
from fastapi import FastAPI, File, UploadFile, Form, Query
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional
import asyncio
from pydantic import BaseModel
import httpx
import hgtk
from difflib import SequenceMatcher
from g2pk import G2p
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
import pandas as pd
import matplotlib.pyplot as plt
from io import BytesIO
from datetime import datetime

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

API_KEY = "067ea6f9-1715-43ab-814f-e23876886b9b"
REFERENCE_AUDIO_PATH = os.path.join(os.getcwd(), 'standard_pronunciation.wav')

recognizer = sr.Recognizer()
microphone = sr.Microphone()
g2p = G2p()

score_history = {}

class Model(BaseModel):
    @staticmethod
    def extract_mfcc(audio_path, sr=16000):
        y, _ = librosa.load(audio_path, sr=sr)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        mfcc_mean = np.mean(mfcc, axis=1)
        mfcc_std = np.std(mfcc, axis=1)
        return np.concatenate((mfcc_mean, mfcc_std))

    @staticmethod
    def compare_mfcc(reference_mfcc, user_mfcc):
        return 1 - cosine(reference_mfcc, user_mfcc)

    @staticmethod
    def compare_mfcc_with_dtw(reference_mfcc, user_mfcc):
        dist, _, _, _ = accelerated_dtw(reference_mfcc.T, user_mfcc.T, dist='euclidean')
        return dist

    @staticmethod
    def get_g2p_pronunciation(text):
        return g2p(text)

    @staticmethod
    def compare_pronunciation(reference_pronunciation, user_pronunciation):
        ref_jamo = Model.split_to_jamo(reference_pronunciation)
        user_jamo = Model.split_to_jamo(user_pronunciation)
        feedback = []
        for i, (ref, user) in enumerate(zip(ref_jamo, user_jamo)):
            if ref != user:
                feedback.append(f"{i+1}번째 발음 차이: 표준 '{ref}' vs 사용 '{user}'")
        return feedback

    @staticmethod
    def remove_spaces_and_punctuation(text):
        return re.sub(r'[^\w\s]', '', text).replace(" ", "")

    @staticmethod
    def split_text_to_characters(text):
        return list(text)

    @staticmethod
    def compare_text_by_characters(reference_text, user_text):
        reference_chars = Model.split_text_to_characters(reference_text)
        user_chars = Model.split_text_to_characters(user_text)
        match_count = sum(1 for r, u in zip(reference_chars, user_chars) if r == u)
        total_chars = max(len(reference_chars), len(user_chars))
        return match_count / total_chars * 100

    @staticmethod
    def compare_text_by_syllables(reference_text, user_text):
        matcher = SequenceMatcher(None, reference_text, user_text)
        return matcher.ratio() * 100

    @staticmethod
    def split_to_jamo(text):
        jamo_list = []
        for char in text:
            if hgtk.checker.is_hangul(char):
                jamo_list.extend(hgtk.letter.decompose(char))
            else:
                jamo_list.append(char)
        return jamo_list

    @staticmethod
    def jamo_distance(jamo1, jamo2):
        similar_vowels = [("ㅓ", "ㅔ"), ("ㅐ", "ㅔ"), ("ㅗ", "ㅜ"), ("ㅑ", "ㅕ"), ("ㅛ", "ㅠ")]
        similar_consonants = [("ㄱ", "ㅋ"), ("ㄷ", "ㅌ"), ("ㅂ", "ㅍ"), ("ㅅ", "ㅆ"), ("ㅈ", "ㅊ")]
        if jamo1 == jamo2:
            return 0
        if (jamo1, jamo2) in similar_vowels or (jamo2, jamo1) in similar_vowels:
            return 0.5
        if (jamo1, jamo2) in similar_consonants or (jamo2, jamo1) in similar_consonants:
            return 0.5
        return 1

    @staticmethod
    def compare_jamo_dtw(reference_text, user_text):
        ref_jamo = Model.split_to_jamo(reference_text)
        user_jamo = Model.split_to_jamo(user_text)
        ref_jamo_np = np.array(ref_jamo).reshape(-1, 1)
        user_jamo_np = np.array(user_jamo).reshape(-1, 1)
        dist, _, _, _ = accelerated_dtw(ref_jamo_np, user_jamo_np, dist=lambda x, y: Model.jamo_distance(x[0], y[0]))
        return dist

    @staticmethod
    def compare_syllables(reference_text, user_text):
        ref_syllables = list(reference_text)
        user_syllables = list(user_text)
        if len(ref_syllables) == len(user_syllables):
            return 0
        lcs_length = Model.lcs(ref_syllables, user_syllables)
        return len(ref_syllables) - lcs_length

    @staticmethod
    def lcs(X, Y):
        m, n = len(X), len(Y)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if X[i - 1] == Y[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1] + 1
                else:
                    dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
        return dp[m][n]

    @staticmethod
    def calculate_score(reference_text, user_text, reference_audio, user_audio):
        syllable_accuracy = Model.compare_text_by_syllables(reference_text, user_text)
        character_accuracy = Model.compare_text_by_characters(reference_text, user_text)
        jamo_dtw = Model.compare_jamo_dtw(reference_text, user_text)
        missing_syllables = Model.compare_syllables(reference_text, user_text)
        lcs_score = Model.lcs(reference_text, user_text)
        mfcc_similarity = Model.compare_mfcc(reference_audio, user_audio)
        mfcc_dtw = Model.compare_mfcc_with_dtw(reference_audio, user_audio)

        syllable_score = syllable_accuracy
        character_score = character_accuracy
        jamo_score = 100 - jamo_dtw
        missing_score = max(0, 100 - (missing_syllables * 10))
        lcs_score = (lcs_score / len(reference_text)) * 100
        mfcc_similarity_score = mfcc_similarity * 100
        mfcc_dtw_score = max(0, 100 - (mfcc_dtw / 10))
        mfcc_score = (mfcc_similarity_score * 0.5) + (mfcc_dtw_score * 0.5)

        total_score = (syllable_score * 0.1 +
                       character_score * 0.1 +
                       jamo_score * 0.25 +
                       missing_score * 0.05 +
                       lcs_score * 0.25 +
                       mfcc_score * 0.25)

        return {
            "total": total_score,
            "syllable_score": syllable_score,
            "character_score": character_score,
            "jamo_score": jamo_score,
            "missing_score": missing_score,
            "lcs_score": lcs_score,
            "mfcc_score": mfcc_score,
        }

@app.post("/evaluate-pronunciation")
async def evaluate_pronunciation(file: UploadFile = File(...), text: str = Form(...), user_id: str = Form(...)):
    try:
        file_path = os.path.join(os.getcwd(), f"temp_{file.filename}")
        with open(file_path, "wb") as f:
            f.write(await file.read())
        processed_audio_path = os.path.join(os.getcwd(), "user_audio_processed.wav")
        audio = AudioSegment.from_file(file_path)
        audio = audio.set_frame_rate(16000).set_channels(1).set_sample_width(2)
        audio.export(processed_audio_path, format="wav")
    except Exception as e:
        return JSONResponse(status_code=500, content={"message": f"파일 처리 실패: {str(e)}"})

    ref_mfcc = Model.extract_mfcc(REFERENCE_AUDIO_PATH)
    user_mfcc = Model.extract_mfcc(processed_audio_path)

    recognized_text = "구지"
    cleaned_recognized_text = Model.remove_spaces_and_punctuation(recognized_text)
    text = Model.remove_spaces_and_punctuation(text)

    reference_pronunciation = Model.get_g2p_pronunciation(text)
    g2p_feedback = Model.compare_pronunciation(reference_pronunciation, cleaned_recognized_text)
    score_result = Model.calculate_score(reference_pronunciation, cleaned_recognized_text, ref_mfcc, user_mfcc)

    if user_id not in score_history:
        score_history[user_id] = []
    score_history[user_id].append({
        "word": text,
        "score": score_result["total"],
        "date": datetime.now().isoformat(),
        "breakdown": score_result
    })

    response_content = {
        "message": "발음 평가 결과입니다.",
        "score": score_result["total"],
        "breakdown": score_result,
        "feedback": {
            "발음 피드백": g2p_feedback
        }
    }
    return JSONResponse(content=response_content)

@app.get("/score-history")
async def get_score_history(user_id: str = Query(...), word: Optional[str] = None):
    user_data = score_history.get(user_id, [])
    if word:
        user_data = [entry for entry in user_data if entry["word"] == word]
    return {"history": user_data}

@app.get("/score-plot")
async def plot_score_history(user_id: str, word: Optional[str] = None):
    user_data = score_history.get(user_id, [])
    if word:
        user_data = [entry for entry in user_data if entry["word"] == word]
    if not user_data:
        return JSONResponse(content={"message": "해당 데이터가 없습니다."}, status_code=404)

    df = pd.DataFrame(user_data)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")

    plt.figure(figsize=(8, 4))
    plt.plot(df["date"], df["score"], marker='o')
    plt.title(f"'{word}' 발음 점수 추이")
    plt.xlabel("날짜")
    plt.ylabel("점수")
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.tight_layout()

    buf = BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    return StreamingResponse(buf, media_type="image/png")

@app.get("/score-comparison")
async def compare_score(user_id: str, word: str):
    user_scores = []
    all_scores = []

    for uid, records in score_history.items():
        for record in records:
            if record["word"] == word:
                all_scores.append(record["score"])
                if uid == user_id:
                    user_scores.append(record["score"])

    if not all_scores:
        return JSONResponse(content={"message": "데이터가 없습니다."}, status_code=404)

    average_score = sum(all_scores) / len(all_scores)
    return {
        "word": word,
        "average_score": average_score,
        "your_scores": user_scores,
        "difference": user_scores[-1] - average_score if user_scores else None
    }



@app.get("/compare-plot")
async def compare_plot(user_id: str, word: str):
    user_scores = []
    all_scores = []

    for uid, records in score_history.items():
        for record in records:
            if record["word"] == word:
                all_scores.append(record["score"])
                if uid == user_id:
                    user_scores.append(record["score"])

    if not all_scores or not user_scores:
        return JSONResponse(content={"message": "해당 단어에 대한 데이터가 없습니다."}, status_code=404)

    average_score = sum(all_scores) / len(all_scores)
    your_latest_score = user_scores[-1]

    # 그래프 생성
    labels = ['전체 평균', '내 점수']
    scores = [average_score, your_latest_score]

    plt.figure(figsize=(6, 4))
    bars = plt.bar(labels, scores)
    plt.title(f"'{word}' 단어 평균 점수 비교")
    plt.ylabel("점수 (100점 만점)")
    plt.ylim(0, 100)
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, height + 1, f"{height:.1f}", ha='center')
    plt.tight_layout()

    buf = BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    return StreamingResponse(buf, media_type="image/png")



from fpdf import FPDF
from fastapi.responses import FileResponse
import os
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
from io import BytesIO

@app.get("/generate-report")
async def generate_report(user_id: str, start_date: str, end_date: str, word: str):
    user_data = score_history.get(user_id, [])

    # 날짜 필터링
    start_dt = datetime.fromisoformat(start_date)
    end_dt = datetime.fromisoformat(end_date)
    filtered = [
        entry for entry in user_data
        if entry["word"] == word and start_dt <= datetime.fromisoformat(entry["date"]) <= end_dt
    ]
    if not filtered:
        return JSONResponse(content={"message": "해당 기간에 데이터가 없습니다."}, status_code=404)

    df = pd.DataFrame(filtered)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")

    # 점수 요약
    avg_score = df["score"].mean()
    max_score = df["score"].max()
    min_score = df["score"].min()

    # 그래프 생성
    plt.figure(figsize=(6, 3))
    plt.plot(df["date"], df["score"], marker='o')
    plt.title(f"'{word}' 발음 점수 추이")
    plt.xlabel("날짜")
    plt.ylabel("점수")
    plt.ylim(0, 100)
    plt.grid(True)
    plt.tight_layout()
    graph_buf = BytesIO()
    plt.savefig(graph_buf, format='png')
    graph_buf.seek(0)
    graph_path = f"/tmp/{user_id}_{word}_plot.png"
    with open(graph_path, "wb") as f:
        f.write(graph_buf.read())

    # PDF 생성
    class ReportPDF(FPDF):
        def header(self):
            self.add_font('NanumCoding', '', 'NanumGothicCoding.ttf', uni=True)
            self.set_font('NanumCoding', '', 16)
            self.cell(0, 10, "발음 학습 리포트", ln=True, align='C')

        def add_content(self):
            self.set_font('NanumCoding', '', 12)
            self.cell(0, 10, f"사용자: {user_id}", ln=True)
            self.cell(0, 10, f"기간: {start_date} ~ {end_date}", ln=True)
            self.cell(0, 10, f"단어: {word}", ln=True)
            self.cell(0, 10, f"평균 점수: {avg_score:.1f}", ln=True)
            self.cell(0, 10, f"최고 점수: {max_score:.1f}", ln=True)
            self.cell(0, 10, f"최저 점수: {min_score:.1f}", ln=True)

        def insert_graph(self, image_path):
            self.image(image_path, w=180)

    pdf = ReportPDF()
    pdf.add_page()
    pdf.add_content()
    pdf.insert_graph(graph_path)

    output_path = f"/mnt/data/{user_id}_{word}_report.pdf"
    pdf.output(output_path)

    return FileResponse(output_path, media_type="application/pdf", filename=os.path.basename(output_path))
