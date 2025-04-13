
import re
import numpy as np
import librosa
import hgtk
from difflib import SequenceMatcher
from g2pk import G2p
from dtw import accelerated_dtw
from scipy.spatial.distance import cosine
import base64
import httpx

API_KEY = "067ea6f9-1715-43ab-814f-e23876886b9b"

class PronunciationModel:
    def __init__(self):
        self.g2p = G2p()

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

    def get_g2p(self, text):
        return self.g2p(text)

    @staticmethod
    def remove_spaces_and_punctuation(text):
        return re.sub(r'[^\w\s]', '', text).replace(" ", "")

    @staticmethod
    def compare_pronunciation(reference_pronunciation, user_pronunciation):
        ref_jamo = PronunciationModel.split_to_jamo(reference_pronunciation)
        user_jamo = PronunciationModel.split_to_jamo(user_pronunciation)
        feedback = []
        for i, (ref, user) in enumerate(zip(ref_jamo, user_jamo)):
            if ref != user:
                feedback.append(f"{i+1}번째 발음 차이: 표준 '{ref}' vs 사용 '{user}'")
        return feedback

    @staticmethod
    def split_text_to_characters(text):
        return list(text)

    @staticmethod
    def compare_text_by_characters(reference_text, user_text):
        reference_chars = PronunciationModel.split_text_to_characters(reference_text)
        user_chars = PronunciationModel.split_text_to_characters(user_text)
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
    def jamo_distance(j1, j2):
        similar_vowels = [("ㅓ", "ㅔ"), ("ㅐ", "ㅔ"), ("ㅗ", "ㅜ"), ("ㅑ", "ㅕ"), ("ㅛ", "ㅠ")]
        similar_consonants = [("ㄱ", "ㅋ"), ("ㄷ", "ㅌ"), ("ㅂ", "ㅍ"), ("ㅅ", "ㅆ"), ("ㅈ", "ㅊ")]
        if j1 == j2:
            return 0
        if (j1, j2) in similar_vowels or (j2, j1) in similar_vowels:
            return 0.5
        if (j1, j2) in similar_consonants or (j2, j1) in similar_consonants:
            return 0.5
        return 1

    @staticmethod
    def compare_jamo_dtw(reference_text, user_text):
        ref_jamo = PronunciationModel.split_to_jamo(reference_text)
        user_jamo = PronunciationModel.split_to_jamo(user_text)
        ref_jamo_np = np.array(ref_jamo).reshape(-1, 1)
        user_jamo_np = np.array(user_jamo).reshape(-1, 1)
        dist, _, _, _ = accelerated_dtw(ref_jamo_np, user_jamo_np, dist=lambda x, y: PronunciationModel.jamo_distance(x[0], y[0]))
        return dist

    @staticmethod
    def compare_syllables(reference_text, user_text):
        ref_syllables = list(reference_text)
        user_syllables = list(user_text)
        if len(ref_syllables) == len(user_syllables):
            return 0
        lcs_length = PronunciationModel.lcs(ref_syllables, user_syllables)
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
        syllable_accuracy = PronunciationModel.compare_text_by_syllables(reference_text, user_text)
        character_accuracy = PronunciationModel.compare_text_by_characters(reference_text, user_text)
        jamo_dtw = PronunciationModel.compare_jamo_dtw(reference_text, user_text)
        missing_syllables = PronunciationModel.compare_syllables(reference_text, user_text)
        lcs_score = PronunciationModel.lcs(reference_text, user_text)
        mfcc_similarity = PronunciationModel.compare_mfcc(reference_audio, user_audio)
        mfcc_dtw = PronunciationModel.compare_mfcc_with_dtw(reference_audio, user_audio)

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

    @staticmethod
    async def transcribe_with_etri(file_path, script=""):
        url = "http://aiopen.etri.re.kr:8000/WiseASR/PronunciationKor"
        headers = {
            "Content-Type": "application/json; charset=UTF-8",
            "Authorization": API_KEY
        }

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
            result = response.json()
            if result.get('result') == 0:
                return result['return_object']['recognized']
        return None
