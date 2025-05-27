
import re
import numpy as np
import librosa
import hgtk
from difflib import SequenceMatcher
from g2pk import G2p
from dtw import dtw
from scipy.spatial.distance import euclidean
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
                try:
                    # ref와 user가 자모 세 개로 분해될 수 있을 때만
                    ref_c, ref_v, ref_f = ref if len(ref) == 3 else (ref, '', '')
                    user_c, user_v, user_f = user if len(user) == 3 else (user, '', '')
                    feedback.append(f"{i+1}번째 초성 차이: '{ref_c}' vs '{user_c}'")
                    feedback.append(f"{i+1}번째 중성 차이: '{ref_v}' vs '{user_v}'")
                    feedback.append(f"{i+1}번째 종성 차이: '{ref_f}' vs '{user_f}'")
                except Exception:
                    # fallback for unexpected input
                    feedback.append(f"{i+1}번째 발음 차이: '{ref}' vs '{user}'")
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
        dist, _, _, _ = dtw(ref_jamo_np, user_jamo_np, dist=lambda x, y: PronunciationModel.jamo_distance(x[0], y[0]))
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
    def compare_mfcc_with_dtw(reference_mfcc, user_mfcc):
        from dtw import dtw

        ref_seq = reference_mfcc.T
        user_seq = user_mfcc.T

        dist, _, _, _ = dtw(
            ref_seq, user_seq,
            dist=lambda x, y: euclidean(x.ravel(), y.ravel())  # 🔥 flatten to 1D
        )
        return dist
    @staticmethod
    def get_mfcc_dtw_feedback(ref_mfcc, user_mfcc):
        from dtw import dtw

        # 길이 맞추기 (필요시 padding 또는 trimming 고려 가능)
        ref_seq = ref_mfcc.T
        user_seq = user_mfcc.T
       
        dist, _, _, _ = dtw(
            ref_seq, user_seq,
            dist=lambda x, y: euclidean(x.ravel(), y.ravel())  # 🔥 flatten to 1D
        )

        # 단순 규칙 예시: 속도나 왜곡이 심한 구간 체크
        if dist > 1000:
            return ["발음 속도가 기준보다 빠르거나 느립니다. 일정하게 발음해보세요."]
        elif dist > 500:
            return ["일부 구간이 길거나 짧습니다. 일정한 속도로 발음해보세요."]
        else:
            return ["발음 속도와 길이는 비교적 안정적입니다."]

    
    @staticmethod
    def analyze_dtw_alignment(reference_mfcc, user_mfcc):
        """DTW warping path로 발음 길이 차이 피드백"""
        from dtw import dtw
        from scipy.spatial.distance import euclidean

        ref_seq = reference_mfcc.T
        user_seq = user_mfcc.T

        dist, cost, acc_cost, path = dtw(
            ref_seq,
            user_seq,
            dist=lambda x, y: euclidean(x.ravel(), y.ravel())
        )

        ref_indices, user_indices = path
        feedback = []

        stretch_count = 0
        compress_count = 0

        for i in range(1, len(ref_indices)):
            ref_prev, user_prev = ref_indices[i - 1], user_indices[i - 1]
            ref_curr, user_curr = ref_indices[i], user_indices[i]

            ref_delta = ref_curr - ref_prev
            user_delta = user_curr - user_prev

            if user_delta > ref_delta:
                stretch_count += 1
            elif user_delta < ref_delta:
                compress_count += 1

        if stretch_count > 5:
            feedback.append("발음을 기준보다 길게 끄는 경향이 있습니다. 조금 더 간결하게 발음해보세요.")
        if compress_count > 5:
            feedback.append("발음이 기준보다 짧습니다. 천천히 또박또박 발음해보세요.")
        if not feedback:
            feedback.append("발음 속도와 길이는 비교적 안정적입니다.")

        return feedback

    @staticmethod
    def analyze_mfcc_bands(ref_mfcc, user_mfcc):
        """MFCC 평균 차이를 기반으로 특정 대역의 부정확성 피드백"""
        ref_mfcc = np.array(ref_mfcc)
        user_mfcc = np.array(user_mfcc)

        if ref_mfcc.shape != user_mfcc.shape or ref_mfcc.ndim != 1:
            return ["MFCC 피드백 분석 실패: 잘못된 입력 형태입니다."]

        diff = np.abs(ref_mfcc - user_mfcc)

        feedback = []

        low_band = np.mean(diff[:4])     # 저주파
        mid_band = np.mean(diff[4:9])    # 중주파
        high_band = np.mean(diff[9:])    # 고주파

        threshold = 12.0  # ✅ 이전보다 높은 임계값
        minor_threshold = 7.0  # ✅ 경고 수준

        if low_band > threshold:
            feedback.append("저주파 대역의 발음이 약합니다. 목소리를 깊이 내보세요.")
        elif low_band > minor_threshold:
            feedback.append("저주파 대역이 조금 약합니다. 안정감 있게 발음해보세요.")

        if mid_band > threshold:
            feedback.append("중간 대역의 발음이 불안정합니다. 천천히 또박또박 발음해보세요.")
        elif mid_band > minor_threshold:
            feedback.append("중간 대역이 다소 흔들립니다. 리듬을 유지해보세요.")

        if high_band > threshold:
            feedback.append("고주파 대역의 발음이 날카롭지 못합니다. 끝음을 또렷하게 해보세요.")
        elif high_band > minor_threshold:
            feedback.append("고주파 대역이 다소 흐릿합니다. 발음을 명확히 해보세요.")
        return feedback

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

        async with httpx.AsyncClient(timeout=30.0) as client:
            try:
                response = await client.post(url, headers=headers, json=data)
                if response.status_code == 200:
                    result = response.json()
                    if result.get('result') == 0:
                        return result['return_object']['recognized']
            except httpx.ReadTimeout:
                print("❌ [ETRI] 요청 시간이 초과되었습니다.")
            except Exception as e:
                print(f"❌ [ETRI] 오류 발생: {e}")

            return None

        if response.status_code == 200:
            result = response.json()
            if result.get('result') == 0:
                return result['return_object']['recognized']
        return None
