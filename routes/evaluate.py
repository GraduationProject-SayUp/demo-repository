# routes/evaluate.py
import pyttsx3
from fastapi import APIRouter, UploadFile, File, Form
from fastapi.responses import JSONResponse
import os
from pydub import AudioSegment
from services.pronunciation_service import (
    calculate_pronunciation_score,
    get_g2p_feedback,
)
from models.pronunciation_model import PronunciationModel

router = APIRouter()
model = PronunciationModel()

SCORE_HISTORY = {}

#STANDARD_AUDIO_PATH = "data/standard_pronunciation.wav"

@router.post("/evaluate-pronunciation")
async def evaluate_pronunciation(file: UploadFile = File(...), text: str = Form(...), user_id: str = Form(...)):
    try:
        file_path = f"temp_{file.filename}"
        with open(file_path, "wb") as f:
            f.write(await file.read())

        engine = pyttsx3.init()

        # TTS 설정 (예: 한국어)
        engine.setProperty('rate', 150)  # 속도 설정
        engine.setProperty('volume', 1)  # 볼륨 설정
        engine.setProperty('language', 'ko')

        # 표준 발음 파일 경로 설정
        REFERENCE_AUDIO_PATH = os.path.join(os.getcwd(), 'standard_pronunciation.wav')

        # TTS로 음성 파일 저장
        engine.save_to_file(text, REFERENCE_AUDIO_PATH)
        engine.runAndWait()  # 음성 파일 저장 후 대기

        processed_audio = "user_audio_processed.wav"
        audio = AudioSegment.from_file(file_path)
        audio = audio.set_frame_rate(16000).set_channels(1).set_sample_width(2)
        audio.export(processed_audio, format="wav")

        ref_mfcc = model.extract_mfcc(REFERENCE_AUDIO_PATH)
        user_mfcc = model.extract_mfcc(processed_audio)

        # from models import PronunciationModel
        recognized_text = await model.transcribe_with_etri(processed_audio)
        #recognized_text = "구지"
        cleaned_recognized_text = model.remove_spaces_and_punctuation(recognized_text)

        reference_pronunciation = model.get_g2p(text)
        g2p_feedback = model.compare_pronunciation(reference_pronunciation, cleaned_recognized_text)
        dtw_feedback = model.get_mfcc_dtw_feedback(ref_mfcc, user_mfcc)
        mfcc_feedback = model.analyze_mfcc_bands(ref_mfcc,user_mfcc)
        dtw_align_feedback = model.analyze_dtw_alignment(ref_mfcc,user_mfcc)
        score_result = model.calculate_score(reference_pronunciation, cleaned_recognized_text, ref_mfcc, user_mfcc)
        if user_id not in SCORE_HISTORY:
            SCORE_HISTORY[user_id] = []

        SCORE_HISTORY[user_id].append({
            "word": text,
            "score": score_result["total"],
            "breakdown": score_result
        })

        return JSONResponse(content={
            "message": "발음 평가 완료",
            "score": score_result["total"],
            "발음 할 것": text,
            "인식된 발음음": cleaned_recognized_text,
            "breakdown": score_result,
            "feedback": {
                "g2p": g2p_feedback,
                "mfcc": mfcc_feedback,
                "dtw_align" : dtw_align_feedback,
                "dtw_feedback" : dtw_feedback
            }
        })
    except Exception as e:
        import traceback
        print("❌ 오류 발생:")
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"message": f"서버 내부 오류: {str(e)}"})
