# routes/evaluate.py

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

STANDARD_AUDIO_PATH = "data/standard_pronunciation.wav"

@router.post("/evaluate-pronunciation")
async def evaluate_pronunciation(file: UploadFile = File(...), text: str = Form(...), user_id: str = Form(...)):
    try:
        file_path = f"temp_{file.filename}"
        with open(file_path, "wb") as f:
            f.write(await file.read())


        processed_audio = "user_audio_processed.wav"
        audio = AudioSegment.from_file(file_path)
        audio = audio.set_frame_rate(16000).set_channels(1).set_sample_width(2)
        audio.export(processed_audio, format="wav")

        ref_mfcc = model.extract_mfcc(STANDARD_AUDIO_PATH)
        user_mfcc = model.extract_mfcc(processed_audio)

        # from models import PronunciationModel
        recognized_text = await model.transcribe_with_etri(processed_audio)
        #recognized_text = "구지"
        cleaned_recognized_text = model.remove_spaces_and_punctuation(recognized_text)

        reference_pronunciation = model.get_g2p(text)
        g2p_feedback = model.compare_pronunciation(reference_pronunciation, cleaned_recognized_text)
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
            "feedback": g2p_feedback
        })
    except Exception as e:
        import traceback
        print("❌ 오류 발생:")
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"message": f"서버 내부 오류: {str(e)}"})
