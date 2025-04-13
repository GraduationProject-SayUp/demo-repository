# routes/history.py

from fastapi import APIRouter, Query
from fastapi.responses import JSONResponse, StreamingResponse
import matplotlib.pyplot as plt
import pandas as pd
from io import BytesIO
from routes.evaluate import SCORE_HISTORY

router = APIRouter()

@router.get("/score-history")
async def get_score_history(user_id: str = Query(...), word: str = None):
    history = SCORE_HISTORY.get(user_id, [])
    if word:
        history = [h for h in history if h["word"] == word]
    return {"history": history}

@router.get("/score-plot")
async def plot_score_history(user_id: str, word: str):
    history = SCORE_HISTORY.get(user_id, [])
    word_scores = [h for h in history if h["word"] == word]
    if not word_scores:
        return JSONResponse(content={"message": "해당 단어에 대한 기록이 없습니다."}, status_code=404)

    df = pd.DataFrame(word_scores)
    df["date"] = pd.date_range(end=pd.Timestamp.now(), periods=len(df))
    df = df.sort_values("date")

    plt.figure(figsize=(6, 3))
    plt.plot(df["date"], df["score"], marker='o')
    plt.title(f"{word} 점수 추이")
    plt.xlabel("날짜")
    plt.ylabel("점수")
    plt.ylim(0, 100)
    plt.tight_layout()

    buf = BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    return StreamingResponse(buf, media_type="image/png")
