# routes/history.py

from fastapi import APIRouter, Query
from fastapi.responses import JSONResponse, StreamingResponse
import matplotlib.pyplot as plt
import pandas as pd
from io import BytesIO
from routes.evaluate import SCORE_HISTORY
import matplotlib.font_manager as fm
import matplotlib as mpl

# ✅ Windows 기본 한글 글꼴 사용 (맑은 고딕)
font_path = "C:/Windows/Fonts/malgun.ttf"  # 경로는 OS에 따라 다를 수 있음
fontprop = fm.FontProperties(fname=font_path)
plt.rcParams['font.family'] = fontprop.get_name()
mpl.rcParams['axes.unicode_minus'] = False  # 마이너스 깨짐 방지

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
    plt.title(f"{word} 점수 추이", fontproperties=fontprop)
    plt.xlabel("날짜", fontproperties=fontprop)
    plt.ylabel("점수", fontproperties=fontprop)
    plt.ylim(0, 100)
    plt.tight_layout()

    buf = BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    return StreamingResponse(buf, media_type="image/png")
