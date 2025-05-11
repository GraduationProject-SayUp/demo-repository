# routes/report.py

from fastapi import APIRouter, Query
from fastapi.responses import StreamingResponse, FileResponse, JSONResponse
from datetime import datetime
from io import BytesIO
import matplotlib.pyplot as plt
import pandas as pd
import os
from fpdf import FPDF
from routes.evaluate import SCORE_HISTORY
import matplotlib.font_manager as fm
import matplotlib as mpl

# ✅ 맑은 고딕 설정 (Windows 기본 한글 폰트)
font_path = "C:/Windows/Fonts/malgun.ttf"
font_prop = fm.FontProperties(fname=font_path)
print("폰트 이름:", font_prop.get_name())

plt.rcParams['font.family'] = font_prop.get_name()
mpl.rcParams['axes.unicode_minus'] = False

router = APIRouter()

@router.get("/compare-plot")
async def compare_plot(user_id: str, word: str):
    user_scores = []
    all_scores = []

    for uid, records in SCORE_HISTORY.items():
        for record in records:
            if record["word"] == word:
                all_scores.append(record["score"])
                if uid == user_id:
                    user_scores.append(record["score"])

    if not all_scores or not user_scores:
        return JSONResponse(content={"message": "데이터가 없습니다."}, status_code=404)

    average = sum(all_scores) / len(all_scores)
    latest = user_scores[-1]

    plt.figure(figsize=(6, 4))
    bars = plt.bar(["전체 평균", "내 점수"], [average, latest])
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, height + 1, f"{height:.1f}", ha='center', fontproperties=font_prop)
    plt.ylim(0, 100)
    plt.title(f"{word} 단어 평균 비교", fontproperties=font_prop)
    plt.tight_layout()

    buf = BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    return StreamingResponse(buf, media_type="image/png")

@router.get("/generate-report")
async def generate_report(user_id: str, word: str, start_date: str, end_date: str):
    start_dt = datetime.fromisoformat(start_date)
    end_dt = datetime.fromisoformat(end_date)

    user_data = SCORE_HISTORY.get(user_id, [])
    filtered = [
        r for r in user_data
        if r["word"] == word and start_dt <= datetime.now() <= end_dt
    ]

    if not filtered:
        return JSONResponse(content={"message": "해당 데이터가 없습니다."}, status_code=404)

    df = pd.DataFrame(filtered)
    df["date"] = pd.date_range(end=pd.Timestamp.now(), periods=len(df))
    df = df.sort_values("date")

    avg = df["score"].mean()
    max_score = df["score"].max()
    min_score = df["score"].min()

    # 그래프 저장
    plt.figure(figsize=(6, 3))
    plt.plot(df["date"], df["score"], marker='o')
    plt.title(f"{word} 발음 점수 추이", fontproperties=font_prop)
    plt.xlabel("날짜", fontproperties=font_prop)
    plt.ylabel("점수", fontproperties=font_prop)
    plt.ylim(0, 100)
    plt.tight_layout()
    graph_path = f"{user_id}_{word}_temp_plot.png"
    plt.savefig(graph_path)

    # PDF 생성
    class ReportPDF(FPDF):
        def header(self):
            self.add_font("Malgun", "", "C:/Windows/Fonts/malgun.ttf", uni=True)  # ✅ 맑은 고딕 등록
            self.set_font("Malgun", "", 16)
            self.cell(0, 10, "발음 학습 리포트", ln=True, align='C')

        def content(self):
            self.set_font("Malgun", "", 12)
            self.cell(0, 10, f"사용자: {user_id}", ln=True)
            self.cell(0, 10, f"기간: {start_date} ~ {end_date}", ln=True)
            self.cell(0, 10, f"단어: {word}", ln=True)
            self.cell(0, 10, f"평균 점수: {avg:.1f}", ln=True)
            self.cell(0, 10, f"최고 점수: {max_score:.1f}", ln=True)
            self.cell(0, 10, f"최저 점수: {min_score:.1f}", ln=True)

        def insert_graph(self, path):
            self.image(path, w=180)

    pdf = ReportPDF()
    pdf.add_page()
    pdf.content()
    pdf.insert_graph(graph_path)
    report_path = f"{user_id}_{word}_report.pdf"
    pdf.output(report_path)

    return FileResponse(report_path, media_type="application/pdf", filename=report_path)
