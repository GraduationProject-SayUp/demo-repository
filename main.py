# main.py

from fastapi import FastAPI
from routes import evaluate, history, report

app = FastAPI(
    title="SayUp 발음 평가 API",
    description="발음 피드백, 점수 시각화, 리포트 PDF 생성까지!",
    version="1.0.0"
)

# 각 라우터 등록
app.include_router(evaluate.router)
app.include_router(history.router)
app.include_router(report.router)

@app.get("/")
def root():
    return {"message": "SayUp API 정상 작동 중 ✅"}
