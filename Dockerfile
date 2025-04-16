# SayUp 발음 평가 서버 - Dockerfile

# 1. Python 베이스 이미지
FROM python:3.9-slim

# 2. 작업 디렉토리 설정
WORKDIR /app

# 3. requirements.txt 복사 및 패키지 설치
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# 4. 전체 소스 복사
COPY . .

# 5. FastAPI 서버 실행 포트
EXPOSE 8000

# 6. 서버 실행 명령어
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]