# SayUp 발음 평가 서버

FastAPI 기반 한국어 발음 분석 서버입니다. 사용자가 업로드한 음성 파일과 텍스트를 비교해 발음 점수를 평가하고, 피드백 이력을 기반으로 발음 개선 과정을 수치화합니다.

## 🔧 주요 기능

- 🎙️ 발음 평가 (`/evaluate-pronunciation`)
- 🧠 G2P(문자 → 발음) 기반 비교
- 📈 발음 점수 기록 및 시각화 (`/score-history`, `/score-plot`)
- 📊 전체 평균 대비 개인 점수 비교 (`/compare-plot`)
- 🧾 개인 리포트 PDF 생성 (`/generate-report`)
- 🧠 ETRI API 연동으로 실시간 발음 인식

---

## 🛠 설치 및 실행

### 1. 가상환경 생성 및 패키지 설치

```bash
python -m venv venv
source venv/bin/activate  # 윈도우: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. 서버 실행

```bash
uvicorn main:app --reload
```

서버가 실행되면 [http://localhost:8000/docs](http://localhost:8000/docs)에서 Swagger UI로 API 테스트 가능

---

## 🧩 디렉토리 구조

```
.
├── main.py
├── routes/
│   ├── evaluate.py
│   ├── history.py
│   └── report.py
├── services/
│   └── pronunciation_service.py
├── models/
│   └── pronunciation_model.py
├── static/
│   └── NanumGothicCoding.ttf
├── data/
│   └── standard_pronunciation.wav
├── requirements.txt
├── Dockerfile
└── README.md
```

---

## 🔐 ETRI API 사용법

`models/pronunciation_model.py`에서 `API_KEY`를 발급받은 키로 교체하거나 `.env` 파일로 분리해 관리할 수 있습니다.

발급: [ETRI AI Open Platform](https://aiopen.etri.re.kr/)

---

## 🧠 발음 피드백 개선 로직

- 사용자 발음 → 음소 단위로 분해 → 표준 발음과 자모 단위 비교
- 실시간 점수 계산 항목:
  - `syllable_score` (음절 정확도)
  - `character_score` (문자 정확도)
  - `jamo_score` (자모 유사도)
  - `lcs_score` (최장 공통 부분 수열 기반 점수)
  - `mfcc_score` (음향 유사도)
  - `missing_score` (누락 음절 수 기반 감점)
- 점수는 `/score-history`로 저장되어 `/score-plot`에서 시각화되고, `/compare-plot`으로 다른 사용자 평균과 비교됩니다.

---

## 🗃️ 데이터베이스 구조 (메모리 기반 예시)

```python
score_history = {
    "user123": [
        {
            "word": "굳이",
            "score": 83.1,
            "breakdown": { ... },
            "date": "2025-04-16T10:21:00"
        },
        ...
    ]
}
```

> 🔁 추후 MongoDB 또는 SQLite로 대체 가능하며, `score_history` 구조와 동일하게 컬렉션 또는 테이블 구성 권장.

---

## 🐳 Docker 배포 가이드

### 1. Dockerfile 작성
루트 디렉토리에 아래 내용의 `Dockerfile`을 작성하세요.

```dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install -r requirements.txt
COPY . .
EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### 2. .dockerignore 파일도 생성 (필수)
```
venv
__pycache__
*.pyc
*.wav
.DS_Store
```

### 3. 이미지 빌드
```bash
docker build -t sayup-server .
```

### 4. 컨테이너 실행
```bash
docker run -p 8000:8000 --name sayup-container sayup-server
```

### 💡 기타 Docker 명령어
- `docker stop sayup-container` : 컨테이너 중지
- `docker start sayup-container` : 재시작
- `docker rm sayup-container` : 컨테이너 삭제
- `docker logs sayup-container` : 로그 확인

---

## 📋 API 예시

### POST `/evaluate-pronunciation`

- `file`: 사용자의 음성 파일 (wav)
- `text`: 사용자가 발음해야 할 기준 문장
- `user_id`: 사용자 식별자

### GET `/score-history`
- 사용자 발음 점수 이력 조회

### GET `/score-plot`
- 점수 추이 그래프 (PNG)

### GET `/compare-plot`
- 평균 대비 사용자 점수 비교 바 그래프

### GET `/generate-report`
- PDF 리포트 생성 (점수 요약 + 그래프 포함)

---

## 📄 라이선스

이 프로젝트는 [MIT 라이선스](LICENSE)를 따릅니다. 자유롭게 사용, 수정, 배포할 수 있으며, 저작권 및 라이선스 고지를 포함해야 합니다.