# SayUp 발음 평가 서버

FastAPI 기반 한국어 발음 분석 서버입니다. 사용자가 업로드한 음성 파일과 텍스트를 비교해 발음 점수를 평가합니다.

## 🔧 주요 기능

- 🎙️ 발음 평가 (`/evaluate-pronunciation`)
- 🧠 G2P(문자 → 발음) 기반 비교
- 📈 발음 점수 기록 및 시각화 (`/score-history`, `/score-plot`)
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
uvicorn score_server:app --reload
```

서버가 실행되면 [http://localhost:8000/docs](http://localhost:8000/docs)에서 Swagger UI로 API 테스트 가능

---

## 🧩 디렉토리 구조

```bash
.
├── main.py
├── routes/
│   └── evaluate.py
├── services/
│   └── pronunciation_service.py
├── models/
│   └── pronunciation_model.py
├── static/
├── data/
│   └── standard_pronunciation.wav
├── requirements.txt
└── README.md
```

---

## 🔐 ETRI API 사용법

`models/pronunciation_model.py`에서 `API_KEY`를 발급받은 키로 교체하거나 `.env`로 관리할 수 있습니다.

발급: [ETRI AI Open Platform](https://aiopen.etri.re.kr/)

---

## 📋 API 예시

### POST `/evaluate-pronunciation`

- `file`: 사용자의 음성 파일 (wav)
- `text`: 사용자가 발음해야 할 기준 문장
- `user_id`: 사용자 식별자

---

## 🧠 향후 확장 아이디어

- Whisper 또는 open-korean-speech dataset 기반 학습 모델 연동
- 음성 텍스트 강조 시각화
- 음소 단위 피드백

---

## 📄 라이선스

MIT
