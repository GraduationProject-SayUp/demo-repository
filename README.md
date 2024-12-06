# Welcome to your organization's demo respository
This code repository (or "repo") is designed to demonstrate the best GitHub has to offer with the least amount of noise.

The repo includes an `index.html` file (so it can render a web page), two GitHub Actions workflows, and a CSS stylesheet dependency.

api_etri_text.py => 마이크로 사용자 음성 입력받아서 텍스트로 변환후 표준 발음과 비교

Mfcc , DTW : 소리의 유사성을 기반으로 음소 단위의 발음 정확도와 언어적 맥락을 반영하는데 한계점.

----발전 해야될 것----
1. 음성을 음소단위로 변환하는 음향 모델(Acoustic Model) : kaldi ,Wav2Vec + Phoneme Classification ,Montreal Forced Aligner (MFA)
2. 음소 단위로 정확성을 분석하고 피드백 제공
3. 손실 함수 : 모델의 예측과 실제 라벨간의 차이를 측정하는 역할 →CTC(connectionist Temporal Classfication)함수 사용
4. 최적화 알고리즘 :스토캐스틱 경사 하강법, adam 옵티마이저 등 다양한 최적화 알고리즘 사용.

----데이터 셋----
1. 발음 학습 데이터 : KsponSpeech
2. 음소-음향 매칭 데이터: 언어별 음소와 음향 특성을 매칭한 데이터셋.
3. 한국어 발음 데이터 : 국립국어원 공개 데이터, 네이버 AI 클라우드 데이터셋.
4. 음성 인식 데이터셋 : TIMIT, LibriSpeech,WSJ


