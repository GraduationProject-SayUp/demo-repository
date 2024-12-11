from datasets import load_dataset, Audio

# 1. 데이터셋 로드 및 전처리
dataset = load_dataset("kresnik/zeroth_korean")
dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))
print(dataset["train"].column_names)
