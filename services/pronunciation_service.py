from models.pronunciation_model import PronunciationModel

model = PronunciationModel()
def calculate_pronunciation_score(reference_text, user_text, reference_audio, user_audio):
    syllable_accuracy = PronunciationModel.compare_text_by_syllables(reference_text, user_text)
    character_accuracy = PronunciationModel.compare_text_by_characters(reference_text, user_text)
    jamo_dtw = PronunciationModel.compare_jamo_dtw(reference_text, user_text)
    missing_syllables = PronunciationModel.compare_syllables(reference_text, user_text)
    lcs_score = PronunciationModel.lcs(reference_text, user_text)
    mfcc_similarity = PronunciationModel.compare_mfcc(reference_audio, user_audio)
    mfcc_dtw = PronunciationModel.compare_mfcc_with_dtw(reference_audio, user_audio)

    syllable_score = syllable_accuracy
    character_score = character_accuracy
    jamo_score = 100 - jamo_dtw
    missing_score = max(0, 100 - (missing_syllables * 10))
    lcs_score = (lcs_score / len(reference_text)) * 100
    mfcc_similarity_score = mfcc_similarity * 100
    mfcc_dtw_score = max(0, 100 - (mfcc_dtw / 10))
    mfcc_score = (mfcc_similarity_score * 0.5) + (mfcc_dtw_score * 0.5)

    total_score = (
        syllable_score * 0.1 +
        character_score * 0.1 +
        jamo_score * 0.25 +
        missing_score * 0.05 +
        lcs_score * 0.25 +
        mfcc_score * 0.25
    )

    return {
        "total": total_score,
        "syllable_score": syllable_score,
        "character_score": character_score,
        "jamo_score": jamo_score,
        "missing_score": missing_score,
        "lcs_score": lcs_score,
        "mfcc_score": mfcc_score,
    }

def get_g2p_feedback(ref_text, user_text):
    ref_g2p = model.get_g2p(ref_text)
    user_clean = PronunciationModel.remove_spaces_and_punctuation(user_text)
    return generate_feedback(ref_g2p, user_clean)

def generate_feedback(ref, user):
    ref_jamo = PronunciationModel.split_to_jamo(ref)
    user_jamo = PronunciationModel.split_to_jamo(user)
    feedback = []
    for i, (r, u) in enumerate(zip(ref_jamo, user_jamo)):
        if r != u:
            feedback.append(f"{i+1}번째 발음 차이: '{r}' vs '{u}'")
    return feedback
