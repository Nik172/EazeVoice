"""
Simple WER Calculator for Fine-tuned Whisper Model
Just provide audio file path and ground truth text
"""

import torch
import librosa
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from jiwer import wer

# ==================== CONFIGURATION ====================
MODEL_PATH = "./whisper-torgo-lora"  # Change this to your model path
BASE_MODEL = "openai/whisper-small"  # Change if you used different base model
# =======================================================

def load_model():
    """Load the fine-tuned model"""
    print(f"Loading model from: {MODEL_PATH}")
    
    processor = WhisperProcessor.from_pretrained(BASE_MODEL)
    model = WhisperForConditionalGeneration.from_pretrained(MODEL_PATH)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()
    
    print(f"✅ Model loaded on {device}\n")
    return model, processor, device

def transcribe(audio_path, model, processor, device):
    """Transcribe an audio file"""
    # Load audio
    audio_array, _ = librosa.load(audio_path, sr=16000)
    
    # Extract features
    input_features = processor.feature_extractor(
        audio_array,
        sampling_rate=16000,
        return_tensors="pt"
    ).input_features.to(device)
    
    # Generate transcription
    with torch.no_grad():
        predicted_ids = model.generate(input_features)
    
    # Decode
    transcription = processor.batch_decode(
        predicted_ids,
        skip_special_tokens=True
    )[0]
    
    return transcription.strip()

def calculate_wer(audio_path, ground_truth):
    """Calculate WER for a single audio file"""
    # Load model
    model, processor, device = load_model()
    
    # Get prediction
    print(f"🎤 Transcribing: {audio_path}")
    prediction = transcribe(audio_path, model, processor, device)
    
    # Calculate WER
    wer_score = wer(ground_truth, prediction) * 100
    
    # Display results
    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    print(f"Ground Truth: '{ground_truth}'")
    print(f"Prediction:   '{prediction}'")
    print(f"\nWER: {wer_score:.2f}%")
    print("="*60)
    
    return wer_score, prediction

if __name__ == "__main__":
    # ==================== TEST YOUR AUDIO HERE ====================
    
    # Example 1: Single test
    audio_file = "Treatment of Hypokinetic Dysarthria.wav"
    ground_truth = "and we are working on speech and treating hypokinetic dysarthria because he has Parkinson's disease so now you're going to watch our session in using LSVT principles and see the difference in his speech sound good yes it'd been very helpful you there's a couple times you said what what I gotta gear my speech up to where you hear that the first time I have the same problem with any people in speak with what what that's going to be you're retaining for them as well as me mm-hmm that's why you get to do your exercises every day you're absolutely I agree with you took me a while I realize that but you're absolutely right yeah patients need a practice for 10 to 15 minutes a day to keep their voice Tron's go ahead and start with your AHS and make sure I want make sure you use your loud good quality voice and try to hold it for as long as you can and make sure you think about being loud and your eyes should sound like this ah ah ah ah repeat this exercise 6 to 10 times so with this parameter I want you to make sure that this little ball stays in between the two arrows and I want you to try to inhale nice and slowly so we're going to have the ball stay in between the arrows for 3 to 5 seconds is your goal okay that was 2.4 when you go they kind of uses up a lot of the capacity in your lung so tried you nice and slow Maurice that was 4.96 high-five that was really good all your directions now we're going to work on are the functional phrases so we're gonna have you use your slop strategies which was slop stand for slow loud open pause where are my shoes what's the weather today let's do that one again what's the weather today good so I want you to take a deep breath and I want you to exert pressure into your palms and I want you to try to exert the same strain and pressure into a vocal grunt like Oh hmm okay how to hold it like how you do the eyes I want you to hold it for like let's try five seconds II your voice sound it's so clear it wasn't shaky or anything that's not a really good yes okay how was that okay okay there we go one big kiss for you your wife is in here it's one of zoom hmm and I want you to hold the grudge until your air supply is Garlin ah sorry it's okay so local solution I would like a strong drink"
    
    calculate_wer(audio_file, ground_truth)
    
    # Example 2: Multiple tests
    # test_samples = [
    #     ("audio1.wav", "hello world"),
    #     ("audio2.wav", "this is a test"),
    #     ("audio3.wav", "dysarthric speech sample"),
    # ]
    # 
    # model, processor, device = load_model()
    # total_wer = 0
    # 
    # for audio_path, truth in test_samples:
    #     prediction = transcribe(audio_path, model, processor, device)
    #     wer_score = wer(truth, prediction) * 100
    #     print(f"\nAudio: {audio_path}")
    #     print(f"Truth: '{truth}' | Prediction: '{prediction}' | WER: {wer_score:.2f}%")
    #     total_wer += wer_score
    # 
    # print(f"\nAverage WER: {total_wer/len(test_samples):.2f}%")