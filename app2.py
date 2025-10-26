import gradio as gr
import numpy as np
import soundfile as sf
import noisereduce as nr
import tempfile 
import torch

from transformers import WhisperForConditionalGeneration, WhisperProcessor
from peft import PeftModel
   
# Load your fine-tuned model
base_model = WhisperForConditionalGeneration.from_pretrained('openai/whisper-small')
model = PeftModel.from_pretrained(base_model, './whisper-torgo-lora')
processor = WhisperProcessor.from_pretrained('./whisper-torgo-lora')

# Set to evaluation mode
model.eval()
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

def transcribe(audio, progress=gr.Progress(track_tqdm=True)):
    if audio is None:
        return None, None, "⚠ Please record or upload an audio file.", None, None

    sr, data = audio
    data = np.array(data, dtype=np.float32)

    progress(0, desc="🔊 Reducing background noise...")
    noise_sample = data[:int(sr * 0.5)] if len(data) > sr * 0.5 else data
    enhanced = nr.reduce_noise(y=data, y_noise=noise_sample, sr=sr)
    enhanced /= np.max(np.abs(enhanced)) if np.max(np.abs(enhanced)) > 0 else 1

    progress(0.5, desc="💾 Saving audio files...")
    orig_path = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name
    enh_path = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name
    sf.write(orig_path, data, sr)
    sf.write(enh_path, enhanced, sr)

    progress(0.8, desc="🧠 Transcribing with Whisper...")
    
    # Process audio with transformers
    audio_input = processor(enhanced, sampling_rate=sr, return_tensors="pt").input_features
    audio_input = audio_input.to(device)
    
    # Generate transcription
    with torch.no_grad():
        predicted_ids = model.generate(audio_input)
        text = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]

    txt_path = tempfile.NamedTemporaryFile(suffix=".txt", delete=False).name
    with open(txt_path, "w") as f:
        f.write(text)

    progress(1.0, desc="✅ Done!")

    return orig_path, enh_path, text, enh_path, txt_path

with gr.Blocks(title="🎙 Whisper Speech-to-Text") as demo:
    gr.Markdown(
        """
        ## 🎧 Whisper Fine-Tuned Speech-to-Text  
        Record or upload audio — the app will clean it, compare versions, and transcribe using Whisper.
        """
    )

    audio_input = gr.Audio(sources=["microphone", "upload"], type="numpy", label="Input Audio")
    transcribe_btn = gr.Button("Transcribe", variant="primary")

    with gr.Row():
        orig_audio = gr.Audio(label="Original Audio", type="filepath")
        enh_audio = gr.Audio(label="Enhanced Audio", type="filepath")

    text_box = gr.Textbox(label="📝 Transcribed Text", lines=4)
    
    with gr.Row():
        download_enh = gr.File(label="⬇ Download Enhanced Audio")
        download_text = gr.File(label="⬇ Download Transcription")

    transcribe_btn.click(
        fn=transcribe,
        inputs=audio_input,
        outputs=[orig_audio, enh_audio, text_box, download_enh, download_text]
    )
    
demo.launch(share=True)