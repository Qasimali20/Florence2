import streamlit as st
from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM
from gtts import gTTS
import torch
import io

model = AutoModelForCausalLM.from_pretrained("microsoft/Florence-2-large", trust_remote_code=True)
processor = AutoProcessor.from_pretrained("microsoft/Florence-2-base", trust_remote_code=True)

def generate_description(image):
    prompt = "<MORE_DETAILED_CAPTION>"
    inputs = processor(text=prompt, images=image, return_tensors="pt")
    with torch.no_grad():
        generated_ids = model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=1024,
            do_sample=False,
            num_beams=3,
        )
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    cleaned_description = generated_text.replace("<s>", "").replace("</s>", "").strip()
    return cleaned_description

def text_to_speech(text, output_path):
    tts = gTTS(text)
    tts.save(output_path)

    st.title("Image Description to Audio App")
    st.write("Upload an image and get its description converted into audio!")

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        description = generate_description(image)

        audio_output_path = "/tmp/audio_output.mp3"  
        text_to_speech(description, audio_output_path)

        st.audio(audio_output_path, format='audio/mp3', start_time=0)

        st.write(f"**Generated Description:** {description}")

if __name__ == "__main__":
    main()
