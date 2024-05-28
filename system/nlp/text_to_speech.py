import requests
import os
from rag.models import get_model, get_client_for_model, BACKENDS, Backends

model_id = "VEJVJeXDkdy7O1Yr5tZS"
API_KEY = os.getenv("ELVENLABS_API_KEY")

def request_speech_to_text(prompt, output_path="output.mp3"):

    url = f"https://api.elevenlabs.io/v1/text-to-speech/{model_id}"
    CHUNK_SIZE = 1024

    data = {
        "text": prompt,
        "model_id": "eleven_multilingual_v2",
        "voice_settings": {
            "stability": 0.5,
            "similarity_boost": 0.5
        },
        "seed": 123,
    }
    headers = {
        "Accept": "audio/mpeg",
        "Content-Type": "application/json",
        "xi-api-key": API_KEY
    }


    response = requests.post(url, json=data, headers=headers)
    with open(output_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=CHUNK_SIZE):
            if chunk:
                f.write(chunk)
    
    print(f"*** Text to speech in {output_path}, {response.status_code}", flush=True)
    if response.status_code != 200:
        print(f"*** Error in text to speech: {response.text}", flush=True)
    return output_path

def requset_text_to_speech_openai(
    prompt: str,
    output_path: str = "output.mp3"
):
    pass
    from openai import OpenAI
    client = OpenAI(
        api_key=BACKENDS[Backends.OPENAI].api_key,
    )

    response = client.audio.speech.create(
      model="tts-1",
      voice="nova",#"alloy",
      input=prompt
    )
    response.stream_to_file(output_path)
    print(f"*** Text to speech in {output_path}", flush=True)
    return output_path
                
    
if __name__ == "__main__":
    request_speech_to_text("Hello, how are you?")
