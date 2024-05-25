import requests
import os

model_id ="21m00Tcm4TlvDq8ikWAM"
API_KEY = os.getenv("ELVENLABS_API_KEY")

def request_speech_to_text(prompt, output_path="output.mp3"):

    url = f"https://api.elevenlabs.io/v1/text-to-speech/{model_id}"
    CHUNK_SIZE = 1024

    data = {
        "text": prompt,
        "model_id": "eleven_monolingual_v1",
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
    
    print(f"*** Text to speech in {output_path}, {response.status_code} {response.text}", flush=True)
    return output_path
                
    
if __name__ == "__main__":
    request_speech_to_text("Hello, how are you?")

