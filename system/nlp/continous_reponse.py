import sounddevice as sd
import numpy as np
from pydub import AudioSegment
from pydub.playback import play
from rag.models import get_model, get_client_for_model, BACKENDS, Backends
import requests
import os
import time
from multiprocessing import Process
from queue import Queue
import threading
from nlp.text_to_speech import request_speech_to_text
import glob
import asyncio

# Constants and Output Directory
timestamp = time.strftime("%Y%m%d-%H%M%S")
output_dir = f"/tmp/recordings_{timestamp}"
print(f"Recording files will be saved to {output_dir}")
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
sample_rate = 44100
duration = 0.2
min_level = -40
silence_threshold = 10
api_key = BACKENDS[Backends.DEEPINFRA].api_key

async def fetch_response(prompt):
    messages = [{
        "role": "system",
        "content": "You are a casual AI assistant"
    }, {
        "role": "user",
        "content": prompt
    }]
    
    model = 'meta-llama/Meta-Llama-3-70B-Instruct'

    completion_params = {
        "model": model,
        "messages": messages,
        "max_tokens": 400,
        "temperature": 0.0,
        "stream": True
    }

    client = get_client_for_model(model, async_client=True)
    response = await client.chat.completions.create(
        **completion_params
    )
    return response

async def process_stream(prompt):
    buffer = ''
    response = await fetch_response(prompt)
    async for chunk in response:
        if len(chunk.choices) > 0:
            delta = chunk.choices[0].delta
            if delta.content:
                buffer += delta.content
                while '.' in buffer:
                    sentence, buffer = buffer.split('.', 1)
                    yield sentence.strip() + '.'
    if buffer:
        yield buffer.strip()

async def process_streamed_response(text):
    async for sentence in process_stream(text):
        print("__>SENTENCE ", sentence, flush=True)

def request_response(text):
    messages = [{
        "role": "system",
        "content": "You are a casual AI assistant"
    }, {
        "role": "user",
        "content": text
    }]
    
    model = 'meta-llama/Meta-Llama-3-70B-Instruct'

    completion_params = {
        "model": model,
        "messages": messages,
        "max_tokens": 400,
        "temperature": 0.0,
    }

    client = get_client_for_model(model)
    response = client.chat.completions.create(
        **completion_params
    )
    res = response.choices[0].message.content
    return res

def recognize_speech(audio_file_path):
    """Function to send an audio file to speech recognition API and print the result."""
    with open(audio_file_path, 'rb') as audio_file:
        headers = {
            'Authorization': f'Bearer {api_key}'
        }
        files = {
            'audio': audio_file
        }
        response = requests.post('https://api.deepinfra.com/v1/inference/openai/whisper-small', headers=headers, files=files)
        
        print(f"Recognition request for {audio_file_path} with status code {response.status_code}", flush=True)
        
    if response.status_code == 200:
        data = response.json()
        text = data.get("text")
        return text
    else:
        print(f"Failed to recognize speech for {audio_file_path} with status code {response.status_code}", flush=True)
        return None
    

def play_audio_files(queue):
    while True:
        audio_file = queue.get()
        if audio_file is None:
            break
        audio_segment = AudioSegment.from_mp3(audio_file)
        play(audio_segment)
        queue.task_done()

def run_async(prompt, queue):
    async def async_process():
        async for sentence in process_stream(prompt):
            print(sentence)
            output_path = f"output_{sentence[:10].replace(' ', '_')}.mp3"
            request_speech_to_text(sentence, output_path)
            queue.put(output_path)

    loop = asyncio.get_event_loop()
    loop.run_until_complete(async_process())

def main(prompt="How is the weather in Aachen today?"):
    audio_queue = Queue()

    # Start the audio playback thread
    playback_thread = threading.Thread(target=play_audio_files, args=(audio_queue,))
    playback_thread.start()

    run_async(prompt, audio_queue)

    # Signal the playback thread to stop
    audio_queue.put(None)
    playback_thread.join()

if __name__ == "__main__":
    main()