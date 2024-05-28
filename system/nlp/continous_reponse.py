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
from nlp.text_to_speech import request_speech_to_text, requset_text_to_speech_openai
import glob
import asyncio

def timed(func):
    def _w(*a, **k):
        then = time.time()
        res = func(*a, **k)
        elapsed = time.time() - then
        return elapsed, res
    return _w

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
    
    #model = 'meta-llama/Meta-Llama-3-70B-Instruct'
    model = 'llama3-70b-8192'

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
    delimiters = ['.'] # ['.', '?', '!', ',']
    start_time = time.time()
    elased_time = 0
    response = await fetch_response(prompt)
    async for chunk in response:
        if len(chunk.choices) > 0:
            delta = chunk.choices[0].delta
            if delta.content:
                buffer += delta.content
                while any(delimiter in buffer for delimiter in delimiters):
                    for delimiter in delimiters:
                        if delimiter in buffer:
                            sentence, buffer = buffer.split(delimiter, 1)
                            elased_time = time.time() - start_time
                            print(f"*** EMMITED Elapsed time: {elased_time:.2f} seconds", flush=True)
                            yield sentence.strip() + delimiter
                            break
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
    

def play_audio_files(queue, output_dir="/tmp/recordings"):
    while True:
        audio_file = queue.get()
        if audio_file is None:
            break
        audio_segment = AudioSegment.from_mp3(output_dir + "/" + audio_file)
        play(audio_segment)
        queue.task_done()

def run_async(prompt, queue, output_dir="/tmp/recordings"):
    async def async_process():
        async for sentence in process_stream(prompt):
            print(sentence)
            output_path = f"output_{sentence[:10].replace(' ', '_')}.mp3"
            #request_speech_to_text(sentence, output_dir + "/" + output_path)
            requset_text_to_speech_openai(
                sentence,
                output_path=output_dir + "/" + output_path
            )
            queue.put(output_path)

    loop = asyncio.get_event_loop()
    loop.run_until_complete(async_process())

def main(prompt="How is the weather in Aachen today?", output_dir="/tmp/recordings"):
    audio_queue = Queue()

    # Start the audio playback thread
    playback_thread = threading.Thread(target=play_audio_files, args=(audio_queue, output_dir))
    playback_thread.start()

    run_async(prompt, audio_queue, output_dir=output_dir)

    # Signal the playback thread to stop
    audio_queue.put(None)
    playback_thread.join()

if __name__ == "__main__":
    main()