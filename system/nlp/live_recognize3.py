import sounddevice as sd
import numpy as np
from pydub import AudioSegment
from pydub.playback import play
from rag.models import get_model, get_client_for_model, BACKENDS, Backends
import requests
import os
import time
from multiprocessing import Process, Queue
import threading
from nlp.text_to_speech import request_speech_to_text
from nlp import continous_reponse
import glob

# Constants and Output Directory
timestamp = time.strftime("%Y%m%d-%H%M%S")
output_dir = f"/tmp/recordings_{timestamp}"
print(f"Recording files will be saved to {output_dir}")
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
sample_rate = 44100
duration = 0.2
min_level = -40
silence_threshold = 6
api_key = BACKENDS[Backends.DEEPINFRA].api_key

def audio_recording(queue):
    """ Continuously record audio. """
    with sd.InputStream(samplerate=sample_rate, channels=2, dtype='int16') as stream:
        while True:
            data, _ = stream.read(int(sample_rate * duration))
            queue.put(data)

def timed(func):
    def _w(*a, **k):
        then = time.time()
        res = func(*a, **k)
        elapsed = time.time() - then
        return elapsed, res
    return _w


def audio_processing(queue):
    """ Process and handle the audio data from the queue. """
    index = 0
    silence_counter = 0
    concatenated = None
    while True:
        data = queue.get()
        if data is None:
            break
        
        # Save audio to file
        rec = np.array(data, dtype=np.int16)
        audio_segment = AudioSegment(rec.tobytes(), frame_rate=sample_rate, sample_width=rec.dtype.itemsize, channels=2)
        file_path = os.path.join(output_dir, f"segment_{index:03d}.mp3")

        if audio_segment.dBFS < min_level:
            print("*** Silence detected, not saving segment", audio_segment.dBFS, flush=True)
            silence_counter += 1
        else:
            silence_counter = 0
            print(f"*** Saving segment {file_path} dBFS: {audio_segment.dBFS}", flush=True)
            if concatenated is None:
                concatenated = audio_segment
            else:
                concatenated += audio_segment

            
        if silence_counter >= silence_threshold:
            break

        # audio_segment.export(file_path, format="mp3")
        index += 1

    print("*** Starting processing of segments", flush=True)
    output_file = os.path.join(output_dir, f"recording_{index:03d}s.mp3")
    time_until_answer = 0.0
    elapsed, _ = timed(concatenated.export)(output_file, format="mp3")
    print(f"*** Exported concatenated audio to in {elapsed:.2f}s", flush=True)
    time_until_answer += elapsed

    elapsed, text = timed(recognize_speech)(output_file)
    time_until_answer += elapsed
    print(f"*** Recognized text in {elapsed:.2f}s: {text}", flush=True)

    elapsed, res = timed(continous_reponse.main)(text, output_dir=output_dir)
    if False:
        time_until_answer += elapsed
        print(f"*** Response in {elapsed:.2f}s: {res}", flush=True)
        out_audio = os.path.join(output_dir, "output.mp3")
        elapsed, _ = timed(request_speech_to_text)(res, output_path=out_audio)
        time_until_answer += elapsed
        print(f"*** Text to speech in {elapsed:.2f}s", flush=True)
        
        print(f"*** Total time until answer: {time_until_answer:.2f}s", flush=True)
        
        audio_segment = AudioSegment.from_mp3(out_audio)
        play(audio_segment)

        queue.put(None)
        queue.close()

SEGMENT_MAP = {}
import asyncio

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
        # Print or process your results here
        print(f"Recognition result for {audio_file_path}:", flush=True)
        data = response.json()
        text = data.get("text")

        
        return text
    else:
        print(f"Failed to recognize speech for {audio_file_path} with status code {response.status_code}", flush=True)
        return None
    

def main():
    queue = Queue(maxsize=1000)  # Buffer up to 10 seconds
    recording_process = Process(target=audio_recording, args=(queue,))
    processing_process = Process(target=audio_processing, args=(queue,))
    
    recording_process.start()
    processing_process.start()

    try:
        while True:
            time.sleep(0.1)  # Keep the main thread alive.
    except KeyboardInterrupt:
        queue.put(None)  # Stop signal
    
    recording_process.join()
    processing_process.join()

if __name__ == "__main__":
    main()