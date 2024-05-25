import sounddevice as sd
import numpy as np
from pydub import AudioSegment
import requests
import os
import time
import threading

# Directory for recordings
output_dir = "recordings"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Configuration
sample_rate = 44100 # Sample rate in Hz
duration = 1  # Record in chunks of 2 second
silence_threshold = 3

def record_audio():
    """Continuously record audio."""
    while True:
        myrecording = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=2, dtype='int16')
        sd.wait()  # Wait until recording is finished
        yield myrecording

def process_recording(index, rec):
    """Save and process recorded audio."""
    # Convert numpy array to audio segment
    audio_segment = AudioSegment(
        rec.tobytes(),
        frame_rate=sample_rate,
        sample_width=rec.dtype.itemsize,
        channels=rec.shape[1]
    )
    # Save audio to an MP3 file
    segment_file = os.path.join(output_dir, f"segment_{index:03d}.mp3")
    audio_segment.export(segment_file, format="mp3")
    return segment_file

def concatenate_and_recognize(index):
    """Concatenate sound files and send to a speech recognition API."""
    concatenated = None
    for i in range(index):
        segment_path = os.path.join(output_dir, f"segment_{i:03d}.mp3")
        audio_segment = AudioSegment.from_mp3(segment_path)
        if concatenated is None:
            concatenated = audio_segment
        else:
            concatenated += audio_segment
            
    # silence check
    for i in range(index-silence_threshold, index):
        segment_path = os.path.join(output_dir, f"segment_{i:03d}.mp3")
        audio_segment = AudioSegment.from_mp3(segment_path)
        print("*** Silence detected, skipping recognition", audio_segment.dBFS, flush=True)
        if audio_segment.dBFS < -50:
            print("*** Silence detected, skipping recognition", flush=True)

    output_file = os.path.join(output_dir, f"recording_{index:03d}s.mp3")
    concatenated.export(output_file, format="mp3")

    # Send request to API for speech recognition
    # recognize_speech(output_file)

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
        
    if response.status_code == 200:
        # Print or process your results here
        print(f"Recognition result for {audio_file_path}:", flush=True)
        print(response.json())
        
        text = response.json().get("text") + "\n"
        
        # check if "recognized.txt" exists
        if not os.path.exists(os.path.join(output_dir, "recognized.txt")):
            with open(os.path.join(output_dir, "recognized.txt"), "w") as f:
                f.write(text)
        else:
            with open(os.path.join(output_dir, "recognized.txt"), "a") as f:
                f.write(text)
    else:
        print(f"Failed to recognize speech for {audio_file_path} with status code {response.status_code}", flush=True)

def main():
    rec_gen = record_audio()
    for index, rec in enumerate(rec_gen):
        process_recording(index, rec)
        if index > 0:
            threading.Thread(target=concatenate_and_recognize, args=(index+1,)).start()
        time.sleep(duration)  # Pause for `duration` seconds

if __name__ == "__main__":
    main()