import sounddevice as sd
import numpy as np
from pydub import AudioSegment
import requests
import os
import time
from multiprocessing import Process, Queue
import threading
from nlp.text_to_speech import request_speech_to_text

# Constants and Output Directory
timestamp = time.strftime("%Y%m%d-%H%M%S")
output_dir = f"/tmp/recordings_{timestamp}"
print(f"Recording files will be saved to {output_dir}")
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
sample_rate = 44100
duration = 0.5
min_level = -38
silence_threshold = 3
api_key = ""  # Replace with your DeepInfra API key

def audio_recording(queue):
    """ Continuously record audio. """
    with sd.InputStream(samplerate=sample_rate, channels=2, dtype='int16') as stream:
        while True:
            data, _ = stream.read(int(sample_rate * duration))
            queue.put(data)

def audio_processing(queue):
    """ Process and handle the audio data from the queue. """
    index = 0
    while True:
        data = queue.get()
        if data is None:
            break
        
        # Save audio to file
        rec = np.array(data, dtype=np.int16)
        audio_segment = AudioSegment(rec.tobytes(), frame_rate=sample_rate, sample_width=rec.dtype.itemsize, channels=2)
        file_path = os.path.join(output_dir, f"segment_{index:03d}.mp3")
        audio_segment.export(file_path, format="mp3")
        index += 1

        if index > 1:
            threading.Thread(target=lambda: concatenate_and_recognize(index, queue)).start()
            
import glob
            
def concatenate_and_recognize(index, queue):
    """Concatenate sound files and send to a speech recognition API."""
    concatenated = None
    segments = []
    silence_streak = 0
    recognition_task = None
    
    potential_segements = glob.glob(os.path.join(output_dir, "segment_*.mp3"))
    
    if os.path.exists(os.path.join(output_dir, f"ENDED")):
        print(f"=====> Skipping task caus ended flag found", flush=True)
        return
    
    for segment in potential_segements:
        segment_file = os.path.basename(segment)
        segment_path = os.path.join(output_dir, segment_file)

        try:
            audio_segment = AudioSegment.from_mp3(segment_path)

            if audio_segment.dBFS < min_level:
                print("*** Silence detected, skipping recognition", audio_segment.dBFS, flush=True)
                os.rename(segment_path, segment_path + "_silence")
            else:
                os.rename(segment_path, segment_path + "_included")

        except Exception as e:
            print(f"*** Error reading segment {segment_file}", flush=True)
            audio_segment = None

    silence_files = [os.path.exists(os.path.join(output_dir, f"segment_{i:03d}.mp3_silence")) for i in range(index - silence_threshold, index)]
    
    if not all(silence_files):
        return
            
    included_segments = glob.glob(os.path.join(output_dir, "segment_*.mp3_included"))
    for segment in included_segments:

        segment_file = os.path.basename(segment)
        segment_path = os.path.join(output_dir, segment_file)

        try:
            audio_segment = AudioSegment.from_mp3(segment_path)

            if audio_segment.dBFS < min_level:
                print("*** Silence detected, skipping recognition", audio_segment.dBFS, flush=True)
                os.rename(segment_path, segment_path + "_silence")

        except Exception as e:
            print(f"*** Error reading segment {segment_file}", flush=True)
            audio_segment = None
        segments.append(segment_file)

        print(f"*** ADDED Segment {segment_file} dBFS: {audio_segment.dBFS}", flush=True)
        if concatenated is None:
            concatenated = audio_segment
        else:
            concatenated += audio_segment

    output_file = os.path.join(output_dir, f"recording_{index:03d}s.mp3")
    concatenated.export(output_file, format="mp3")


    with open(os.path.join(output_dir, f"ENDED"), "w") as f:
        f.write("END")

    print(f"=====> Silence streak detected, skipping recognition", flush=True)
    recognize_speech(output_file, index)
    queue.put(None)
    queue.close()


def append_recognized_text(text):
    text = text + "\n"
    if not os.path.exists(os.path.join(output_dir, "recognized.txt")):
        with open(os.path.join(output_dir, "recognized.txt"), "w") as f:
            f.write(text)
    else:
        with open(os.path.join(output_dir, "recognized.txt"), "a") as f:
            f.write(text)
    print(f"*** Recognized text: {text}", flush=True)
    request_speech_to_text(text)

SEGMENT_MAP = {}

def recognize_speech(audio_file_path, index):
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
        segments = data.get("segments")
        
        if len(segments) == 0:
            print(f"*** Silence detected, renaming segment to _ignored: {audio_file_path}", flush=True)
            # if len(segments) == 0: its mostlikely silence -> we renamve the segment to _ignore
            os.rename(audio_file_path, audio_file_path + "_ignored")
            # And then we need to remove the segment from the list of segments
            
            # 1 - read the segments file
            segments_file = os.path.join(output_dir, f"recording_{index:03d}s_segments.txt")
            with open(segments_file, "r") as f:
                lines = f.readlines()
                
            lines = [line.strip() for line in lines]
                
            print(f"Silence detected, removing segment from list of segments: {lines}", flush=True)
            
            for i, line in enumerate(lines):
                if os.path.exists(os.path.join(output_dir, line)):
                    os.rename(os.path.join(output_dir, line), os.path.join(output_dir, line + "_ignored"))
                    print(f"Renamed {line} to {line}_ignored", flush=True)
        else:
            # If segments are detected, we need to create a regognized_segments.txt file

            print(f"*** Segments recognized: {segments}", flush=True)
            pass
        
        append_recognized_text(text)
    else:
        print(f"Failed to recognize speech for {audio_file_path} with status code {response.status_code}", flush=True)

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