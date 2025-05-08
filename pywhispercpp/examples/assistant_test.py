#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import importlib.metadata
import queue
import time
import threading
from typing import Callable, Optional
import numpy as np
import sounddevice as sd
import pywhispercpp.constants as constants
import webrtcvad
import logging
from pywhispercpp.model import Model

__version__ = importlib.metadata.version('pywhispercpp')

__header__ = f"""
=====================================
PyWhisperCpp
A simple assistant using Whisper.cpp
Version: {__version__}               
=====================================
"""


class AssistantWithPause:
    def __init__(self,
                 model='tiny',
                 input_device: int = None,
                 silence_threshold: int = 8,
                 q_threshold: int = 16,
                 block_duration: int = 30,
                 commands_callback: Callable[[str], None] = None,
                 **model_params):

        self.input_device = input_device
        self.sample_rate = constants.WHISPER_SAMPLE_RATE
        self.channels = 1
        self.block_duration = block_duration
        self.block_size = int(self.sample_rate * self.block_duration / 1000)
        self.q = queue.Queue()

        self.vad = webrtcvad.Vad()
        self.silence_threshold = silence_threshold
        self.q_threshold = q_threshold
        self._silence_counter = 0

        self._stream: Optional[sd.InputStream] = None
        self._running = False  # Controls background thread
        self._lock = threading.Lock()

        self.pwccp_model = Model(model,
                                 print_realtime=False,
                                 print_progress=False,
                                 print_timestamps=False,
                                 single_segment=True,
                                 no_context=True,
                                 language='auto',
                                 **model_params)
        self.commands_callback = commands_callback

    def _audio_callback(self, indata, frames, time, status):
        if status:
            logging.warning(f"underlying audio stack warning: {status}")

        assert frames == self.block_size
        audio_data = map(lambda x: (x + 1) / 2, indata)  # normalize from [-1,+1] to [0,1]
        audio_data = np.fromiter(audio_data, np.float16)
        audio_data = audio_data.tobytes()
        detection = self.vad.is_speech(audio_data, self.sample_rate)

        if detection:
            self.q.put(indata.copy())
            self._silence_counter = 0
        else:
            if self._silence_counter >= self.silence_threshold:
                if self.q.qsize() > self.q_threshold:
                    text = self._transcribe_speech()
                    self._get_text_callback(text)
                    self._silence_counter = 0
            else:
                self._silence_counter += 1

    def _transcribe_speech(self):
        logging.info("Speech detected ...")
        audio_data = np.array([])
        while not self.q.empty():
            audio_data = np.append(audio_data, self.q.get())
        audio_data = np.concatenate([audio_data, np.zeros((int(self.sample_rate) + 10))])
        res = self.pwccp_model.transcribe(audio_data, new_segment_callback=self._new_segment_callback)
        if len(res) == 0:
            return None
        return res[0].get_text()

    def _new_segment_callback(self, seg):
        if self.commands_callback:
            self.commands_callback(seg.text)

    def run_async(self):
        if not self._running:
            self._running = True
            thread = threading.Thread(target=self._loop)
            thread.daemon = True
            thread.start()

    def _loop(self):
        logging.info("AssistantWithPause is idle. Use `resume_listening()` to start.")
        while self._running:
            time.sleep(0.1)

    def resume_listening(self):

        with self._lock:
            if self._stream is not None:
                logging.info("AssistantWithPause already listening.")
                return

            logging.info("AssistantWithPause resumed (listening started).")
            self._stream = sd.InputStream(
                device=self.input_device,
                channels=self.channels,
                samplerate=self.sample_rate,
                blocksize=self.block_size,
                callback=self._audio_callback
            )

            self._stream.start()


    def pause_listening(self):
        with self._lock:
            if self._stream:
                logging.info("AssistantWithPause paused (listening stopped).")
                self._stream.stop()
                self._stream.close()
                self._stream = None

    def stop(self):
        self._running = False
        self.pause_listening()
        logging.info("AssistantWithPause fully stopped.")

    @staticmethod
    def available_devices():
        return sd.query_devices()

    def _get_text_callback(self, text):
        pass


def _main():
    parser = argparse.ArgumentParser(description="", allow_abbrev=True)
    parser.add_argument('-m', '--model', default='tiny.en', type=str,
                        help="Whisper.cpp model, default to %(default)s")
    parser.add_argument('-ind', '--input_device', type=int, default=None,
                        help=f'Id of The input device (aka microphone)\n'
                             f'available devices {AssistantWithPause.available_devices()}')
    parser.add_argument('-st', '--silence_threshold', default=16, type=int,
                        help="The duration of silence after which the inference will run")
    parser.add_argument('-bd', '--block_duration', default=30,
                        help="Minimum time between audio updates in ms")

    args = parser.parse_args()

    my_assistant = AssistantWithPause(model=args.model,
                             input_device=args.input_device,
                             silence_threshold=args.silence_threshold,
                             block_duration=args.block_duration,
                             commands_callback=lambda txt: print('BEBRA', txt))

    my_assistant.run_async()  # Start background runner (idle)

    try:
        while True:
            print("hello pidor")
            cmd = input("Enter command (resume/pause/exit): ").strip().lower()
            if cmd == "resume":
                my_assistant.resume_listening()
            elif cmd == "pause":
                my_assistant.pause_listening()
            elif cmd == "exit":
                my_assistant.stop()
                break
    except KeyboardInterrupt:
        my_assistant.stop()
        print("\nExiting.")



if __name__ == '__main__':
    _main()
