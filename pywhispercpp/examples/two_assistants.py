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

logging.basicConfig(level=logging.INFO)


class Assistant:
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
        self._running = False
        self._lock = threading.Lock()

        self.pwccp_model = Model(model,
                                 print_realtime=False,
                                 print_progress=False,
                                 print_timestamps=False,
                                 single_segment=True,
                                 no_context=True,
                                 **model_params)
        self.commands_callback = commands_callback

    def _audio_callback(self, indata, frames, time, status):
        if status:
            logging.warning(f"Audio callback warning: {status}")

        assert frames == self.block_size
        audio_data = map(lambda x: (x + 1) / 2, indata)
        audio_data = np.fromiter(audio_data, np.float16)
        audio_data = audio_data.tobytes()
        detection = self.vad.is_speech(audio_data, self.sample_rate)

        if detection:
            self.q.put(indata.copy())
            self._silence_counter = 0
        else:
            if self._silence_counter >= self.silence_threshold:
                if self.q.qsize() > self.q_threshold:
                    self._transcribe_speech()
                    self._silence_counter = 0
            else:
                self._silence_counter += 1

    def _transcribe_speech(self):
        logging.info("Speech detected...")
        audio_data = np.array([])
        while not self.q.empty():
            audio_data = np.append(audio_data, self.q.get())
        audio_data = np.concatenate([audio_data, np.zeros((int(self.sample_rate) + 10))])
        self.pwccp_model.transcribe(audio_data, new_segment_callback=self._new_segment_callback)

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
        logging.info("Assistant is idle. Use `resume_listening()` to start.")
        while self._running:
            time.sleep(0.1)

    def resume_listening(self):
        with self._lock:
            if self._stream is not None:
                logging.info("Already listening.")
                return

            logging.info("Listening started.")
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
                logging.info("Listening paused.")
                self._stream.stop()
                self._stream.close()
                self._stream = None

            # Clear any pending audio data
            with self.q.mutex:
                self.q.queue.clear()
            self._silence_counter = 0

    def stop(self):
        self._running = False
        self.pause_listening()
        logging.info("Assistant fully stopped.")

    @staticmethod
    def available_devices():
        return sd.query_devices()


def _main():
    parser = argparse.ArgumentParser(description="Dual Assistant with State Machine", allow_abbrev=True)
    parser.add_argument('-m', '--model', default='tiny.en', type=str,
                        help="Whisper.cpp model path or name (default: %(default)s)")
    args = parser.parse_args()
    model_path = args.model

    keyword_trigger = "nina"

    state = {
        "mode": "idle",        # idle, keyword, message
        "to_message": False    # флаг на переключение
    }

    # Callback для keyword detection
    def keyword_callback(text):
        logging.info(f"[Keyword Assistant] Heard: {text}")
        if keyword_trigger in text.lower():
            logging.info("Keyword detected! Preparing to switch...")
            state["to_message"] = True

    # Callback для полного сообщения
    def message_callback(text):
        logging.info(f"[Message Assistant] Heard: {text}")
        message_assistant.pause_listening()
        state["mode"] = "idle"
        logging.info("Message captured. Returning to idle.")

    keyword_assistant = Assistant(
        model=model_path,
        silence_threshold=8,
        commands_callback=keyword_callback
    )

    message_assistant = Assistant(
        model=model_path,
        silence_threshold=16,
        commands_callback=message_callback
    )

    keyword_assistant.run_async()
    message_assistant.run_async()

    def main_loop():
        while True:
            if state["to_message"]:
                logging.info("Switching from keyword to message assistant...")
                keyword_assistant.pause_listening()
                time.sleep(0.5)
                message_assistant.resume_listening()
                state["mode"] = "message"
                state["to_message"] = False

            time.sleep(0.1)

    def command_loop():
        while True:
            cmd = input("Enter command (start/exit): ").strip().lower()

            if cmd == "start":
                if state["mode"] == "idle":
                    logging.info("Starting keyword assistant...")
                    keyword_assistant.resume_listening()
                    state["mode"] = "keyword"

            elif cmd == "exit":
                keyword_assistant.stop()
                message_assistant.stop()
                break

    threading.Thread(target=main_loop, daemon=True).start()
    command_loop()


if __name__ == '__main__':
    _main()
