import logging
import threading
from datetime import datetime
from typing import Callable

import sounddevice as sd

class Microphone:
    """
    Recording utility for keyword spotting inference
    Microphone.on_stream creates another Thread and process streamed records in parallel manner
    """

    def __init__(self, duration_ms: int = 1000, sound_rate: int = 16_000, ) -> None:
        self.duration_ms = duration_ms
        self.rate = sound_rate

    def get_chunk(self):
        seconds = self.duration_ms / 1000
        # mono-channel record
        record = sd.rec(int(seconds * self.rate),
                        samplerate=self.rate, channels=1)
        # wait until recording complete
        sd.wait()
        return record

    def __action_loop(self, action: Callable):
        def wrapper():
            logging.info("start microphone action loop..")
            while True:
                record = self.get_chunk()
                start_process = datetime.now()
                action(record)
                logging.info(
                    f"done action process in {(datetime.now() - start_process).seconds}s")

        return wrapper

    def on_stream(self, action: Callable):
        thread = threading.Thread(
            name="microphone_audio_processor",
            target=self.__action_loop(action=action)
        )

        thread.start()
