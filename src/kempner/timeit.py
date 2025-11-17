"""
  Time a block of code
"""
import time

class Timeit:

    def __init__(self, name: str):

        self._name = name
        self._start = 0

    def __enter__(self):

        self._start = time.thread_time()

    def __exit__(self, *args):

        delta = time.thread_time() - self._start
        print(f"{self._name} took {delta} seconds")
