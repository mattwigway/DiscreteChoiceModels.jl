import timeit
import numpy as np
import multiprocessing
import tempfile
import os

"""
This class represents something that can be benchmarked. Subclasses should
override the setup() and measurable() methods. Only the time taken by measurable() will be benchmarked.
"""


class Benchmarkable(multiprocessing.Process):
    def __init__(self, queue):
        super().__init__()
        self.queue = queue

    def run(self):
        # biogeme writes reams of output - hide that
        with tempfile.TemporaryDirectory() as tmpdir:
            os.chdir(tmpdir)
            self.setup()
            extime = timeit.timeit("self.measurable()", number=1, globals={"self": self})
            self.queue.put(extime)

    def setup(self):
        pass

    def measurable(self):
        raise NotImplementedError("Override measurable() in subclass")

    @classmethod
    def benchmark(object, number=100, func=np.median):
        times = np.full(number, np.nan, "float64")
        # do executions sequentially so they don't interfere with each other
        for i in range(number):
            q = multiprocessing.Queue()
            # benchmarkable extends multiprocessing.Process
            p = object(q)
            p.start()
            p.join()
            times[i] = q.get()

        return func(times)
