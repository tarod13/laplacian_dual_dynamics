import time
import uuid


class Timer:

    def __init__(self):
        self._start_time = time.time()
        self._step_time = time.time()
        self._step = 0

    def reset(self):
        self._start_time = time.time()
        self._step_time = time.time()
        self._step = 0

    def set_step(self, step):
        self._step = step
        self._step_time = time.time()

    def time_cost(self):
        return time.time() - self._start_time

    def steps_per_sec(self, step):
        sps = (step - self._step) / (time.time() - self._step_time)
        self._step = step
        self._step_time = time.time()
        return sps


def get_time_identifier():
    '''
        Generate an 8-digit hexadecimal identifier based on the current time.
    '''
    # Get the current time in seconds since the epoch
    current_time = int(time.time())

    # Convert the current time to a hexadecimal string
    hex_time = hex(current_time)[2:]  # Remove the "0x" prefix

    # Pad the hexadecimal time string with zeros to make it 8 characters long
    hex_time = hex_time.zfill(8)

    # Generate a random 4-character hexadecimal string
    random_hex = uuid.uuid4().hex[:4]

    # Combine the two hexadecimal strings to create the 8-digit identifier
    hex_identifier = hex_time + random_hex

    return hex_identifier