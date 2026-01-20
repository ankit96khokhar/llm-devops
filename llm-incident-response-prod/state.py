
import time

COOLDOWN = 60  # seconds
_state = {}

def should_act(key):
    if key not in _state:
        return True
    return time.time() - _state[key] > COOLDOWN

def record_action(key):
    _state[key] = time.time()
