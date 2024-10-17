from transformers import TrainerCallback, TrainerState, TrainerControl
import time

class CrocCallback(TrainerCallback):
    def __init__(self):
        super().__init__()
        self.last_log_time = None
        self.elapsed_time = None

    def on_step_begin(self, args, state, control, logs=None, **kwargs):
        self.last_log_time = time.time()

    def on_step_end(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        """
        Event called at the end of a training step.
        """
        if self.last_log_time is not None and state.is_local_process_zero:
            self.elapsed_time = time.time() - self.last_log_time
            self.last_log_time = time.time()

    def on_log(self, args, state, control, logs=None, **kwargs):
        logs['step_time'] = self.elapsed_time