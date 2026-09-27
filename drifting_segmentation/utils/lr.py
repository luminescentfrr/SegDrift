import math


def build_lr_lambda(total_steps, warmup_steps=0, schedule="cosine"):
    def lr_lambda(step):
        if warmup_steps and step < warmup_steps:
            return max(step, 1) / warmup_steps
        if schedule == "constant":
            return 1.0
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

    return lr_lambda
