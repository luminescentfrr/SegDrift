from pathlib import Path


class TeeLogger:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, message):
        print(message)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(str(message) + "\n")
