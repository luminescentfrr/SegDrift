import ast
from pathlib import Path


def _parse_scalar(value):
    value = value.strip()
    if value in {"true", "True"}:
        return True
    if value in {"false", "False"}:
        return False
    if value in {"null", "None"}:
        return None
    try:
        return ast.literal_eval(value)
    except Exception:
        return value.strip("'\"")


def load_config(path):
    """Load a small YAML config without requiring PyYAML."""
    path = Path(path)
    root = {}
    stack = [(-1, root)]
    for raw_line in path.read_text().splitlines():
        line = raw_line.split("#", 1)[0].rstrip()
        if not line.strip():
            continue
        indent = len(line) - len(line.lstrip(" "))
        key, _, value = line.strip().partition(":")
        while stack and indent <= stack[-1][0]:
            stack.pop()
        parent = stack[-1][1]
        if value.strip() == "":
            node = {}
            parent[key] = node
            stack.append((indent, node))
        else:
            parent[key] = _parse_scalar(value)
    return root


def deep_update(config, updates):
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(config.get(key), dict):
            deep_update(config[key], value)
        else:
            config[key] = value
    return config
