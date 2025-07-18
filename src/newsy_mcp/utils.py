import traceback

DEBUG = True


def dprint(msg: str, error: bool = False) -> None:
    """Print debug messages if DEBUG is enabled"""
    if DEBUG:
        prefix = "[ERROR]" if error else "[DEBUG]"
        print(f"{prefix} {msg}")
        if error:
            print(traceback.format_exc())
