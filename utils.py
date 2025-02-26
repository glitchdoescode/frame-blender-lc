# utils.py
logs = []

def log_message(message):
    """Add a log message with a timestamp."""
    from datetime import datetime
    logs.append(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - {message}")