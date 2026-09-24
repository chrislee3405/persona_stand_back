"""Exceptions are diagnostic metadata, not an alternate prompt logging channel."""
import logging
import traceback


class SafeExceptionFilter(logging.Filter):
    def filter(self, record):
        if record.exc_info:
            # Do not format exception values/chains or source lines: SQL parameters,
            # provider response bodies and even a source line can contain content.
            frames = traceback.extract_tb(record.exc_info[2])
            locations = " -> ".join(f"{f.name}:{f.lineno}" for f in frames)
            record.exc_text = f"{record.exc_info[0].__name__} at {locations} (details omitted)"
            record.exc_info = None
        return True


def configure_safe_logging():
    # Sanitize before *any* handler, including handlers added later by uvicorn
    # or test capture. Preserve an existing custom factory and install once.
    previous = logging.getLogRecordFactory()
    if getattr(previous, "_persona_safe_exceptions", False):
        return
    sanitizer = SafeExceptionFilter()

    def factory(*args, **kwargs):
        record = previous(*args, **kwargs)
        sanitizer.filter(record)
        return record

    factory._persona_safe_exceptions = True
    logging.setLogRecordFactory(factory)
