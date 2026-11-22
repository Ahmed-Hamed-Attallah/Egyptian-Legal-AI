import logging
import time
from contextlib import contextmanager

logger = logging.getLogger("rag_pipeline")


@contextmanager
def trace_step(step_name: str):
    start = time.perf_counter()

    logger.info("▶ START: %s", step_name)

    try:
        yield

        elapsed = time.perf_counter() - start
        logger.info(
            "✓ SUCCESS: %s | %.3f seconds",
            step_name,
            elapsed
        )

    except Exception:
        elapsed = time.perf_counter() - start

        logger.exception(
            "✗ FAILED: %s | %.3f seconds",
            step_name,
            elapsed
        )

        raise