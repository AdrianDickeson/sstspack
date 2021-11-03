import time
import logging

from sstspack import Utilities as utl

import example_nile_data as nile
import example_seatbelt_data as seatbelt

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

stream_handler = utl.getSetupStreamHandler(logging.INFO)

logger.addHandler(stream_handler)

examples = {"Nile": nile, "Seatbelt": seatbelt}


if __name__ == "__main__":
    for name, example in examples.items():
        example.logger.addHandler(stream_handler)

        logger.info(f"Running {name} example")
        start_time = time.time()
        try:
            example.main()
        except Exception:
            logger.exception(f"Unexpected error encountered running {name} example")
        else:
            end_time = time.time()
            logger.info(
                f"{name} example finished: Time taken:- {end_time - start_time:.2f}"
            )
    logger.info("All exmaples finished")
