import time
import logging

from sstspack import Utilities as utl

import example_nile_data as nile
import example_seatbelt_data as seatbelt
import example_internet_data as internet
import example_motorcycle_data as motorcycle

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

examples = {
    "Nile": nile,
    "Seatbelt": seatbelt,
    "Internet": internet,
    "Motorcycle": motorcycle,
}


if __name__ == "__main__":
    stream_handler = utl.getSetupStreamHandler(logging.INFO)
    logger.addHandler(stream_handler)

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
