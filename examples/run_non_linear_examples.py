import time
import logging

from sstspack import Utilities as utl

import example_particle_filter as pf
import example_uk_visits_abroad_data as uk_abroad

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

examples = {"Particle Filter": pf, "UK visitors abroad": uk_abroad}


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
                f"{name} example finished: Time taken: "
                + f"{end_time - start_time:.2f} seconds"
            )
    logger.info("All exmaples finished")
