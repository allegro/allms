from dataclasses import dataclass


@dataclass
class GeneralDefaults:
    MAX_RETRIES = 8
    MAX_CONCURRENCY = 1000
