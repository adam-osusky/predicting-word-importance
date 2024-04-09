import math


def get_rank_limit(limit: int | float, length: int) -> int:
    if isinstance(limit, int):
        return limit
    elif isinstance(limit, float):
        return math.ceil(length * limit)
