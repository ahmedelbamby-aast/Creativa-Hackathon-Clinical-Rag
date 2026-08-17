"""Small scoring helpers shared by retrieval and tests."""


def cosine_distance_to_score(distance: float) -> float:
    """Convert cosine distance to a similarity score in ``[0, 1]``."""
    return round(max(0.0, min(1.0, 1.0 - float(distance))), 4)

