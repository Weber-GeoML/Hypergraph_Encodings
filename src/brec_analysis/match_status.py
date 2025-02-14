"""Enum for match status."""

from enum import Enum


class MatchStatus(Enum):
    """Enum for match status."""

    NO_MATCH = "NO_MATCH"
    EXACT_MATCH = "EXACT_MATCH"
    SCALED_MATCH = "SCALED_MATCH"
    TIMEOUT = "TIMEOUT"
