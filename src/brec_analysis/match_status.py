from enum import Enum


class MatchStatus(Enum):
    NO_MATCH = "NO_MATCH"
    EXACT_MATCH = "EXACT_MATCH"
    SCALED_MATCH = "SCALED_MATCH"
