"""Conservative local bounds for renewable remote execution authorization."""
from __future__ import annotations

import math

CLOCK_SKEW_SECONDS = 5
AUTHORIZATION_SECONDS = 900


def remote_time_observable(timestamp: float, now: float) -> bool:
    """Allow a small server-ahead timestamp without extending any expiry."""
    return math.isfinite(timestamp) and 0 < timestamp <= now + CLOCK_SKEW_SECONDS


def local_authorization_deadline(lease_expiry: float, received_at: float,
                                 consent_expiry: float = math.inf) -> float:
    """Account for the worst allowed server-ahead clock and the receipt cap."""
    return min(lease_expiry-CLOCK_SKEW_SECONDS,received_at+AUTHORIZATION_SECONDS,
               consent_expiry-CLOCK_SKEW_SECONDS)
