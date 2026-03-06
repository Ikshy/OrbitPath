import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

import numpy as np

from satellite_tracker import SatelliteRecord, propagate_satellite


logger = logging.getLogger(__name__)


THRESHOLD_CRITICAL_KM = 1.0      # Imminent conjunction – immediate action required
THRESHOLD_HIGH_KM     = 5.0      # High risk – manoeuvre recommended
THRESHOLD_MEDIUM_KM   = 25.0     # Medium risk – monitor closely
THRESHOLD_LOW_KM      = 50.0     # Low risk – flagged for awareness

# Minimum altitude filter: debris below this altitude is decaying rapidly
MIN_ALTITUDE_KM = 200.0



@dataclass
class CollisionEvent:
    """A single conjunction event between two satellites."""
    sat1_id:       int
    sat1_name:     str
    sat2_id:       int
    sat2_name:     str
    distance_km:   float
    risk_level:    str           # "CRITICAL" | "HIGH" | "MEDIUM" | "LOW"
    timestamp:     str           # ISO-8601 UTC when distance was evaluated
    # Midpoint of the conjunction (ECI, km)
    midpoint_x:   float
    midpoint_y:   float
    midpoint_z:   float
    # Relative velocity (km/s) – proxy for impact energy
    relative_speed: float
    # Lat/lon of midpoint (approximate, for UI marker placement)
    midpoint_lat:  Optional[float] = None
    midpoint_lon:  Optional[float] = None



def classify_risk(distance_km: float) -> Optional[str]:
    """
    Map a separation distance to a risk level string.

    Returns None when the distance exceeds LOW threshold (no risk flagged).
    """
    if distance_km <= THRESHOLD_CRITICAL_KM:
        return "CRITICAL"
    if distance_km <= THRESHOLD_HIGH_KM:
        return "HIGH"
    if distance_km <= THRESHOLD_MEDIUM_KM:
        return "MEDIUM"
    if distance_km <= THRESHOLD_LOW_KM:
        return "LOW"
    return None



def detect_collisions(satellites: list[SatelliteRecord], dt: datetime | None = None) -> list[CollisionEvent]:
    """
    Evaluate pairwise conjunction risks for a list of satellites.

    Args:
        satellites: List of SatelliteRecord objects (position already populated).
        dt:         Epoch for propagation. Defaults to now (UTC).

    Returns:
        List of CollisionEvent objects sorted by distance (closest first).
    """
    if dt is None:
        dt = datetime.now(timezone.utc)

    timestamp_iso = dt.isoformat()

   
    valid: list[SatelliteRecord] = []
    positions: list[np.ndarray]  = []   # ECI position vectors
    velocities: list[np.ndarray] = []   # ECI velocity vectors

    for sat in satellites:
        pos = sat.position if sat.position else propagate_satellite(sat, dt)
        if pos is None:
            continue
        # Skip very low-altitude (decaying) objects to reduce noise
        if pos.altitude < MIN_ALTITUDE_KM:
            continue
        valid.append(sat)
        positions.append(np.array([pos.x, pos.y, pos.z], dtype=np.float64))
        velocities.append(np.array([pos.vx, pos.vy, pos.vz], dtype=np.float64))

    n = len(valid)
    if n < 2:
        logger.info("Fewer than 2 valid satellites – no conjunctions to evaluate.")
        return []

    pos_array = np.array(positions)   # shape (n, 3)
    vel_array = np.array(velocities)  # shape (n, 3)

    events: list[CollisionEvent] = []

    for i in range(n):
        # Vectorised: compute distances from sat i to all sats j > i
        delta_pos = pos_array[i + 1:] - pos_array[i]          # (n-i-1, 3)
        distances = np.linalg.norm(delta_pos, axis=1)          # (n-i-1,)

        for k, dist in enumerate(distances):
            j = i + 1 + k
            risk = classify_risk(float(dist))
            if risk is None:
                continue  # Below risk threshold, skip

            # Relative velocity magnitude
            delta_vel     = vel_array[j] - vel_array[i]
            rel_speed     = float(np.linalg.norm(delta_vel))

            # Conjunction midpoint (ECI)
            mid           = (pos_array[i] + pos_array[j]) / 2.0
            mid_lat, mid_lon = _eci_midpoint_to_latlon(mid, dt)

            events.append(CollisionEvent(
                sat1_id=valid[i].norad_id,
                sat1_name=valid[i].name,
                sat2_id=valid[j].norad_id,
                sat2_name=valid[j].name,
                distance_km=round(float(dist), 4),
                risk_level=risk,
                timestamp=timestamp_iso,
                midpoint_x=round(float(mid[0]), 3),
                midpoint_y=round(float(mid[1]), 3),
                midpoint_z=round(float(mid[2]), 3),
                relative_speed=round(rel_speed, 4),
                midpoint_lat=mid_lat,
                midpoint_lon=mid_lon,
            ))

    # Sort: closest conjunction first
    events.sort(key=lambda e: e.distance_km)
    logger.info("Conjunction analysis complete: %d events found (n=%d satellites)", len(events), n)
    return events



def collision_summary(events: list[CollisionEvent]) -> dict:
    """
    Aggregate collision events into a summary dict for dashboard display.

    Args:
        events: Output of detect_collisions().

    Returns:
        Dictionary with counts per risk level and the closest conjunction.
    """
    summary: dict = {
        "total_events": len(events),
        "critical":     0,
        "high":         0,
        "medium":       0,
        "low":          0,
        "closest_km":   None,
        "closest_pair": None,
    }

    for event in events:
        level = event.risk_level.lower()
        if level in summary:
            summary[level] += 1

    if events:
        closest = events[0]
        summary["closest_km"]   = closest.distance_km
        summary["closest_pair"] = f"{closest.sat1_name} ↔ {closest.sat2_name}"

    return summary



def collision_event_to_dict(event: CollisionEvent) -> dict:
    """Convert a CollisionEvent dataclass to a JSON-serialisable dictionary."""
    return {
        "sat1_id":        event.sat1_id,
        "sat1_name":      event.sat1_name,
        "sat2_id":        event.sat2_id,
        "sat2_name":      event.sat2_name,
        "distance_km":    event.distance_km,
        "risk_level":     event.risk_level,
        "timestamp":      event.timestamp,
        "midpoint": {
            "x":   event.midpoint_x,
            "y":   event.midpoint_y,
            "z":   event.midpoint_z,
            "lat": event.midpoint_lat,
            "lon": event.midpoint_lon,
        },
        "relative_speed_km_s": event.relative_speed,
    }



def _eci_midpoint_to_latlon(mid: np.ndarray, dt: datetime) -> tuple[Optional[float], Optional[float]]:
    """
    Convert an ECI midpoint vector to approximate geodetic lat/lon.

    Args:
        mid: 3-element NumPy array [x, y, z] in km (ECI).
        dt:  UTC datetime for sidereal time calculation.

    Returns:
        (latitude_deg, longitude_deg) or (None, None) on failure.
    """
    try:
        import math
        j2000_days = (dt - datetime(2000, 1, 1, 12, tzinfo=timezone.utc)).total_seconds() / 86400.0
        gmst       = math.fmod(280.46061837 + 360.98564736629 * j2000_days, 360.0)
        gmst_rad   = math.radians(gmst)

        cos_g, sin_g = math.cos(gmst_rad), math.sin(gmst_rad)
        x_ecef =  cos_g * mid[0] + sin_g * mid[1]
        y_ecef = -sin_g * mid[0] + cos_g * mid[1]
        z_ecef =  mid[2]

        r   = float(np.linalg.norm(mid))
        lat = round(math.degrees(math.asin(z_ecef / r)), 4)
        lon = round(math.degrees(math.atan2(y_ecef, x_ecef)), 4)
        return lat, lon
    except Exception:
        return None, None