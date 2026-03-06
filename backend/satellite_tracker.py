import logging
import math
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

import numpy as np
import requests
from sgp4.api import Satrec, jday


logger = logging.getLogger(__name__)


CELESTRAK_BASE = "https://celestrak.org/NORAD/elements/gp.php"
EARTH_RADIUS_KM = 6371.0          # Mean Earth radius
DEFAULT_FETCH_TIMEOUT = 15        # seconds
CACHE_TTL_SECONDS = 300           # Re-fetch TLEs every 5 minutes

# CelesTrak catalogue groups to pull (name → URL query param)
TLE_GROUPS = {
    "stations":   "STATIONS",
    "active":     "ACTIVE",
    "starlink":   "STARLINK",
    "debris":     "DEBRIS",
}

DEFAULT_GROUP = "stations"        # Used when no group is specified



@dataclass
class TLERecord:
    """Raw Two-Line Element set for a single satellite."""
    name:  str
    line1: str
    line2: str


@dataclass
class SatellitePosition:
    """Geocentric Cartesian + geodetic position snapshot."""
    norad_id:  int
    name:      str
    timestamp: str          # ISO-8601 UTC
    # Earth-Centred Inertial (km)
    x: float
    y: float
    z: float
    # Geodetic
    latitude:  float        # degrees
    longitude: float        # degrees
    altitude:  float        # km above ellipsoid
    # Velocity (km/s)
    vx: float
    vy: float
    vz: float
    speed: float            # magnitude km/s


@dataclass
class SatelliteRecord:
    """Full satellite record: metadata + live position."""
    norad_id:    int
    name:        str
    tle_line1:   str
    tle_line2:   str
    satrec:      object       # sgp4.api.Satrec – not JSON-serialisable
    position:    Optional[SatellitePosition] = None



class TLECache:
    """
    Simple in-process TTL cache for raw TLE strings.
    Thread-safety note: adequate for a single-threaded Flask dev server;
    add a threading.Lock for multi-threaded production WSGI.
    """

    def __init__(self, ttl: int = CACHE_TTL_SECONDS) -> None:
        self._store: dict[str, tuple[list[TLERecord], float]] = {}
        self._ttl = ttl

    def get(self, group: str) -> Optional[list[TLERecord]]:
        if group in self._store:
            records, ts = self._store[group]
            if time.monotonic() - ts < self._ttl:
                logger.debug("TLE cache hit for group '%s'", group)
                return records
        return None

    def set(self, group: str, records: list[TLERecord]) -> None:
        self._store[group] = (records, time.monotonic())
        logger.debug("TLE cache updated for group '%s' (%d records)", group, len(records))


_tle_cache = TLECache()



def fetch_tle_data(group: str = DEFAULT_GROUP) -> list[TLERecord]:
    """
    Fetch TLE data for a named CelesTrak group.

    Args:
        group: Key from TLE_GROUPS dict (e.g. 'stations', 'starlink').

    Returns:
        List of TLERecord objects.

    Raises:
        ValueError: Unknown group name.
        requests.RequestException: Network error.
    """
    if group not in TLE_GROUPS:
        raise ValueError(f"Unknown TLE group '{group}'. Valid: {list(TLE_GROUPS)}")

    cached = _tle_cache.get(group)
    if cached is not None:
        return cached

    catalogue = TLE_GROUPS[group]
    url = f"{CELESTRAK_BASE}?GROUP={catalogue}&FORMAT=TLE"
    logger.info("Fetching TLEs from %s", url)

    response = requests.get(url, timeout=DEFAULT_FETCH_TIMEOUT)
    response.raise_for_status()

    records = _parse_tle_text(response.text)
    _tle_cache.set(group, records)
    logger.info("Fetched %d TLE records for group '%s'", len(records), group)
    return records


def _parse_tle_text(raw: str) -> list[TLERecord]:
    """
    Parse a raw TLE file (3-line format) into TLERecord objects.

    Each entry in the file consists of:
        Line 0: Satellite name (24 chars max)
        Line 1: TLE line 1
        Line 2: TLE line 2

    Args:
        raw: Raw text content from CelesTrak.

    Returns:
        List of TLERecord objects.
    """
    lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
    records: list[TLERecord] = []

    i = 0
    while i + 2 < len(lines):
        name  = lines[i]
        line1 = lines[i + 1]
        line2 = lines[i + 2]

        # Basic sanity: TLE lines start with '1 ' and '2 '
        if line1.startswith("1 ") and line2.startswith("2 "):
            records.append(TLERecord(name=name, line1=line1, line2=line2))
            i += 3
        else:
            i += 1  # Skip malformed entry

    return records



def build_satellite_records(tle_records: list[TLERecord]) -> list[SatelliteRecord]:
    """
    Instantiate sgp4 Satrec objects from raw TLE strings.

    Args:
        tle_records: Parsed TLE records.

    Returns:
        List of SatelliteRecord objects (without position yet).
    """
    satellites: list[SatelliteRecord] = []
    for tle in tle_records:
        try:
            satrec = Satrec.twoline2rv(tle.line1, tle.line2)
            satellites.append(
                SatelliteRecord(
                    norad_id=int(satrec.satnum),
                    name=tle.name,
                    tle_line1=tle.line1,
                    tle_line2=tle.line2,
                    satrec=satrec,
                )
            )
        except Exception as exc:
            logger.warning("Failed to parse TLE for '%s': %s", tle.name, exc)
    return satellites


def propagate_satellite(sat: SatelliteRecord, dt: Optional[datetime] = None) -> Optional[SatellitePosition]:
    """
    Propagate a satellite's position to a given datetime using SGP4.

    Args:
        sat: SatelliteRecord containing an initialised Satrec.
        dt:  Target datetime (UTC). Defaults to now.

    Returns:
        SatellitePosition, or None if propagation fails (e.g. decayed orbit).
    """
    if dt is None:
        dt = datetime.now(timezone.utc)

    # Julian date components required by sgp4
    jd, fr = jday(dt.year, dt.month, dt.day, dt.hour, dt.minute, dt.second + dt.microsecond * 1e-6)

    error_code, position, velocity = sat.satrec.sgp4(jd, fr)

    if error_code != 0:
        # SGP4 error codes: 1=mean motion, 2=eccentricity, 3=decay, etc.
        logger.debug("SGP4 error %d for '%s'", error_code, sat.name)
        return None

    x, y, z   = position   # km, ECI frame
    vx, vy, vz = velocity   # km/s, ECI frame

    lat, lon, alt = _eci_to_geodetic(x, y, z, dt)

    return SatellitePosition(
        norad_id=sat.norad_id,
        name=sat.name,
        timestamp=dt.isoformat(),
        x=round(x, 3),
        y=round(y, 3),
        z=round(z, 3),
        latitude=round(lat, 5),
        longitude=round(lon, 5),
        altitude=round(alt, 3),
        vx=round(vx, 5),
        vy=round(vy, 5),
        vz=round(vz, 5),
        speed=round(math.sqrt(vx**2 + vy**2 + vz**2), 5),
    )


def propagate_orbit_track(sat: SatelliteRecord, steps: int = 90, step_seconds: int = 60) -> list[dict]:
    """
    Generate a sequence of positions to draw an orbit track.

    Args:
        sat:          SatelliteRecord.
        steps:        Number of sample points (default: 90 → 1.5 h at 60 s cadence).
        step_seconds: Time between samples in seconds.

    Returns:
        List of dicts with lat/lon/alt/x/y/z keys.
    """
    now = datetime.now(timezone.utc)
    track: list[dict] = []

    for i in range(steps):
        offset_s  = i * step_seconds
        sample_dt = datetime(
            now.year, now.month, now.day,
            now.hour, now.minute, now.second,
            tzinfo=timezone.utc,
        )
        # Manual offset (avoid timedelta import cycle issue)
        total_seconds = (
            sample_dt.hour * 3600 + sample_dt.minute * 60 + sample_dt.second + offset_s
        )
        jd, fr = jday(
            now.year, now.month, now.day,
            total_seconds // 3600 % 24,
            (total_seconds % 3600) // 60,
            total_seconds % 60,
        )
        err, pos, _ = sat.satrec.sgp4(jd, fr)
        if err != 0:
            continue
        x, y, z = pos
        lat, lon, alt = _eci_to_geodetic(x, y, z, now)
        track.append({
            "lat": round(lat, 4),
            "lon": round(lon, 4),
            "alt": round(alt, 3),
            "x":   round(x,   3),
            "y":   round(y,   3),
            "z":   round(z,   3),
        })

    return track



def _eci_to_geodetic(x: float, y: float, z: float, dt: datetime) -> tuple[float, float, float]:
    """
    Convert Earth-Centred Inertial (ECI) coordinates to geodetic (lat, lon, alt).

    Uses the simplified spherical Earth model (adequate for SSA visualisation).
    For high-precision work, replace with an WGS-84 iterative algorithm.

    Args:
        x, y, z : ECI position in km.
        dt      : UTC datetime (needed for Greenwich Sidereal Time).

    Returns:
        (latitude_deg, longitude_deg, altitude_km)
    """
    # Greenwich Mean Sidereal Time (radians) — simplified formula
    j2000_days = (dt - datetime(2000, 1, 1, 12, tzinfo=timezone.utc)).total_seconds() / 86400.0
    gmst = math.fmod(280.46061837 + 360.98564736629 * j2000_days, 360.0)
    gmst_rad = math.radians(gmst)

    # Rotate ECI → ECEF
    cos_g, sin_g = math.cos(gmst_rad), math.sin(gmst_rad)
    x_ecef =  cos_g * x + sin_g * y
    y_ecef = -sin_g * x + cos_g * y
    z_ecef =  z

    # Spherical to geodetic
    r    = math.sqrt(x_ecef**2 + y_ecef**2 + z_ecef**2)
    lat  = math.degrees(math.asin(z_ecef / r))
    lon  = math.degrees(math.atan2(y_ecef, x_ecef))
    alt  = r - EARTH_RADIUS_KM

    return lat, lon, alt



def get_all_satellites(group: str = DEFAULT_GROUP, limit: int = 100) -> list[SatelliteRecord]:
    """
    Fetch TLEs, build Satrec objects, and propagate current positions.

    Args:
        group: TLE group name.
        limit: Maximum number of satellites to return.

    Returns:
        List of SatelliteRecord with .position populated.
    """
    tle_records = fetch_tle_data(group)[:limit]
    satellites  = build_satellite_records(tle_records)

    now = datetime.now(timezone.utc)
    for sat in satellites:
        sat.position = propagate_satellite(sat, now)

    # Filter out satellites with failed propagation
    return [s for s in satellites if s.position is not None]


def satellite_to_dict(sat: SatelliteRecord) -> dict:
    """Serialise a SatelliteRecord to a JSON-safe dictionary."""
    pos = sat.position
    base = {
        "norad_id": sat.norad_id,
        "name":     sat.name,
    }
    if pos:
        base.update({
            "timestamp": pos.timestamp,
            "latitude":  pos.latitude,
            "longitude": pos.longitude,
            "altitude":  pos.altitude,
            "x": pos.x, "y": pos.y, "z": pos.z,
            "vx": pos.vx, "vy": pos.vy, "vz": pos.vz,
            "speed": pos.speed,
        })
    return base