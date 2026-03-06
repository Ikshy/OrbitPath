import logging
import os
import sys
from datetime import datetime, timezone
from functools import wraps
from typing import Any

from flask import Flask, jsonify, request, Response
from flask_cors import CORS

# Local modules
from satellite_tracker import (
    DEFAULT_GROUP,
    TLE_GROUPS,
    get_all_satellites,
    propagate_orbit_track,
    satellite_to_dict,
)
from collision_detection import (
    collision_event_to_dict,
    collision_summary,
    detect_collisions,
)
from anomaly_detection import (
    anomaly_report_to_dict,
    run_anomaly_detection,
)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def create_app() -> Flask:
    """Create and configure the Flask application."""
    app = Flask(__name__)
    CORS(app)  # Allow cross-origin requests (required for the Three.js frontend)

  
    def api_response(data: Any, status: int = 200, meta: dict | None = None) -> Response:
        """Wrap data in a consistent JSON envelope."""
        body = {
            "success":   True,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "data":      data,
        }
        if meta:
            body["meta"] = meta
        return jsonify(body), status

    def api_error(message: str, status: int = 400) -> Response:
        """Return a standardised error response."""
        return jsonify({
            "success": False,
            "error":   message,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }), status

    def validate_group(func):
        """Decorator: validate the ?group= query parameter."""
        @wraps(func)
        def wrapper(*args, **kwargs):
            group = request.args.get("group", DEFAULT_GROUP).lower()
            if group not in TLE_GROUPS:
                return api_error(
                    f"Invalid group '{group}'. Valid options: {list(TLE_GROUPS.keys())}"
                )
            return func(*args, **kwargs)
        return wrapper

    def parse_limit(default: int = 100, maximum: int = 500) -> int:
        """Parse and clamp the ?limit= query parameter."""
        try:
            limit = int(request.args.get("limit", default))
            return max(1, min(limit, maximum))
        except (TypeError, ValueError):
            return default

    

    @app.get("/health")
    def health():
        """
        Health check endpoint.

        Returns:
            200 OK with service status and UTC timestamp.
        """
        return api_response({"status": "healthy", "service": "OrbitPath API"})


    @app.get("/satellites")
    @validate_group
    def satellites():
        """
        Return current positions for all tracked satellites.

        Query params:
            group (str)  : TLE catalogue group. Default: stations
            limit (int)  : Max number of satellites. Default: 100, max: 500

        Returns:
            JSON list of satellite position objects.
        """
        group = request.args.get("group", DEFAULT_GROUP).lower()
        limit = parse_limit()

        try:
            sats = get_all_satellites(group=group, limit=limit)
        except Exception as exc:
            logger.exception("Failed to fetch satellites")
            return api_error(f"Could not fetch satellite data: {exc}", 502)

        payload = [satellite_to_dict(s) for s in sats]
        return api_response(
            payload,
            meta={
                "count": len(payload),
                "group": group,
                "limit": limit,
            },
        )


    @app.get("/orbits/<int:norad_id>")
    def orbit_track(norad_id: int):
        """
        Return an orbit track (list of lat/lon/alt/x/y/z points) for a single satellite.

        Path param:
            norad_id (int) : NORAD catalogue number.

        Query params:
            group   (str) : TLE group to search. Default: stations
            steps   (int) : Number of sample points. Default: 90  (= ~1.5 h at 60 s)
            step_s  (int) : Seconds between samples. Default: 60

        Returns:
            JSON orbit track for the requested satellite.
        """
        group  = request.args.get("group",  DEFAULT_GROUP).lower()
        steps  = max(10, min(int(request.args.get("steps",  90)),  360))
        step_s = max(10, min(int(request.args.get("step_s", 60)), 300))

        try:
            sats = get_all_satellites(group=group, limit=500)
        except Exception as exc:
            logger.exception("Failed to fetch satellites for orbit track")
            return api_error(f"Data fetch error: {exc}", 502)

        # Find requested satellite
        target = next((s for s in sats if s.norad_id == norad_id), None)
        if target is None:
            return api_error(f"Satellite NORAD ID {norad_id} not found in group '{group}'.", 404)

        try:
            track = propagate_orbit_track(target, steps=steps, step_seconds=step_s)
        except Exception as exc:
            logger.exception("Orbit propagation failed for NORAD %d", norad_id)
            return api_error(f"Propagation error: {exc}", 500)

        return api_response(
            {
                "norad_id": norad_id,
                "name":     target.name,
                "track":    track,
            },
            meta={"steps": steps, "step_seconds": step_s, "points": len(track)},
        )


    @app.get("/collision-risk")
    @validate_group
    def collision_risk():
        """
        Evaluate pairwise conjunction risks for all tracked satellites.

        Query params:
            group  (str) : TLE group. Default: stations
            limit  (int) : Max satellites to analyse. Default: 100

        Returns:
            JSON with collision events and summary statistics.
        """
        group = request.args.get("group", DEFAULT_GROUP).lower()
        limit = parse_limit(default=100, maximum=300)

        try:
            sats = get_all_satellites(group=group, limit=limit)
        except Exception as exc:
            logger.exception("Failed to fetch satellites for collision analysis")
            return api_error(f"Data fetch error: {exc}", 502)

        try:
            events  = detect_collisions(sats)
            summary = collision_summary(events)
        except Exception as exc:
            logger.exception("Collision detection failed")
            return api_error(f"Collision analysis error: {exc}", 500)

        payload = {
            "summary": summary,
            "events":  [collision_event_to_dict(e) for e in events],
        }
        return api_response(
            payload,
            meta={"satellites_analysed": len(sats), "group": group},
        )


    @app.get("/anomaly-report")
    @validate_group
    def anomaly_report():
        """
        Run AI anomaly detection on the current satellite catalogue.

        Query params:
            group    (str)  : TLE group. Default: stations
            limit    (int)  : Max satellites. Default: 100
            retrain  (bool) : Force model retraining ('true'/'1'). Default: false

        Returns:
            JSON anomaly report with per-satellite scores and metadata.
        """
        group        = request.args.get("group",   DEFAULT_GROUP).lower()
        limit        = parse_limit(default=100, maximum=300)
        force_retrain = request.args.get("retrain", "false").lower() in ("true", "1", "yes")

        try:
            sats = get_all_satellites(group=group, limit=limit)
        except Exception as exc:
            logger.exception("Failed to fetch satellites for anomaly detection")
            return api_error(f"Data fetch error: {exc}", 502)

        try:
            reports, model_meta = run_anomaly_detection(sats, force_retrain=force_retrain)
        except ValueError as exc:
            return api_error(str(exc), 422)
        except Exception as exc:
            logger.exception("Anomaly detection failed")
            return api_error(f"Anomaly detection error: {exc}", 500)

        payload = {
            "anomalies": [anomaly_report_to_dict(r) for r in reports if r.is_anomalous],
            "normal":    [anomaly_report_to_dict(r) for r in reports if not r.is_anomalous],
        }
        return api_response(payload, meta={**model_meta, "group": group})


   
    @app.errorhandler(404)
    def not_found(exc):
        return api_error("Endpoint not found.", 404)

    @app.errorhandler(405)
    def method_not_allowed(exc):
        return api_error("Method not allowed.", 405)

    @app.errorhandler(500)
    def internal_error(exc):
        logger.exception("Unhandled server error")
        return api_error("Internal server error.", 500)

    return app



app = create_app()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("FLASK_DEBUG", "false").lower() in ("true", "1")
    logger.info("Starting OrbitPath API on http://localhost:%d  (debug=%s)", port, debug)
    app.run(host="0.0.0.0", port=port, debug=debug)