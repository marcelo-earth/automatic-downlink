"""SimSat API client for fetching satellite position and imagery."""

from __future__ import annotations

import json
from dataclasses import dataclass
from io import BytesIO
from typing import Any

import requests
from PIL import Image


@dataclass
class SatellitePosition:
    lon: float
    lat: float
    alt: float
    timestamp: str


@dataclass
class SentinelMetadata:
    image_available: bool
    source: str
    spectral_bands: list[str]
    footprint: Any
    size_km: float
    cloud_cover: float | None
    datetime: str | None
    satellite_position: list[float] | None = None
    timestamp: str | None = None


@dataclass
class MapboxMetadata:
    target_visible: bool
    image_available: bool
    elevation_degrees: float
    zoom_factor: float
    bearing: float
    pitch: float
    satellite_position: list[float] | None = None
    timestamp: str | None = None


@dataclass
class SentinelResult:
    image: Image.Image | None
    metadata: SentinelMetadata


@dataclass
class MapboxResult:
    image: Image.Image | None
    metadata: MapboxMetadata


class SimSatClient:
    """Client for the SimSat satellite simulation API."""

    def __init__(self, base_url: str = "http://localhost:9005", timeout: int = 60):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def _get(self, endpoint: str, params: dict | None = None) -> requests.Response:
        url = f"{self.base_url}{endpoint}"
        response = requests.get(url, params=params, timeout=self.timeout)
        response.raise_for_status()
        return response

    def _parse_image(self, content: bytes) -> Image.Image | None:
        if not content:
            return None
        try:
            return Image.open(BytesIO(content)).convert("RGB")
        except Exception:
            return None

    def get_position(self) -> SatellitePosition:
        """Get current satellite position."""
        response = self._get("/data/current/position")
        data = response.json()
        lon_lat_alt = data["lon-lat-alt"]
        return SatellitePosition(
            lon=lon_lat_alt[0],
            lat=lon_lat_alt[1],
            alt=lon_lat_alt[2],
            timestamp=data["timestamp"],
        )

    def get_sentinel_current(
        self,
        spectral_bands: list[str] | None = None,
        size_km: float = 5.0,
        window_seconds: float = 864000,
    ) -> SentinelResult:
        """Get Sentinel-2 image at current satellite position."""
        params: dict[str, Any] = {
            "spectral_bands": spectral_bands or ["red", "green", "blue"],
            "size_km": size_km,
            "return_type": "png",
            "window_seconds": window_seconds,
        }
        response = self._get("/data/current/image/sentinel", params=params)
        raw_meta = json.loads(response.headers.get("sentinel_metadata", "{}"))
        metadata = SentinelMetadata(**raw_meta)
        image = self._parse_image(response.content) if metadata.image_available else None
        return SentinelResult(image=image, metadata=metadata)

    def get_sentinel_historical(
        self,
        lon: float,
        lat: float,
        timestamp: str,
        spectral_bands: list[str] | None = None,
        size_km: float = 5.0,
        window_seconds: float = 864000,
    ) -> SentinelResult:
        """Get Sentinel-2 image for a specific location and time."""
        params: dict[str, Any] = {
            "lon": lon,
            "lat": lat,
            "timestamp": timestamp,
            "spectral_bands": spectral_bands or ["red", "green", "blue"],
            "size_km": size_km,
            "return_type": "png",
            "window_seconds": window_seconds,
        }
        response = self._get("/data/image/sentinel", params=params)
        raw_meta = json.loads(response.headers.get("sentinel_metadata", "{}"))
        metadata = SentinelMetadata(**raw_meta)
        image = self._parse_image(response.content) if metadata.image_available else None
        return SentinelResult(image=image, metadata=metadata)

    def get_mapbox_current(
        self,
        lon: float | None = None,
        lat: float | None = None,
    ) -> MapboxResult:
        """Get Mapbox image from current satellite position."""
        params: dict[str, Any] = {}
        if lon is not None:
            params["lon"] = lon
        if lat is not None:
            params["lat"] = lat
        response = self._get("/data/current/image/mapbox", params=params)
        raw_meta = json.loads(response.headers.get("mapbox_metadata", "{}"))
        metadata = MapboxMetadata(**raw_meta)
        image = self._parse_image(response.content) if metadata.image_available else None
        return MapboxResult(image=image, metadata=metadata)

    def get_mapbox_historical(
        self,
        lon_target: float,
        lat_target: float,
        lon_satellite: float,
        lat_satellite: float,
        alt_satellite: float,
    ) -> MapboxResult:
        """Get Mapbox image for specific satellite and target positions."""
        params: dict[str, Any] = {
            "lon_target": lon_target,
            "lat_target": lat_target,
            "lon_satellite": lon_satellite,
            "lat_satellite": lat_satellite,
            "alt_satellite": alt_satellite,
        }
        response = self._get("/data/image/mapbox", params=params)
        raw_meta = json.loads(response.headers.get("mapbox_metadata", "{}"))
        metadata = MapboxMetadata(**raw_meta)
        image = self._parse_image(response.content) if metadata.image_available else None
        return MapboxResult(image=image, metadata=metadata)

    def is_healthy(self) -> bool:
        """Check if the SimSat API is reachable."""
        try:
            response = self._get("/")
            return response.status_code == 200
        except Exception:
            return False
