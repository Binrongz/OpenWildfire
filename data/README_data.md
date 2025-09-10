# 📁 Data Overview

This document provides a description of the data files used in this project. Due to privacy or licensing restrictions, **the full datasets are not included**, but a **sample dataset** is provided for demonstration and testing purposes.

---

## 🔥 1. `nested_california_fire_risk_enhanced_dataset.jsonl`

**Purpose**: Provides spatial, administrative, historical fire, and land-type data for each 5km grid point across California.

**Structure**: JSON Lines (`.jsonl`) — one JSON object per grid point.

### Sample Fields:

| Field Path                        | Description                                 | Example                         |
|----------------------------------|---------------------------------------------|---------------------------------|
| `location.name`                  | Grid point ID                               | `"GridPoint_12345"`             |
| `location.coordinates.lat`       | Latitude                                    | `38.5563`                       |
| `location.coordinates.lon`       | Longitude                                   | `-121.4922`                     |
| `admin.city`                     | City name (if available)                    | `"Sacramento"`                  |
| `admin.county`                   | County name                                 | `"Sacramento County"`           |
| `land.land_type`                | Type of land (Private, Federal, etc.)       | `"Private Land"`                |
| `fire_risk.fhsz_risk_level`      | FHSZ-defined wildfire risk level            | `"Moderate"`                    |
| `fire_risk.fire_count`           | Historical fire count                       | `3`                             |
| `fire_risk.fire_dates`           | List of historical fire incident dates      | `["6/2010", "7/2015"]`          |
| `resources.nearby_stations`      | Text description of nearby fire stations    | `"Station A (10.1 km); ..."`    |

---

## 📷 2. `camera_monitoring_dataset.jsonl`

**Purpose**: Simulated or real-world camera monitoring data near high-risk areas.

**Structure**: JSON Lines — one record per camera unit.

### Sample Fields:

| Field                | Description                              | Example                                 |
|---------------------|------------------------------------------|-----------------------------------------|
| `camera_id`          | Unique camera identifier                 | `"CAM_Tulare_GridPoint_17419_01"`       |
| `lat`                | Latitude                                 | `36.215`                                |
| `lon`                | Longitude                                | `-119.325`                              |
| `detection_status`   | Fire/smoke/normal                        | `"fire"`                                |
| `timestamp`          | Detection time (ISO 8601)                | `"2025-08-15T12:30:00Z"`                |

> Note: In this version, camera locations and detections are randomly simulated for demonstration purposes.

---

## 🌦️ 3. Weather Data (Dynamic)

**Source**: [OpenWeatherMap API](https://openweathermap.org/api)

**Usage**: Retrieved at runtime via `weather_fetcher.py`

**Data Points**:
- Temperature (°C/°F)
- Humidity (%)
- Wind Speed (m/s and mph)
- Weather condition (`"overcast clouds"`, `"clear sky"`, etc.)

> ⚠️ Not stored locally — fetched dynamically during AI risk assessment.

---

## 🚒 4. Fire Station Coverage

**Derived from**: Geospatial station database or external CSV sources

**Used in**: 
- Emergency resource mapping
- Report generation

**Fields Example**:
| Field           | Description                         | Example                            |
|----------------|-------------------------------------|------------------------------------|
| `name`          | Station name                        | `"Fawn Lodge FS"`                  |
| `distance_km`   | Distance to grid center (km)        | `10.1`                             |
| `location`      | Full address / contact info         | `"60 Fawn Lodge Road, 96052"`      |

> 🔐 Source dataset not included; used internally for demonstration reports.

---

## 📁 Sample Files Included

- `data/sample_grid_data.jsonl` — small subset of 10 grid points
- `data/sample_camera_data.jsonl` — small camera cluster (randomized)

---

## ❗ Notes

- All samples are **anonymized or simulated**.
- Full-scale execution requires access to the original datasets.
- If you're a reviewer or judge and need temporary access, please contact the author.
