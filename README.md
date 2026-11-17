# Find My BR Train

A real-time train tracking system for Bangladesh Railway (BR) trains. This FastAPI-based server provides live train position updates and allows users to contribute location data.

## 🚂 About

This project helps passengers track Bangladesh Railway trains in real-time by crowdsourcing location updates from users. The system processes multiple user reports to provide accurate train positions.

## 🛠️ Setup

### Prerequisites
- Python 3.8+
- pip

### Installation
1. Clone the repository:
```bash
git clone https://github.com/jisangain/find-my-br-train.git
cd find-my-br-train
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the server:
```bash
python3 main.py
```

The server will start on `http://localhost:8000`

## 📊 API Endpoints

- `GET /` - API information and available endpoints
- `GET /health` - Server health check
- `GET /initrevision` - Get current data revision
- `GET /alltrains` - Download complete train database
- `GET /current/{train_ids}` - Get current positions for specified trains
- `POST /sendupdate` - Submit location update
- `POST /fix` - Report incorrect information
- `GET /docs` - Interactive API documentation
- `WS /ws/live` - Live positions over one WebSocket (subscribe to trains, riders send updates) - see `functions/live_hub.py` for the protocol
- `GET /history/{train_id}/delays?days=7` - Per-day delay over the last 7/14/30 days
- `GET /predict/{train_id}` - Predicted delay and ETA at each upcoming stop, learned from 30 days of history

## 🛤️ Live tracking: track geometry, speed, history

- **Track geometry** - `track/track.db` holds the real rail polyline between every pair of adjacent stations, built from OpenStreetMap by `tools/build_track.py` (see its docstring for the Overpass query). The same file ships inside the app as `assets/track/track.db`. Rebuild after changing stations/routes:
  ```bash
  python3 tools/build_track.py --app-asset ../assets/track/track.db --preview 702
  ```
  `track/track_report.txt` lists station coordinates that are far from any track (likely wrong) and pairs that fall back to straight lines. Track data © OpenStreetMap contributors (ODbL).
- **Speed** - derived per train from along-track progress of the aggregated position, blended with riders' phone GPS speed (`functions/motion.py`), and included as `speed_kmh` / `moving` in `/current`, `/activetrains` and socket updates.
- **History** - every train's aggregated position is recorded to `history.db` (SQLite, 30-day retention, train-level only - no user ids). Seed it from the existing GPS log once after deploying:
  ```bash
  python3 tools/backfill_history.py --gps gps_updates.db --history history.db
  ```
- **Reverse proxy** - `/ws/live` is a WebSocket: the proxy in front of uvicorn must pass the `Upgrade`/`Connection` headers for it (the same way it already does for `/chat/`). The app falls back to HTTP polling if the socket can't connect.
- **Several workers** - live updates fan out across workers through Redis pub/sub (`live:train_updates`), so `gunicorn -w N` is safe.
- Tests: `python3 -m pytest scratch/test_live_features.py -q` (needs local Redis; uses db 15).

## 🤝 Contributing

We welcome contributions to improve train data accuracy! Most contributions will involve updating the `data.json` file.

### 📝 Contributing to data.json

The `data.json` file contains all train and station information. **After each modification, please increment `CURRENT_REVISION` by 1.**

#### Structure Overview

```json
{
  "CURRENT_REVISION": 128,
  "DATA": {
    "sid_to_sname": { "station_id": "station_name" },
    "sid_to_sloc": { "station_id": [latitude, longitude] },
    "train_names": { "train_id": "train_name" },
    "tid_to_stations": { "train_id": [["station_id", stop_type, "time"]] }
  }
}
```

#### 🚉 Adding Stations (sid_to_sname & sid_to_sloc)

**sid_to_sname**: Station ID to Station Name
- **Always append new stations** - don't modify existing IDs
- **Check if station already exists** before adding
- Use lowercase names (e.g., "dhaka", "chittagong")

**sid_to_sloc**: Station ID to Location [latitude, longitude]
- Find accurate coordinates using Google Maps, OpenStreetMap, or other mapping services
- Use decimal degrees format
- Ensure high precision for accuracy

Example:
```json
"sid_to_sname": {
  "Jashore": "Jashore/Jessore"
},
"sid_to_sloc": {
  "Jashore": [23.1634, 89.2182]
}
```

#### 🚆 Adding Trains (train_names)

- Use official train IDs from [Bangladesh Railway eTicket](https://eticket.railway.gov.bd/en)
- Use proper train names as listed on official sources
- Follow existing ID format (3-digit numbers as strings)

Example:
```json
"train_names": {
  "109": "Parabat Express"
}
```

#### 🛤️ Adding Train Routes (tid_to_stations)

Each station entry has three elements: `["station_id", stop_type, "reaching_time"]`

**Stop Types:**
- `1`: Train stops at this station (regular stop)
- `0`: Train passes through but doesn't stop
- `-1`: Important landmark/irregular stop (bridges, junctions, etc.)

**Time Format:** Use 24-hour format "HH:MM"

**Important:** Keep stations in **sequence order** of the train's journey

Example:
```json
"tid_to_stations": {
  "109": [
    ["Dhaka", 1, "18:00"],    // Dhaka - stops
    ["Comilla", 0, "19:30"],    // Comilla - passes through
    ["Chittagong", 1, "21:15"]     // Chittagong - stops
  ]
}
```

### 📋 Contribution Guidelines

1. **Fork** the repository
2. **Create a new branch** for your changes
3. **Update data.json** following the structure above
4. **Increment CURRENT_REVISION** by 1
5. **Test your changes** by running data_validator.py
6. **Location_analysis.py** creates html file to debug geo location.
7. **Submit a pull request** with a clear description

### ✅ Before Contributing

- Verify train information from official Bangladesh Railway sources
- Double-check station coordinates using multiple mapping services
- Ensure station names are consistent with existing naming conventions
- Test that your JSON is valid (use a JSON validator)

### 🔍 Data Sources

- [Bangladesh Railway eTicket](https://eticket.railway.gov.bd/en) - Official train schedules
- [Google Maps](https://maps.google.com) - Station coordinates
- [OpenStreetMap](https://www.openstreetmap.org) - Alternative mapping source

## 📄 License

This project is open source. Please ensure all contributed data is accurate and from reliable sources.

## 🚀 Live Server

The production server runs at: `findmytrain.freeddns.org`, [Android App](https://play.google.com/store/apps/details?id=com.mytraintrackerbd.app) 

## 📞 Support

For questions or issues, please open a GitHub issue or contribute to the project!
