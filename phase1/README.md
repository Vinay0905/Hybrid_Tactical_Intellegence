# Phase 1: Data Ingestion & Normalization

This phase covers loading the raw tracking data from SkillCorner's open dataset, parsing the player and ball coordinate sequences, and normalizing the spatial coordinate systems to prepare them for neural network inputs.

---

## 1. Directory & File Setup Guide

Follow these steps to set up your project directory and create the necessary files for Phase 1:

1. **Create the package directory:**
   Create a folder named `halfspace` in the root of your workspace (`Football/halfspace`).
2. **Create the initialization file:**
   Create a file named `__init__.py` inside the `halfspace` folder (`Football/halfspace/__init__.py`). Leave it empty for now; we will populate it as we create more modules.
3. **Create the data loader file:**
   Create a file named `data.py` inside the `halfspace` folder (`Football/halfspace/data.py`). This is where you will add the parser and normalization code below.
4. **Create a local data cache directory:**
   Create a folder named `data_raw` in your workspace root (`Football/data_raw`). Place your SkillCorner metadata JSON (e.g. `match_data.json`) and structured tracking JSON (e.g. `structured_data.json`) inside this directory.

---

## 2. Complete Phase 1 Code

Add the following code to your newly created file `Football/halfspace/data.py`:

```python
import json
import numpy as np

def load_skillcorner_match(match_data_path: str, structured_data_path: str) -> dict:
    """
    Loads and parses SkillCorner open tracking and metadata files.
    
    Args:
        match_data_path: Path to the metadata JSON file containing team, player, and pitch dimensions.
        structured_data_path: Path to the main tracking JSON file containing frame-by-frame coordinates.
    Returns:
        parsed_match: Dictionary containing pitch dimensions, teams, and synchronized coordinates.
    """
    # 1. Load metadata
    with open(match_data_path, 'r') as f:
        meta = json.load(f)
        
    # Get physical pitch dimensions (or fallback to defaults if not specified)
    pitch_length = meta.get("pitch_length", 105.0)
    pitch_width = meta.get("pitch_width", 68.0)
    
    home_team_id = meta["home_team"]["id"]
    away_team_id = meta["away_team"]["id"]
    
    # Map player IDs to squad indexes
    home_players = {p["id"]: i for i, p in enumerate(meta["home_team"]["players"])}
    away_players = {p["id"]: i for i, p in enumerate(meta["away_team"]["players"])}
    
    # 2. Load tracking data
    with open(structured_data_path, 'r') as f:
        tracking = json.load(f)
        
    frames_data = []
    
    # Parse each tracked frame
    for frame in tracking:
        timestamp = frame["timestamp"]
        clock = frame.get("clock", "00:00")
        
        # Initialize coordinate matrices (11 players per team, x/y coords)
        # 0.0 is used as a padding value for missing/occluded players
        home_coords = np.zeros((11, 2))
        away_coords = np.zeros((11, 2))
        ball_coords = np.zeros(2)
        
        # Flags indicating if the player is observed in the current frame
        home_observed = np.zeros(11)
        away_observed = np.zeros(11)
        
        # Read moving objects
        for obj in frame.get("data", []):
            track_id = obj.get("trackable_object")
            coords = obj.get("coordinates")
            if coords is None:
                continue
                
            x, y = coords[0], coords[1]
            
            # Identify object category
            if track_id == "ball":
                ball_coords = np.array([x, y])
            elif obj.get("group") == "home team":
                if track_id in home_players:
                    idx = home_players[track_id]
                    if idx < 11:  # Ensure we only track outfielders/match-squad up to 11
                        home_coords[idx] = [x, y]
                        home_observed[idx] = 1.0
            elif obj.get("group") == "away team":
                if track_id in away_players:
                    idx = away_players[track_id]
                    if idx < 11:
                        away_coords[idx] = [x, y]
                        away_observed[idx] = 1.0
                        
        frames_data.append({
            "timestamp": timestamp,
            "clock": clock,
            "ball": ball_coords,
            "home": home_coords,
            "home_obs": home_observed,
            "away": away_coords,
            "away_obs": away_observed
        })
        
    return {
        "pitch_length": pitch_length,
        "pitch_width": pitch_width,
        "frames": frames_data
    }


def normalize_coordinates(coords: np.ndarray, pitch_length: float, pitch_width: float) -> np.ndarray:
    """
    Normalizes raw metric coordinates to range [-1, 1] relative to pitch dimensions.
    
    Args:
        coords: Numpy array of shape (..., 2) representing [x, y] coordinates.
        pitch_length: Total length of the pitch in meters (horizontal axis).
        pitch_width: Total width of the pitch in meters (vertical axis).
    """
    normalized = coords.copy()
    normalized[..., 0] = normalized[..., 0] / (pitch_length / 2.0)
    normalized[..., 1] = normalized[..., 1] / (pitch_width / 2.0)
    return np.clip(normalized, -1.0, 1.0)


def denormalize_coordinates(coords: np.ndarray, pitch_length: float, pitch_width: float) -> np.ndarray:
    """
    Translates normalized coordinates back into raw metric values for match visualization.
    """
    denormalized = coords.copy()
    denormalized[..., 0] = denormalized[..., 0] * (pitch_length / 2.0)
    denormalized[..., 1] = denormalized[..., 1] * (pitch_width / 2.0)
    return denormalized
```

---

## 3. Snippet-by-Snippet Explanation

### 3.1 Loader Initialization & Metadata Extraction
```python
    with open(match_data_path, 'r') as f:
        meta = json.load(f)
        
    pitch_length = meta.get("pitch_length", 105.0)
    pitch_width = meta.get("pitch_width", 68.0)
```
* **Explanation:** Opens and extracts details from the match metadata. This yields pitch length and width, which are necessary for coordinate calculations. We fall back to standard professional dimensions (105m x 68m) if they are missing.

### 3.2 Squad Index Mapping
```python
    home_players = {p["id"]: i for i, p in enumerate(meta["home_team"]["players"])}
    away_players = {p["id"]: i for i, p in enumerate(meta["away_team"]["players"])}
```
* **Explanation:** Creates lookup dictionaries mapping unique SkillCorner player IDs (which are large strings or arbitrary ints) to squad indexes `0` through `10`. This allows us to consistently index the coordinates in fixed-size arrays ($11 \times 2$) for model training.

### 3.3 Coordinate Matrix Assembly & Object Identification
```python
        home_coords = np.zeros((11, 2))
        away_coords = np.zeros((11, 2))
        ball_coords = np.zeros(2)
        
        home_observed = np.zeros(11)
        away_observed = np.zeros(11)
```
* **Explanation:** For each frame, we prepare empty numpy arrays. Since broadcast tracking only captures visible players in the camera frame, players outside the camera view are initialized to zero coordinates. The `_observed` arrays act as binary masks (1.0 for visible/present, 0.0 for occluded/absent).

```python
            if track_id == "ball":
                ball_coords = np.array([x, y])
            elif obj.get("group") == "home team":
                ...
                home_coords[idx] = [x, y]
                home_observed[idx] = 1.0
```
* **Explanation:** Parses the list of tracked objects in the frame, classifying them as ball, home team player, or away team player. We update the coordinate arrays at their respective squad indexes and mark their observation status.

### 3.4 Pitch Normalization
```python
    normalized[..., 0] = normalized[..., 0] / (pitch_length / 2.0)
    normalized[..., 1] = normalized[..., 1] / (pitch_width / 2.0)
```
* **Explanation:** In SkillCorner tracking data, the center of the pitch is $(0, 0)$. The horizontal coordinate $x$ goes from $-\text{pitch\_length}/2$ to $+\text{pitch\_length}/2$, and vertical coordinate $y$ from $-\text{pitch\_width}/2$ to $+\text{pitch\_width}/2$. Dividing by half-dimensions scales the spatial coordinate ranges to $[-1.0, 1.0]$, which provides maximum numerical stability for neural networks.

---

## 4. Testing Suggestions

To verify that your parser is operating correctly:
1. **Dimension Validation:** Load a sample match file and assert that:
   - The returned `home` coordinates shape is exactly `(Frames, 11, 2)`.
   - The returned `away` coordinates shape is exactly `(Frames, 11, 2)`.
   - The `ball` coordinates shape is `(Frames, 2)`.
2. **Value Bounds Check:** Verify that after running `normalize_coordinates`, all coordinates lie strictly within the range $[-1.0, 1.0]$.
3. **Reversibility Test:** Assert that passing a coordinate array through `normalize_coordinates` and then `denormalize_coordinates` returns the original metric coordinates without any floating-point drift (use `np.allclose`).
4. **Mock Dataset Verification:** Create a mini JSON structure with 2 frames to check that the player ID squad-mapping correctly places coordinates at the expected index positions.
