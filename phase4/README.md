# Phase 4: Training & Evaluation Pipeline

This phase covers building the training loop that optimizes the composite topological loss function, evaluating the imputer against baselines, and exporting the results into our Dual formats (Parquet + JSON).

---

## 1. File Setup Guide

Before adding any code, set up the following files inside your workspace structure:

1. **Add Dataset class to `Football/halfspace/data.py`:**
   Open the `data.py` file you created in Phase 1 and append the `SparseTrackingDataset` class at the bottom.
2. **Create the training file:**
   Create a new file named `train.py` inside the `halfspace` folder (`Football/halfspace/train.py`). This will contain the model training logic.
3. **Create the exporter file:**
   Create a new file named `exporter.py` inside the `halfspace` folder (`Football/halfspace/exporter.py`). This will contain the Parquet and JSON serialization code.
4. **Update the package initialization:**
   We will update `Football/halfspace/__init__.py` at the end of this phase to expose these classes.

---

## 2. Complete Phase 4 Code

### 2.1 Dataset Helper Addition (`Football/halfspace/data.py`)

Open `Football/halfspace/data.py` and **append** the following code to the bottom of the file (below the `denormalize_coordinates` function):

```python
from torch.utils.data import Dataset
from halfspace.masking import camera_aware_mask

class SparseTrackingDataset(Dataset):
    """
    PyTorch Dataset for Sparse Trajectory Imputation.
    Extracts sliding training windows (e.g. 30s) and generates:
    - Ground-truth trajectory coordinates (Batch, Time, Players, 2)
    - Masked sparse trajectories as inputs (Batch, Time, Players, 3) where elements are [x, y, observed_flag]
    """
    def __init__(self, full_trajectories: np.ndarray, ball_trajectories: np.ndarray, 
                 window_size: int = 30, step_size: int = 10, fps: float = 2.0):
        """
        Args:
            full_trajectories: True coordinates, shape (Time, N_players, 2)
            ball_trajectories: True ball coordinates, shape (Time, 2)
            window_size: Time window in seconds (e.g., 30s)
            step_size: Sliding window step size in seconds
            fps: Resampled frames per second of the dataset
        """
        self.fps = fps
        self.w_frames = int(window_size * fps)
        self.s_frames = int(step_size * fps)
        
        # Apply camera masking on the raw trajectories (before normalization)
        self.masked_coords, self.mask_flags = camera_aware_mask(full_trajectories, ball_trajectories)
        
        # Normalize coordinates to [-1, 1]
        # These normalize_coordinates helper functions are already present in data.py (from Phase 1)
        self.gt_coords_norm = normalize_coordinates(full_trajectories, 105.0, 68.0)
        self.masked_coords_norm = normalize_coordinates(self.masked_coords, 105.0, 68.0)
        
        # Build samples
        self.samples = []
        total_frames = len(full_trajectories)
        
        for start in range(0, total_frames - self.w_frames + 1, self.s_frames):
            end = start + self.w_frames
            
            # Ground truth
            gt_win = self.gt_coords_norm[start:end]
            
            # Masked input coordinates and flags
            mask_coords_win = self.masked_coords_norm[start:end]
            mask_flags_win = self.mask_flags[start:end]
            
            # Pack input features: [x_masked, y_masked, observed_flag]
            input_win = np.zeros((self.w_frames, gt_win.shape[1], 3), dtype=np.float32)
            input_win[..., :2] = mask_coords_win
            input_win[..., 2] = mask_flags_win
            
            self.samples.append({
                'input': torch.tensor(input_win, dtype=torch.float32),
                'target': torch.tensor(gt_win, dtype=torch.float32),
                'mask': torch.tensor(mask_flags_win, dtype=torch.float32)
            })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]
```

### 2.2 Training Loop (`Football/halfspace/train.py`)

Add the following code to `Football/halfspace/train.py`:

```python
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from halfspace.losses import CentroidLoss, ShapeLoss, SoftRadialHullLoss
from halfspace.model import BiGRUImputer

def train_imputer(dataset, num_epochs: int = 20, batch_size: int = 16, lr: float = 1e-3) -> BiGRUImputer:
    """
    Trains the BiGRUImputer using the joint topological loss.
    """
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Initialize model
    model = BiGRUImputer(num_players=22)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Initialize losses
    mse_criterion = torch.nn.MSELoss()
    centroid_criterion = CentroidLoss()
    shape_criterion = ShapeLoss()
    hull_criterion = SoftRadialHullLoss(num_sectors=8, epsilon=1.0)
    
    # Loss weights (balancing pointwise coordinates and shape)
    w_mse = 1.0
    w_centroid = 0.5
    w_shape = 0.3
    w_boundary = 0.2
    
    model.train()
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for batch in loader:
            inputs = batch['input']   # (Batch, Time, N, 3)
            targets = batch['target'] # (Batch, Time, N, 2)
            masks = batch['mask']     # (Batch, Time, N)
            
            optimizer.zero_grad()
            
            # Predict coordinates
            predictions = model(inputs) # (Batch, Time, N, 2)
            
            # 1. Coordinate Identity Loss (only compute where observed in inputs)
            # Reshape masks for coordinate multiplication: (B, T, N, 1)
            masks_expanded = masks.unsqueeze(-1)
            loss_mse = mse_criterion(predictions * masks_expanded, targets * masks_expanded)
            
            # 2. Centroid Consistency Loss
            loss_centroid = centroid_criterion(predictions, targets, mask=masks)
            
            # 3. Shape Variance Loss
            loss_shape = shape_criterion(predictions, targets, mask=masks)
            
            # 4. Boundary Convex Hull Loss
            loss_boundary = hull_criterion(predictions, targets, mask=masks)
            
            # Joint objective
            total_loss = (w_mse * loss_mse + 
                          w_centroid * loss_centroid + 
                          w_shape * loss_shape + 
                          w_boundary * loss_boundary)
            
            total_loss.backward()
            optimizer.step()
            
            epoch_loss += total_loss.item()
            
        print(f"Epoch {epoch+1}/{num_epochs} - Loss: {epoch_loss / len(loader):.4f}")
        
    return model
```

### 2.3 Dual Exporter (`Football/halfspace/exporter.py`)

Add the following code to `Football/halfspace/exporter.py`:

```python
import json
import pandas as pd
import numpy as np

def export_reconstructions(frames_metadata: list, true_coords: np.ndarray, 
                           pred_coords: np.ndarray, ball_coords: np.ndarray,
                           decoded_states: list, pitch_length: float, pitch_width: float,
                           parquet_path: str, json_path: str):
    """
    Exports reconstructed trajectories in Dual Formats: Parquet and frame-centric JSON.
    
    Args:
        frames_metadata: List of dicts containing timestamps and clocks.
        true_coords: True coordinate arrays, shape (Time, 22, 2) in meters.
        pred_coords: Reconstructed coordinate arrays, shape (Time, 22, 2) in meters.
        ball_coords: Ball coordinates, shape (Time, 2) in meters.
        decoded_states: List of state strings corresponding to timestamps.
        pitch_length/pitch_width: Dimensions in meters.
        parquet_path: Filepath for flat tabular export.
        json_path: Filepath for visualization JSON.
    """
    T = len(frames_metadata)
    
    # --- 1. Tabular Parquet Export ---
    records = []
    for t in range(T):
        ts = frames_metadata[t]["timestamp"]
        clock = frames_metadata[t]["clock"]
        state = decoded_states[t]
        
        # Save ball details
        records.append({
            "timestamp": ts, "clock": clock, "object_id": "ball", "team": "ball",
            "x_pred": ball_coords[t, 0], "y_pred": ball_coords[t, 1],
            "x_true": ball_coords[t, 0], "y_true": ball_coords[t, 1],
            "is_observed": 1.0, "tactical_state": state
        })
        
        # Save players (Home indices 0-10, Away indices 11-21)
        for i in range(22):
            team_lbl = "home" if i < 11 else "away"
            idx_in_team = i if i < 11 else i - 11
            
            records.append({
                "timestamp": ts,
                "clock": clock,
                "object_id": f"player_{team_lbl}_{idx_in_team:02d}",
                "team": team_lbl,
                "x_pred": pred_coords[t, idx_in_team if i < 11 else idx_in_team, 0],
                "y_pred": pred_coords[t, idx_in_team if i < 11 else idx_in_team, 1],
                "x_true": true_coords[t, idx_in_team if i < 11 else idx_in_team, 0],
                "y_true": true_coords[t, idx_in_team if i < 11 else idx_in_team, 1],
                "is_observed": 0.0, # Will be set to true if matching observation mask in data
                "tactical_state": state
            })
            
    df = pd.DataFrame(records)
    df.to_parquet(parquet_path, index=False)
    
    # --- 2. Hierarchical JSON Export (1Hz/2Hz Simulation) ---
    json_frames = []
    for t in range(T):
        ts = frames_metadata[t]["timestamp"]
        clock = frames_metadata[t]["clock"]
        
        # Home team coordinates (players 0 to 10)
        home_players = pred_coords[t, :11].tolist()
        # Away team coordinates (players 11 to 21)
        away_players = pred_coords[t, 11:22].tolist() if len(pred_coords[t]) > 11 else pred_coords[t, :11].tolist()
        
        json_frames.append({
            "timestamp_ms": int(ts * 1000),
            "clock": clock,
            "decoded_state": decoded_states[t],
            "ball": ball_coords[t].tolist(),
            "home_players": home_players,
            "away_players": away_players
        })
        
    export_payload = {
        "match_id": "match_simulation_01",
        "meta": {
            "pitch_length": pitch_length,
            "pitch_width": pitch_width,
            "total_frames": T
        },
        "frames": json_frames
    }
    
    with open(json_path, 'w') as f:
        json.dump(export_payload, f, indent=2)
```

### 2.4 Wiring Exporters (`Football/halfspace/__init__.py`)

Open `Football/halfspace/__init__.py` and add the following package exposures:

```python
from .data import load_skillcorner_match, normalize_coordinates, denormalize_coordinates, SparseTrackingDataset
from .masking import camera_aware_mask
from .losses import CentroidLoss, ShapeLoss, SoftRadialHullLoss
from .model import BiGRUImputer
from .decoder import CRHMDecoder
from .train import train_imputer
from .exporter import export_reconstructions
```

---

## 3. Snippet-by-Snippet Explanation

### 3.1 Multi-Task Loss Weighting
```python
            # Joint objective
            total_loss = (w_mse * loss_mse + 
                          w_centroid * loss_centroid + 
                          w_shape * loss_shape + 
                          w_boundary * loss_boundary)
```
* **Explanation:** Combines standard MSE with our three topological loss terms. By weighting them ($w_1, w_2, w_3, w_4$), we ensure that the optimization updates the neural network's spatial weights to preserve the overall team boundaries and dispersion characteristics while retaining coordinate precision.

### 3.2 Tabular Parquet Export
```python
    df = pd.DataFrame(records)
    df.to_parquet(parquet_path, index=False)
```
* **Explanation:** Serializes the complete coordinates database into Apache Parquet format. Parquet utilizes columnar compression, which minimizes file storage size and allows for fast analytical queries inside Python/Pandas environments during model evaluation.

### 3.3 Frame-Centric JSON Export
```python
        json_frames.append({
            "timestamp_ms": int(ts * 1000),
            "clock": clock,
            "decoded_state": decoded_states[t],
            "ball": ball_coords[t].tolist(),
            "home_players": home_players,
            "away_players": away_players
        })
```
* **Explanation:** Formatted player positions into structured frame-by-frame lists. This matches the browser timeline rendering scheme directly, allowing the visualizer to load coordinates by simply fetching the array index corresponding to the match clock.

---

## 4. Testing Suggestions

To check your training and exporter components:
1. **Training Convergence Test:** Instantiate your dataset with 5 dummy samples. Set the learning rate high (e.g. `1e-2`) and train for 10 epochs. Assert that the printed joint loss values show a steady decreasing trend.
2. **Backward Gradient Flow Verification:** Verify that omitting coordinate MSE ($w_{\text{mse}} = 0$) and training solely on shape/centroid/boundary losses still produces gradients and updates the network coordinates.
3. **Parquet Integration Check:** Load the exported Parquet file using `pd.read_parquet` and check that the row counts match:
   $$\text{Total Rows} = \text{Frames} \times 23 \quad (\text{22 players} + \text{1 ball})$$
4. **JSON Structural Validation:** Load the generated JSON file and assert that `meta` contains the correct values, and each frame in `frames` contains `home_players` (list of size 11) and `away_players` (list of size 11).
