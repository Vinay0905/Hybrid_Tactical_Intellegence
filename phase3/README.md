# Phase 3: Baseline Imputer & CR-HMM Decoder

This phase covers building the Bidirectional GRU model for trajectory reconstruction, extracting tactical features from the reconstructed paths, and decoding the tactical sequence using a Constraint-Regularized Hidden Markov Model.

---

## 1. File Setup Guide

Before adding any code, set up the following files inside your workspace structure:

1. **Create the model file:**
   Create a new file named `model.py` inside the `halfspace` folder (`Football/halfspace/model.py`). This will contain the recurrent network.
2. **Create the decoder file:**
   Create a new file named `decoder.py` inside the `halfspace` folder (`Football/halfspace/decoder.py`). This will contain the CR-HMM decoder.
3. **Update the package initialization:**
   We will update `Football/halfspace/__init__.py` at the end of this phase to expose these functions to the rest of the project.

---

## 2. Complete Phase 3 Code

### 2.1 Bidirectional GRU Imputer (`Football/halfspace/model.py`)

Add the following code to `Football/halfspace/model.py`:

```python
import torch
import torch.nn as nn

class BiGRUImputer(nn.Module):
    """
    Bidirectional GRU Trajectory Imputer (Section 4.3):
    Reconstructs continuous coordinates from sparse coordinate inputs and observation masks.
    """
    def __init__(self, num_players: int = 22, hidden_dim: int = 128, num_layers: int = 2):
        super(BiGRUImputer, self).__init__()
        self.num_players = num_players
        self.input_dim = num_players * 3  # Features: [x, y, observed_flag] per player
        self.output_dim = num_players * 2 # Predictions: [x, y] coordinates per player
        
        # Encoder projecting flat inputs to hidden dim
        self.encoder = nn.Sequential(
            nn.Linear(self.input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Temporal bidirectional GRU layers
        self.rnn = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True
        )
        
        # Decoder mapping bidirectional states to coordinates
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.output_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor, shape (Batch, Time, N_players, 3)
        Returns:
            pred_coords: Predicted coordinates, shape (Batch, Time, N_players, 2)
        """
        B, T, N, D = x.shape
        assert N == self.num_players, f"Expected {self.num_players} players, got {N}"
        
        # Flatten player dimension: (B, T, N, 3) -> (B, T, N*3)
        x_flat = x.view(B, T, -1)
        
        # Forward pass
        h_enc = self.encoder(x_flat)
        h_rnn, _ = self.rnn(h_enc) # Output shape: (B, T, hidden_dim * 2)
        out_flat = self.decoder(h_rnn)
        
        # Reshape to (B, T, N, 2)
        pred_coords = out_flat.view(B, T, N, 2)
        return pred_coords
```

### 2.2 CR-HMM Decoder (`Football/halfspace/decoder.py`)

Add the following code to `Football/halfspace/decoder.py`:

```python
import numpy as np

class CRHMDecoder:
    """
    Constraint-Regularized HMM Decoder (Section 4.4):
    Decodes tactical phases from team features under physical transition constraints.
    """
    def __init__(self, gamma: float = 10.0):
        self.gamma = gamma
        self.states = ['HIGH_PRESS', 'MID_BLOCK', 'LOW_BLOCK', 'TRANSITION']
        self.num_states = len(self.states)
        self.state_to_idx = {state: i for i, state in enumerate(self.states)}
        self.idx_to_state = {i: state for i, state in enumerate(self.states)}
        
        # Transition log-probabilities: -inf represents physically impossible transitions
        self.log_transition_matrix = np.log([
            # HIGH_PRESS, MID_BLOCK, LOW_BLOCK, TRANSITION
            [0.70,         0.15,      0.00,       0.15], # HIGH_PRESS
            [0.10,         0.70,      0.10,       0.10], # MID_BLOCK
            [0.00,         0.15,      0.70,       0.15], # LOW_BLOCK
            [0.20,         0.20,      0.20,       0.40]  # TRANSITION
        ])
        
        # Initial Gaussian emission properties: [line_height, compactness, width]
        self.emission_means = np.array([
            [0.40,  0.25, 0.50],  # HIGH_PRESS
            [-0.05, 0.18, 0.40],  # MID_BLOCK
            [-0.50, 0.12, 0.35],  # LOW_BLOCK
            [0.00,  0.30, 0.60]   # TRANSITION
        ])
        self.emission_stds = np.array([
            [0.15, 0.08, 0.10],
            [0.15, 0.05, 0.08],
            [0.15, 0.04, 0.08],
            [0.30, 0.12, 0.15]
        ])

    def extract_features(self, coords: np.ndarray, defending_goal_x: float = -1.0) -> np.ndarray:
        """
        Computes team [line_height, compactness, width] from trajectories.
        Args:
            coords: Player coordinates, shape (Time, N_players, 2) scaled in [-1, 1].
            defending_goal_x: Goal x coordinate (-1.0 for left goal, 1.0 for right goal).
        """
        T, N, _ = coords.shape
        features = np.zeros((T, 3))
        
        for t in range(T):
            frame_coords = coords[t]
            centroid = np.mean(frame_coords, axis=0)
            
            # Width (lateral spread)
            y_coords = frame_coords[:, 1]
            width = np.max(y_coords) - np.min(y_coords)
            
            # Compactness (mean distance from centroid)
            compactness = np.mean(np.linalg.norm(frame_coords - centroid, axis=1))
            
            # Defensive line height (average x position of the 3 deepest defenders)
            x_coords = frame_coords[:, 0]
            if defending_goal_x == -1.0:
                deepest_x = np.partition(x_coords, 3)[:3]
                line_height = np.mean(deepest_x)
            else:
                deepest_x = np.partition(x_coords, -3)[-3:]
                line_height = -np.mean(deepest_x)
                
            features[t] = [line_height, compactness, width]
        return features

    def decode(self, features: np.ndarray, analyst_constraints: dict = None) -> list:
        """
        Constrained Viterbi decoder.
        Args:
            features: Features, shape (Time, 3)
            analyst_constraints: Constraint dictionary.
        """
        T = len(features)
        if T == 0:
            return []
            
        # Log emissions
        log_emissions = np.zeros((T, self.num_states))
        for s in range(self.num_states):
            mean = self.emission_means[s]
            std = self.emission_stds[s]
            log_pdf = -0.5 * np.log(2 * np.pi * std**2) - 0.5 * ((features - mean) / std)**2
            log_emissions[:, s] = np.sum(log_pdf, axis=1)
            
        # Apply analyst constraints
        if analyst_constraints is not None:
            for state_name, consts in analyst_constraints.items():
                if state_name in self.state_to_idx:
                    s_idx = self.state_to_idx[state_name]
                    for t in range(T):
                        lh = features[t, 0]
                        penalty = 0.0
                        if 'min_line_height' in consts and lh < consts['min_line_height']:
                            penalty += self.gamma * (consts['min_line_height'] - lh)
                        if 'max_line_height' in consts and lh > consts['max_line_height']:
                            penalty += self.gamma * (lh - consts['max_line_height'])
                        log_emissions[t, s_idx] -= penalty

        # Viterbi execution
        v_table = np.full((T, self.num_states), -np.inf)
        backpointers = np.zeros((T, self.num_states), dtype=int)
        
        v_table[0] = log_emissions[0]
        
        for t in range(1, T):
            for s in range(self.num_states):
                # v_table[t-1, s'] + A[s', s]
                probs = v_table[t-1] + self.log_transition_matrix[:, s]
                best_prev = np.argmax(probs)
                
                v_table[t, s] = probs[best_prev] + log_emissions[t, s]
                backpointers[t, s] = best_prev
                
        # Backtrace
        decoded = []
        curr_state = np.argmax(v_table[-1])
        decoded.append(curr_state)
        
        for t in range(T - 1, 0, -1):
            curr_state = backpointers[t, curr_state]
            decoded.append(curr_state)
            
        decoded.reverse()
        return [self.idx_to_state[idx] for idx in decoded]
```

### 2.3 Wiring Exporters (`Football/halfspace/__init__.py`)

Open `Football/halfspace/__init__.py` and add the following package exposures:

```python
from .data import load_skillcorner_match, normalize_coordinates, denormalize_coordinates
from .masking import camera_aware_mask
from .losses import CentroidLoss, ShapeLoss, SoftRadialHullLoss
from .model import BiGRUImputer
from .decoder import CRHMDecoder
```

---

## 3. Snippet-by-Snippet Explanation

### 3.1 Bidirectional GRU Encoder-Decoder Interface
```python
        h_enc = self.encoder(x_flat)
        h_rnn, _ = self.rnn(h_enc) # Output shape: (B, T, hidden_dim * 2)
```
* **Explanation:** Encodes spatial coordinate sequences before running them through a bidirectional GRU. Since the GRU processes features forwards and backwards simultaneously, it extracts bidirectional context, allowing it to reconstruct missing coordinates at timestamp $t$ by evaluating preceding and succeeding coordinate movements.

### 3.2 Hard Transition Mask
```python
        self.log_transition_matrix = np.log([
            # HIGH_PRESS, MID_BLOCK, LOW_BLOCK, TRANSITION
            [0.70,         0.15,      0.00,       0.15], # HIGH_PRESS
            ...
```
* **Explanation:** Implements hard constraints by setting impossible direct transitions to zero probability ($0.00$). Taking the log shifts this transition potential to $-\infty$, preventing the Viterbi solver from selecting sequences with direct transitions from `LOW_BLOCK` to `HIGH_PRESS` without traversing `TRANSITION` or `MID_BLOCK`.

### 3.3 Analyst Constraint Penalty
```python
                        if 'min_line_height' in consts and lh < consts['min_line_height']:
                            penalty += self.gamma * (consts['min_line_height'] - lh)
```
* **Explanation:** Implements soft regularized constraints. If an analyst defines a minimum line height for `HIGH_PRESS` (e.g. $0.1$), but the team's reconstructed coordinates produce a line height of $-0.2$, the emission log-likelihood is penalized by $\gamma \cdot (0.1 - (-0.2)) = 10 \cdot 0.3 = 3.0$. This heavily discourages, but does not strictly block, the state classification.

---

## 4. Testing Suggestions

To check your imputer model and Viterbi decoder:
1. **Network IO Test:** Create a dummy input tensor of shape `(2, 50, 22, 3)` (representing batch=2, sequence=50 frames, players=22, input features=3). Feed it into `BiGRUImputer` and check that the output shape is exactly `(2, 50, 22, 2)`.
2. **Transition Mask Enforcement Test:** Build a mock input sequence of features where the team line height is deep. Decode the sequence.
   - Now, manually change the first frame to force a `LOW_BLOCK` prediction, and the second frame to force a `HIGH_PRESS` prediction.
   - Run the decoder. Check that the Viterbi output is structurally constrained from outputting a direct transition path (it must insert a `TRANSITION` or `MID_BLOCK` state between them, or flag an error/divergence if log probabilities go to $-\infty$).
3. **Analyst Constraint Test:** Force a constant line height of $-0.6$ (extremely deep defensive block). Verify that passing `analyst_constraints={'HIGH_PRESS': {'min_line_height': 0.0}}` prevents the decoder from outputting `HIGH_PRESS` for that window due to the soft penalties.
