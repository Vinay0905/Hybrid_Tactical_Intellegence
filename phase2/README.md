# Phase 2: Camera-Aware Masking & Custom Losses

This phase introduces broadcast camera emulation (to simulate sparse observations) and implements the differentiable topology-preserving loss functions in PyTorch.

---

## 1. File Setup Guide

Before adding any code, set up the following files inside your workspace structure:

1. **Create the masking file:**
   Create a new file named `masking.py` inside the `halfspace` folder (`Football/halfspace/masking.py`). This will contain the camera viewport clipping algorithm.
2. **Create the losses file:**
   Create a new file named `losses.py` inside the `halfspace` folder (`Football/halfspace/losses.py`). This will contain the PyTorch loss modules.
3. **Update the package initialization:**
   We will update `Football/halfspace/__init__.py` at the end of this phase to expose these functions to the rest of the project.

---

## 2. Complete Phase 2 Code

### 2.1 Camera-Aware Masking (`Football/halfspace/masking.py`)

Add the following code to `Football/halfspace/masking.py`:

```python
import numpy as np

def camera_aware_mask(player_coords: np.ndarray, ball_coords: np.ndarray, 
                      cam_width: float = 60.0, cam_height: float = 40.0) -> tuple:
    """
    Simulates a broadcast camera tracking the ball. Players outside a
    cam_width x cam_height bounding box centered on the ball are masked out.
    
    Args:
        player_coords: Player coordinates, shape (Time, N_players, 2) in meters.
                       Pitch center is (0,0).
        ball_coords: Ball coordinates, shape (Time, 2) in meters.
        cam_width: Horizontal viewport size in meters.
        cam_height: Vertical viewport size in meters.
    Returns:
        masked_coords: Masked coordinates, shape (Time, N_players, 2)
        mask_flags: Binary observation mask, shape (Time, N_players)
    """
    T, N, _ = player_coords.shape
    masked_coords = player_coords.copy()
    mask_flags = np.ones((T, N))
    
    for t in range(T):
        ball_x, ball_y = ball_coords[t]
        
        # Calculate viewport bounds
        min_x = ball_x - cam_width / 2.0
        max_x = ball_x + cam_width / 2.0
        min_y = ball_y - cam_height / 2.0
        max_y = ball_y + cam_height / 2.0
        
        for i in range(N):
            px, py = player_coords[t, i]
            
            # Check if player falls outside camera frame
            if px < min_x or px > max_x or py < min_y or py > max_y:
                masked_coords[t, i] = [0.0, 0.0]  # Zero-out coordinates
                mask_flags[t, i] = 0.0             # Mark as unobserved
                
    return masked_coords, mask_flags
```

### 2.2 Topology-Preserving Losses (`Football/halfspace/losses.py`)

Add the following code to `Football/halfspace/losses.py`:

```python
import torch
import torch.nn as nn
import numpy as np

class CentroidLoss(nn.Module):
    """
    Centroid Consistency Loss:
    Minimizes the squared distance between predicted and target team centroids.
    """
    def __init__(self):
        super(CentroidLoss, self).__init__()

    def forward(self, pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            pred: Reconstructed player coordinates, shape (Batch, Time, N, 2)
            target: True player coordinates, shape (Batch, Time, N, 2)
            mask: Optional observation mask, shape (Batch, Time, N)
        """
        if mask is not None:
            mask_expanded = mask.unsqueeze(-1)
            pred_masked = pred * mask_expanded
            target_masked = target * mask_expanded
            
            sum_mask = mask_expanded.sum(dim=2).clamp(min=1.0)
            c_pred = pred_masked.sum(dim=2) / sum_mask
            c_target = target_masked.sum(dim=2) / sum_mask
        else:
            c_pred = pred.mean(dim=2)
            c_target = target.mean(dim=2)
            
        return torch.mean(torch.sum((c_pred - c_target) ** 2, dim=-1))


class ShapeLoss(nn.Module):
    """
    Spatial Dispersion Loss:
    Penalizes changes in the principal eigenvalues of the spatial covariance matrix
    to preserve lateral block width and longitudinal depth.
    """
    def __init__(self):
        super(ShapeLoss, self).__init__()

    def forward(self, pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        B, T, N, D = pred.shape
        pred_flat = pred.view(-1, N, 2)
        target_flat = target.view(-1, N, 2)
        
        if mask is not None:
            mask_flat = mask.view(-1, N, 1)
            sum_mask = mask_flat.sum(dim=1, keepdim=True).clamp(min=2.0)
            
            c_pred = (pred_flat * mask_flat).sum(dim=1, keepdim=True) / sum_mask
            c_target = (target_flat * mask_flat).sum(dim=1, keepdim=True) / sum_mask
            
            pred_zero = (pred_flat - c_pred) * mask_flat
            target_zero = (target_flat - c_target) * mask_flat
            
            cov_pred = torch.bmm(pred_zero.transpose(1, 2), pred_zero) / (sum_mask - 1)
            cov_target = torch.bmm(target_zero.transpose(1, 2), target_zero) / (sum_mask - 1)
        else:
            c_pred = pred_flat.mean(dim=1, keepdim=True)
            c_target = target_flat.mean(dim=1, keepdim=True)
            
            pred_zero = pred_flat - c_pred
            target_zero = target_flat - c_target
            
            cov_pred = torch.bmm(pred_zero.transpose(1, 2), pred_zero) / (N - 1)
            cov_target = torch.bmm(target_zero.transpose(1, 2), target_zero) / (N - 1)
            
        # Analytical 2D eigenvalue calculation
        def get_eigenvalues(cov: torch.Tensor) -> torch.Tensor:
            trace = cov[:, 0, 0] + cov[:, 1, 1]
            det = cov[:, 0, 0] * cov[:, 1, 1] - cov[:, 0, 1] * cov[:, 1, 0]
            
            sqrt_term = torch.clamp(trace ** 2 - 4 * det, min=0.0).sqrt()
            lambda_1 = (trace + sqrt_term) / 2.0
            lambda_2 = (trace - sqrt_term) / 2.0
            
            return torch.stack([lambda_1, lambda_2], dim=-1)
            
        eig_pred = get_eigenvalues(cov_pred)
        eig_target = get_eigenvalues(cov_target)
        
        return torch.mean(torch.sum((eig_pred - eig_target) ** 2, dim=-1))


class SoftRadialHullLoss(nn.Module):
    """
    Differentiable Convex Hull Loss:
    Checks radial distances in M directions around the centroid using logsumexp.
    """
    def __init__(self, num_sectors: int = 8, epsilon: float = 1.0):
        super(SoftRadialHullLoss, self).__init__()
        self.M = num_sectors
        self.epsilon = epsilon
        
        thetas = np.linspace(0, 2 * np.pi, num_sectors, endpoint=False)
        u_vectors = np.stack([np.cos(thetas), np.sin(thetas)], axis=1)
        self.register_buffer("u_vectors", torch.tensor(u_vectors, dtype=torch.float32))

    def _get_soft_hull_projections(self, coords: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        BT, N, _ = coords.shape
        
        if mask is not None:
            sum_mask = mask.sum(dim=1, keepdim=True).clamp(min=1.0)
            centroid = (coords * mask).sum(dim=1, keepdim=True) / sum_mask
        else:
            centroid = coords.mean(dim=1, keepdim=True)
            
        r = coords - centroid
        projections = torch.matmul(r, self.u_vectors.t()) # (BT, N, M)
        
        if mask is not None:
            projections = projections + (1.0 - mask) * (-1e9)
            
        return self.epsilon * torch.logsumexp(projections / self.epsilon, dim=1)

    def forward(self, pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        B, T, N, D = pred.shape
        pred_flat = pred.view(-1, N, 2)
        target_flat = target.view(-1, N, 2)
        
        mask_flat = mask.unsqueeze(-1).view(-1, N, 1) if mask is not None else None
        
        proj_pred = self._get_soft_hull_projections(pred_flat, mask_flat)
        proj_target = self._get_soft_hull_projections(target_flat, mask_flat)
        
        return torch.mean(torch.sum((proj_pred - proj_target) ** 2, dim=-1))
```

### 2.3 Wiring Exporters (`Football/halfspace/__init__.py`)

Open `Football/halfspace/__init__.py` and add the following package exposures:

```python
from .data import load_skillcorner_match, normalize_coordinates, denormalize_coordinates
from .masking import camera_aware_mask
from .losses import CentroidLoss, ShapeLoss, SoftRadialHullLoss
```

---

## 3. Snippet-by-Snippet Explanation

### 3.1 Bounding Box Viewport Bounds
```python
        min_x = ball_x - cam_width / 2.0
        max_x = ball_x + cam_width / 2.0
        min_y = ball_y - cam_height / 2.0
        max_y = ball_y + cam_height / 2.0
```
* **Explanation:** Centers a camera viewport box on the ball's location at timestamp $t$. Any player whose coordinates fall outside these bounds is designated as unobserved (coordinates mapped to zero, mask flag set to 0.0), replicating real broadcast occlusion patterns.

### 3.2 Analytical 2x2 Covariance Eigenvalues
```python
            trace = cov[:, 0, 0] + cov[:, 1, 1]
            det = cov[:, 0, 0] * cov[:, 1, 1] - cov[:, 0, 1] * cov[:, 1, 0]
            sqrt_term = torch.clamp(trace ** 2 - 4 * det, min=0.0).sqrt()
            lambda_1 = (trace + sqrt_term) / 2.0
            lambda_2 = (trace - sqrt_term) / 2.0
```
* **Explanation:** Since standard eigenvalue solvers (like `torch.linalg.eigvalsh`) can have unstable gradients when eigenvalues are near-identical, we calculate the eigenvalues analytically using the quadratic formula for $2\times2$ covariance matrices. This approach is completely differentiable and robust under backpropagation.

### 3.3 Log-Sum-Exp Convex Hull Approximation
```python
        projections = torch.matmul(r, self.u_vectors.t())
        if mask is not None:
            projections = projections + (1.0 - mask) * (-1e9)
        return self.epsilon * torch.logsumexp(projections / self.epsilon, dim=1)
```
* **Explanation:** Finds the maximum distance of players in $M$ angular directions. Since the maximum operator (`torch.max`) is non-differentiable (only propagating gradients to a single player), we use a smooth Log-Sum-Exp approximation. Unobserved players are penalized with a large negative value ($-10^9$) so they do not impact the soft maximum.

---

## 4. Testing Suggestions

To check your camera mask and loss modules:
1. **Camera Bounding Test:** Create a mock sequence where a player is placed at coordinate $(30, 20)$ and the ball is at $(0, 0)$. Using a $60 \times 40$ camera size, the player should be marked as **observed** (boundary limits are $[-30, 30]$ and $[-20, 20]$). Move the player to $(31, 20)$ and verify that they are marked as **unobserved** (coordinates zeroed and mask flag set to 0.0).
2. **Differentiability Verification:** Write a quick PyTorch script that takes random tensors `pred` with `requires_grad=True`, computes `ShapeLoss` and `SoftRadialHullLoss`, runs `.backward()`, and checks that `pred.grad` is not `None` and does not contain `NaN` or `inf` values.
3. **Centroid Invariance Test:** Verify that translating both `pred` and `target` coordinates by a constant factor does not change the calculated `ShapeLoss` or `SoftRadialHullLoss` value (as they are translation-invariant relative to team centroids).
