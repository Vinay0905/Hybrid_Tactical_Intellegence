# HalfSpace: Product Requirement Document (PRD)

**Project Core:** Topology-Preserving Trajectory Reconstruction & Downstream Tactical Phase Segmentation

**Development Paradigm:** 0₹ / Open-Source & Open-Data

---

## 1. Executive Summary & Intent

### 1.1 Objective

**HalfSpace** is a spatiotemporal machine learning framework designed to reconstruct continuous multi-player tracking data from highly sparse, event-driven observation windows (such as StatsBomb 360 freeze-frames) and classify dynamic team-level tactical states.

### 1.2 The Core Thesis

Standard spatial imputation models minimize pointwise Euclidean displacement, which leads to *"mean collapse"* (where predicted coordinates cluster artificially towards the centroid due to spatial uncertainty). HalfSpace optimizes a joint multi-task loss function that penalizes deformations of the team's collective shape. This ensures that reconstructed player positions retain their tactical topology (defensive line height, team width, compactness, and boundary structures) to support robust downstream coaching insights.

---

## 2. Mathematical Specification & Loss Formulations

The optimization objective minimizes a composite loss function over the predicted player coordinate matrix $\hat{\mathbf{P}}_t \in \mathbb{R}^{N \times 2}$ relative to the ground-truth coordinate matrix $\mathbf{P}_t \in \mathbb{R}^{N \times 2}$ for $N$ outfield players at frame $t$:

$$\mathcal{L}_{\text{total}} = w_1 \mathcal{L}_{\text{MSE}} + w_2 \mathcal{L}_{\text{centroid}} + w_3 \mathcal{L}_{\text{shape}} + w_4 \mathcal{L}_{\text{boundary}}$$

### 2.1 Coordinate Identity Loss ($\mathcal{L}_{\text{MSE}}$)

Standard Mean Squared Error forces reconstructed positions to align with observed targets:

$$\mathcal{L}_{\text{MSE}} = \frac{1}{N} \sum_{i=1}^N ||\mathbf{p}_{i,t} - \hat{\mathbf{p}}_{i,t}||^2$$

### 2.2 Centroid Consistency Loss ($\mathcal{L}_{\text{centroid}}$)

Prevents global drift of the reconstructed team's center of mass:

$$\mathcal{L}_{\text{centroid}} = ||\mathbf{c}_t - \hat{\mathbf{c}}_t||^2 \quad \text{where} \quad \mathbf{c}_t = \frac{1}{N} \sum_{i=1}^N \mathbf{p}_{i,t}$$

### 2.3 Spatial Dispersion & Aspect Ratio Loss ($\mathcal{L}_{\text{shape}}$)

Preserves lateral team width and longitudinal block height without relying on static player identity indexing or alignment. We compute the eigenvalues of the spatial covariance matrix $\mathbf{\Sigma}_t \in \mathbb{R}^{2 \times 2}$:

$$\mathbf{\Sigma}_t = \frac{1}{N} (\mathbf{P}_t - \mathbf{1}\mathbf{c}_t^T)^T (\mathbf{P}_t - \mathbf{1}\mathbf{c}_t^T)$$

Let $\lambda_{1,t}$ and $\lambda_{2,t}$ (where $\lambda_{1,t} \ge \lambda_{2,t}$) be the eigenvalues of $\mathbf{\Sigma}_t$, representing the variance along the principal axes of the team's spatial footprint. The shape loss is formulated as:

$$\mathcal{L}_{\text{shape}} = (\lambda_{1,t} - \hat{\lambda}_{1,t})^2 + (\lambda_{2,t} - \hat{\lambda}_{2,t})^2$$

### 2.4 Differentiable Soft Convex Hull Loss ($\mathcal{L}_{\text{boundary}}$)

Standard convex hull algorithms use discrete vertex selectors that are non-differentiable. We introduce a differentiable alternative using **Softmax Radial Projections**.

Let $\mathbf{u}_{\theta} = [\cos\theta, \sin\theta]^T$ represent a unit direction vector for $M$ discrete angular sectors (e.g., $M = 8$ directions representing $\theta \in \{0, \frac{\pi}{4}, \frac{\pi}{2}, \dots\}$). The relative coordinate vector of player $i$ from the centroid is $\mathbf{r}_{i,t} = \mathbf{p}_{i,t} - \mathbf{c}_t$. The soft maximum projection distance $\tilde{r}_{\theta,t}$ in direction $\theta$ is defined using a log-sum-exp operator with a smooth scale parameter $\epsilon > 0$:

$$\tilde{r}_{\theta,t} = \epsilon \log \sum_{i=1}^N \exp\left( \frac{\mathbf{r}_{i,t}^T \mathbf{u}_{\theta}}{\epsilon} \right)$$

$$\mathcal{L}_{\text{boundary}} = \frac{1}{M} \sum_{\theta=1}^M \left( \tilde{r}_{\theta,t} - \hat{\tilde{r}}_{\theta,t} \right)^2$$

---

## 3. Mathematical Proof: Gradient Legitimacy & Differentiability

To validate that $\mathcal{L}_{\text{boundary}}$ is a mathematically sound, differentiable objective for backpropagation, we derive its analytical gradient with respect to any individual player coordinate vector $\mathbf{p}_{i,t}$.

Let the scalar projection of player $i$ relative to the centroid in direction $\mathbf{u}_{\theta}$ be $f_i = \mathbf{r}_i^T \mathbf{u}_{\theta} = (\mathbf{p}_i - \mathbf{c})^T \mathbf{u}_{\theta}$. We seek to evaluate:

$$\frac{\partial \tilde{r}_{\theta}}{\partial \mathbf{p}_i}$$

### 3.1 Chain Rule Application

By applying the multivariate chain rule across all projection states:

$$\frac{\partial \tilde{r}_{\theta}}{\partial \mathbf{p}_i} = \sum_{j=1}^N \frac{\partial \tilde{r}_{\theta}}{\partial f_j} \frac{\partial f_j}{\partial \mathbf{p}_i}$$

### 3.2 Evaluation of the Log-Sum-Exp Derivative

The derivative of the log-sum-exp function with respect to its $j$-th exponent input is:

$$\frac{\partial \tilde{r}_{\theta}}{\partial f_j} = \frac{\partial}{\partial f_j} \left[ \epsilon \log \sum_{k=1}^N \exp\left( \frac{f_k}{\epsilon} \right) \right] = \frac{\exp\left( \frac{f_j}{\epsilon} \right)}{\sum_{k=1}^N \exp\left( \frac{f_k}{\epsilon} \right)} = w_j$$

where $w_j \in (0, 1)$ represents the standard softmax weight of player $j$'s projection in direction $\theta$.

### 3.3 Evaluation of the Projection Gradient

Because the relative projection is defined as $f_j = \left( \mathbf{p}_j - \frac{1}{N}\sum_{m=1}^N \mathbf{p}_m \right)^T \mathbf{u}_{\theta}$, its derivative with respect to $\mathbf{p}_i$ is:

$$\frac{\partial f_j}{\partial \mathbf{p}_i} = \mathbf{u}_{\theta} \left( \delta_{ij} - \frac{1}{N} \right)$$

where $\delta_{ij}$ represents the Kronecker delta ($\delta_{ij} = 1$ if $i = j$, and $0$ otherwise).

### 3.4 Analytical Vector Gradient

Substituting these components back into the chain rule formulation:

$$\frac{\partial \tilde{r}_{\theta}}{\partial \mathbf{p}_i} = \sum_{j=1}^N w_j \mathbf{u}_{\theta} \left( \delta_{ij} - \frac{1}{N} \right) = \mathbf{u}_{\theta} \left( w_i - \frac{1}{N}\sum_{j=1}^N w_j \right)$$

Since $\sum_{j=1}^N w_j = 1$ by definition of the softmax distribution, the final gradient simplifies to:

$$\frac{\partial \tilde{r}_{\theta}}{\partial \mathbf{p}_i} = \mathbf{u}_{\theta} \left( w_i - \frac{1}{N} \right)$$

### 3.5 Core Proof Validation Findings

* **Continuity:** The vector gradient $\frac{\partial \tilde{r}_{\theta}}{\partial \mathbf{p}_i}$ is infinitely continuous for all $\epsilon > 0$. It is completely free of the non-differentiable jumps, singularities, or sorting operations that break traditional convex hull formulations.

* **Physical Interpretation:** This gradient applies a localized physical force. The outward pulling force applied to player $i$ in direction $\theta$ is scaled by $w_i$ (representing how close they are to being the outermost boundary player in that sector). This force is balanced by a constant global contraction pull towards the centroid scaled by $-\frac{1}{N}$. This formulation is highly robust and structurally stable during neural network training.

---

## 4. Technical Pipeline Architecture

The end-to-end data processing and model pipeline executes across three primary stages under a strict 0₹ open data architecture:

```
+──────────────────────────+       +──────────────────────────+
│  SkillCorner Open Data   │       │   StatsBomb 360 Open     │
│  (Complete 10Hz Tracking)│       │ (Event-Driven Freeze)    │
+────────────┬─────────────+       +────────────┬─────────────+
             │                                  │
             ▼                                  ▼
+──────────────────────────+                    │
│ Camera-Aware Masking     │                    │
│ (Hides Off-Camera/Event) │                    │
+────────────┬─────────────+                    │
             │                                  │
             └─────────────────┬────────────────┘
                                │
                                ▼
            +────────────────────────────────────+
            │         HalfSpace-Imputer          │
            │  Set Transformer GNN Autoencoder   │
            │  Optimizing Topological Joint Loss │
            +──────────────────┬─────────────────+
                                │
                                ▼
            +────────────────────────────────────+
            │      Constraint-Regularized        │
            │           HMM Decoder              │
            │ (Downstream Tactical Segmentation) │
            +────────────────────────────────────+
```

### 4.1 Input Specification & Ingestion

| Dataset Source | Modality | Sample Rate | Licensing/Cost |
| --- | --- | --- | --- |
| StatsBomb Open Data | Event JSON with 360 freeze-frames | Event-based sparse coordinates | Free Open-Source / 0₹ |
| SkillCorner Open Data | Continuous broadcast-derived tracking | 10Hz multi-agent coordinates | Free Open-Source / 0₹ |

### 4.2 Camera-Aware Masking & Simulation

To evaluate the imputer's performance against ground-truth tracking, complete tracking data is artificially masked to simulate real broadcast constraints:

1. Ingest continuous tracking coordinates from the SkillCorner Open Dataset.
2. Isolate the ball coordinate $(x_{\text{ball},t}, y_{\text{ball},t})$ at each timestamp.
3. Apply a $60 \times 40$ meter bounding box centered on $(x_{\text{ball},t}, y_{\text{ball},t})$ to simulate a standard broadcast camera view.
4. Set the coordinates of any player outside this bounding box to `NaN`.
5. Mask all frames between major on-ball events to replicate StatsBomb 360 sparsity.

### 4.3 Reconstruction Engine (HalfSpace-Imputer)

* **Backbone:** A GNN-based Set Transformer Autoencoder that models spatiotemporal multi-agent interactions in an equivariant and permutation-invariant manner.
* **Loss Operator:** Employs the topological joint loss function ($\mathcal{L}_{\text{total}}$) to reconstruct the masked coordinates back into a continuous 2D pitch field.

### 4.4 Downstream Tactical Segmenter (CR-HMM)

A first-order, non-homogeneous Hidden Markov Model decodes the latent sequence of tactical states $z_t \in \{\text{High Press}, \text{Mid-Block}, \text{Low-Block}, \text{Transition}\}$:

* **Emissions:** Computed from reconstructed features $\mathbf{x}_t = [\text{line\_height}_t, \text{compactness}_t, \text{width}_t]^T$.
* **Viterbi Decoder with Analyst Constraints:** The transition matrix incorporates analyst-defined boundary thresholds to ensure logical, domain-aligned sequence outputs:

$$\theta_t(z_t, z_{t-1}) = \log A_{z_{t-1}, z_t} - \gamma \cdot \mathbb{I} \left( \mathbf{A}_{z_t} [h_t, w_t]^T > \mathbf{b}_{z_t} \right)$$

---

## 5. Benchmarking & Empirical Validation Protocol

To satisfy rigorous academic review, the model must be validated using the following protocol:

### 5.1 Baselines

Evaluate the HalfSpace-Imputer against:

1. **ARMAX-Interpolation** (Matthew Penn's open-source tracking interpolator).
2. **Graph Imputer** (SOTA non-autoregressive GNN imputer).

### 5.2 Metrics for Evaluation

Reconstruction accuracy must be reported using both coordinate and topological metrics:

| Metric Group | Metric Name | Definition |
| --- | --- | --- |
| Pointwise Spatial | Average Displacement Error (ADE) | Mean Euclidean distance between predicted and true coordinates. |
| Pointwise Spatial | Final Displacement Error (FDE) | Euclidean distance at the end of missing sequences. |
| Global Structural | Centroid Drift | Average error in team centroid location. |
| Global Structural | Compactness Error | Mean absolute difference in average radial player distance. |
| Global Structural | Boundary Deformation | Absolute difference in area of predicted and true convex hulls. |

### 5.3 Downstream Tactical Classification Ablation

Beyond reconstruction-only metrics, the central thesis of HalfSpace — that shape-preservation improves *tactical utility* even when it does not improve pointwise accuracy — must be validated with a dedicated ablation:

1. Train two otherwise-identical imputers: one with $\mathcal{L}_{\text{total}}$ (full topological joint loss) and one with $\mathcal{L}_{\text{MSE}}$ only (coordinate-only baseline).
2. Feed each model's reconstructed trajectories into the identical, frozen CR-HMM decoder from Section 4.4.
3. Compare downstream Macro-F1 and segment stability (number of spurious state switches per match) between the two reconstructions, holding the decoder constant.
4. Report results even where ADE/FDE favor the coordinate-only baseline — the ablation's purpose is precisely to show that better pointwise accuracy does not imply better tactical classification.

> **Note:** This ablation is a required validation step, not an optional extension. It is the empirical proof of Section 1.2's core thesis and should be treated as a release-blocking deliverable alongside the Section 5.2 benchmark table.

---

## 6. Open-Source Benchmark Design (SB360-Tactical-Bench)

> **Flag — proposed design, not yet executed.** This section was introduced during PRD drafting and specifies a *target* annotation protocol. Every quantity below (match count, annotator count, split sizes) is a planning placeholder. None of this has been built, recruited, or run yet — treat these numbers as decisions still to be made and confirmed, not as completed work.

To establish a reproducible evaluation framework, HalfSpace proposes releasing a standardized, open-source benchmark dataset curated from the public StatsBomb 360 collection.

### 6.1 Proposed Dataset Composition & Target Split

A draft proposal is to curate roughly 20 matches from top-tier competitions (e.g., La Liga and English Premier League), split as:

* **Train Set:** ~12 matches (fully annotated events and freeze-frames).
* **Validation Set:** ~3 matches (used for hyperparameter and threshold selection).
* **Test Set:** ~5 matches (held out for final trajectory and phase evaluation).

These counts should be revisited once actual annotation effort is scoped — given the 0₹/solo-developer constraint, a smaller pilot (e.g., 3–5 matches) is a more realistic first milestone than the full 20-match set.

### 6.2 Proposed Ground-Truth Annotation Protocol

A rigorous protocol would have every defensive sequence in the benchmark matches manually annotated frame-by-frame by multiple independent reviewers, to determine the timestamps of the four macro-tactical states, with labels consolidated and inter-rater agreement reported (e.g., via Fleiss' Kappa).

> **Reality check:** Recruiting multiple independent football analysts is a real resourcing constraint outside the 0₹ budget. A realistic first version of this protocol is self-annotation by the project author against the FIFA Football Language phase definitions, with inter-rater agreement deferred until/unless additional annotators are recruited (e.g., via course peers).

### 6.3 Standardized Data Schema

The annotated dataset would be serialized into a single, unified JSON directory structure:

```json
{
  "match_id": 12345,
  "timestamps": [
    {
      "timestamp_ms": 1000,
      "ball_location": [34.5, 42.1],
      "tactical_phase_gt": "Mid-Block",
      "visible_freeze_frame": true,
      "players": [
        {
          "player_id": "player_01",
          "team": "home",
          "is_goalkeeper": false,
          "true_coords": [31.2, 22.4],
          "observed": true
        }
      ]
    }
  ]
}
```

---

## 7. Interactive Visualization and Demo Layer

### 7.1 Concept

To translate abstract mathematical formulations into an intuitive, visually inspectable interface for non-technical domain stakeholders, HalfSpace implements a lightweight, interactive two-panel widget.

```
┌────────────────────────────────────────────────────────────────────────┐
│                              TOP PANEL                                 │
│                  2D Pitch View (SVG or HTML5 Canvas)                   │
│                                                                        │
│                       ● Home   ● Away   ★ Ball                         │
│                                                                        │
├────────────────────────────────────────────────────────────────────────┤
│                            BOTTOM PANEL                                │
│                     Dynamic Tactical Timeline                          │
│                                                                        │
│ [High Press] [    Mid-Block    ] [ Low-Block ] [   Transition    ]     │
│ 0m           15m                45m           72m                  90m │
└────────────────────────────────────────────────────────────────────────┘
```

The panel architecture is split into two connected views:

* **Top Panel (2D Pitch View):** A structurally accurate, normalized $120 \times 80$ meter pitch projection. It renders all 22 outfield players and the ball as coordinates at the active timestamp, colored distinctly by team alignment.
* **Bottom Panel (Tactical Timeline):** A horizontal, time-linear progress strip spanning the full match duration (0 to 90+ minutes). It is split into contiguous, color-coded blocks where each color corresponds to one of the decoded tactical states from the CR-HMM.

### 7.2 Core Interaction

* **Scrub & Sync:** Clicking, dragging, or scrubbing the slider along the horizontal timeline immediately triggers a state change in the 2D Pitch View. It updates player and ball positions to match the selected frame.
* **Phase Badge Sync:** Near the pitch canvas, a dedicated dynamic badge updates in real-time to display the decoded tactical state (e.g., `High Press`, `Mid-Block`, `Low-Block`, `Transition`), directly demonstrating the structural connection between macro sequence classification and micro spatial coordinates.

### 7.3 Scientific-Honesty UI Requirements

Because StatsBomb 360 data only contains positions at event freeze-frames (leaving the large majority of match frames without raw coordinates), the UI must not mislead the user into assuming continuous tracking data was originally observed.

```
                     Comparison Toggle States (UI View):

      [RAW SPARSE MODE]  ───> Displays ONLY raw freeze-frame points
                              (Players literally vanish in gap frames)

      [HALFSPACE MODE]   ───> Displays reconstructed continuous paths
                              (Smooth trajectories with confidence rings)
```

To maintain academic and professional transparency, the interface enforces the following mechanisms:

* **Frame Classification Indicator:** A prominent, color-coded HUD badge labeled **"Observed Context"** (green, when the selected frame matches an actual event timestamp in the raw dataset) or **"Imputed Approximation"** (amber, when coordinates are model-predicted during tracking gaps).
* **Sparse vs. Reconstructed Toggle:** A master toggle that lets the analyst instantly compare:
  1. *Raw Sparse Snapshots:* Players are rendered *only* at actual event frames; scrubbing to intermediate frames leaves the pitch empty except for the ball, visualizing the raw data's extreme sparsity.
  2. *HalfSpace Reconstruction:* Displays the smooth, continuous trajectories generated by our topology-preserving model.
* **Multi-Model Validation Overlay (SkillCorner Regime):** When visualizing the masked validation matches, the user can toggle an optional comparative overlay:
  * *Green dots:* True ground-truth paths (from unmasked SkillCorner).
  * *Blue dots:* HalfSpace reconstructed paths (retaining team block height, width, and shape).
  * *Red dots:* Pointwise RMSE-only baseline paths (illustrating "mean collapse," where players cluster unnaturally towards the center).

### 7.4 Functional/Technical Specifications

#### 7.4.1 Widget Input Data Schema

The visualization component consumes a standardized, precomputed frame-by-frame payload generated by the Python pipeline:

| Key Name | Data Type | Description |
| --- | --- | --- |
| `timestamp_sec` | `float` | Elapsed match time in seconds from kickoff. |
| `is_observed` | `boolean` | `true` if this timestamp contains a raw StatsBomb 360 freeze-frame. |
| `decoded_state` | `enum` | State classification: `[HIGH_PRESS, MID_BLOCK, LOW_BLOCK, TRANSITION]`. |
| `home_players` | `array` | Coordinates $[x, y]$ for Home players 1 through 11 (and optional baseline/GT sets). |
| `away_players` | `array` | Coordinates $[x, y]$ for Away players 1 through 11 (and optional baseline/GT sets). |
| `ball_coords` | `array` | Pitch coordinates $[x, y]$ of the match ball. |

#### 7.4.2 Timeline Segment Construction via Run-Length Encoding

The horizontal dynamic timeline is constructed by executing a Run-Length Encoding (RLE) parsing pass over the sequence of latent states $Z = \{z_1, z_2, \dots, z_T\}$ produced by the HMM decoder.

$$\text{RLE}(Z) = \left\{ \left( \text{state}_k, \text{start\_time}_k, \text{end\_time}_k \right) \right\}_{k=1}^K$$

Each segment block $k$ is mapped to a percentage width on the progress bar element using its temporal duration:

$$\text{Width}\% = \frac{\text{end\_time}_k - \text{start\_time}_k}{\text{Total Match Duration}} \times 100$$

#### 7.4.3 Seeking and Snapping Behavior

* **Sparse (StatsBomb 360) Mode:** The timeline scrub bar defaults to discrete frame snapping. Dragging the scrub bar pulls the playhead directly to the *nearest observed event frame* where raw validation context is available, preventing the false assumption of measured physical continuity between events.
* **Validation (SkillCorner) Mode:** The scrub bar allows continuous, 10Hz linear frame scrubbing, utilizing smooth interpolation across frames since continuous, high-fidelity ground truth is fully available.

#### 7.4.4 Color Encoding Standards

The 4-state taxonomy and corresponding teams use a highly contrasting color scheme to ensure visual accessibility in both paper figures and web-based demos:

| Tactical State / Team | Hex Code | CSS / Tailwind Class | Visual Role |
| --- | --- | --- | --- |
| **High Press** | `#E74C3C` | `bg-red-600` | High intensity, attacking-third defensive state. |
| **Mid-Block** | `#F39C12` | `bg-amber-500` | Compact, middle-third defensive configuration. |
| **Low-Block** | `#2980B9` | `bg-blue-600` | Deep, defensive penalty-area consolidation. |
| **Transition** | `#8E44AD` | `bg-purple-600` | High-tempo, unorganized possession-change phase. |
| **Home Outfielders** | `#16A085` | `text-teal-600` | Dot rendering color for home team players. |
| **Away Outfielders** | `#2C3E50` | `text-slate-700` | Dot rendering color for away team players. |

#### 7.4.5 Primary Use Cases

* **Academic Paper Figures:** Exporting high-resolution, vector-graphics frame snapshots. These are used to illustrate how pointwise RMSE-only baseline models distort the defensive block (mean collapse) while HalfSpace successfully reconstructs accurate compactness, width, and line height.
* **Portfolio Showcase Demo:** A lightweight, interactive web app (such as Streamlit) designed for non-technical coaches and analysts to evaluate the model's reconstructions of open matches.
* **Internal Debugging Tool:** Evaluates model failure cases during training. It allows developers to scrub directly to frames where shape metrics spike, identifying spatiotemporal edge cases (e.g., set pieces or quick transition turnovers).

#### 7.4.6 Explicit Non-Goals

This interactive visualization layer is strictly a tool for presentation and evaluation. It **does not** generate synthetic player movements and is not a simulation engine. It is designed to visualize real model outputs only, consistent with the project's open-data-only constraint.

To optimize resources, this layer must not consume development time before the core spatiotemporal machine learning model is trained and verified. It is scheduled to be built only after the joint loss functions ($\mathcal{L}_{\text{total}}$) produce validated tracking outputs in the Python backend (i.e., after Milestone M3 in the project roadmap).

---

## 8. Next Steps for Computational Processing

To begin code-level implementation, process the mathematical formulations specified in Section 2, Section 3, and Section 7. Use them to generate:

1. **Differentiable PyTorch Module:** Implement the `SoftRadialHullLoss` class exactly as described in Section 2.4, verifying gradient continuity per the proof in Section 3.
2. **Camera-Aware Masking Script:** Set up a data preparation script to ingest complete tracking files, compute ball coordinates, apply the $60 \times 40$ meter field-of-view bounding box, and mask off-camera players (Section 4.2).
3. **Sequence Classification Pipeline:** Build the non-homogeneous HMM decoder utilizing custom transition potentials to ensure logical, analyst-constrained state transitions (Section 4.4).
4. **Ablation Testing Suite:** Implement the downstream classification ablation specified in Section 5.3 — evaluating HMM tactical classification performance on coordinate-only ($\mathcal{L}_{\text{MSE}}$-only) versus topology-preserving ($\mathcal{L}_{\text{total}}$) reconstructed trajectories, with the decoder held constant across both runs.
