Here’s a single, heavy-duty `PROJECT.md` style document for **HalfSpace** that you can give directly to Antigravity or to ChatGPT to convert into a downloadable file.

***

# HalfSpace – Project Specification / Requirements Document

## 1. Overview

**Name:** HalfSpace  
**Domain:** Football (soccer) analytics  
**Type:** Open-source research system + prototype tactical analysis tool  
**Budget:** Strict 0₹ – only free/open-source software and free/open datasets. [github](https://github.com/statsbomb/open-data)

**Core idea:**  
HalfSpace is a football analytics project that studies how to reconstruct **sparse, event-driven spatial data** (like StatsBomb 360 freeze-frames) in a way that preserves **collective team tactical shape**, and then uses that reconstruction for **tactical state recognition** (e.g., high press / mid-block / low block). [github](https://github.com/statsbomb/open-data)

We are not building a generic dashboard.  
We are building:

1. A **topology-preserving trajectory imputation method** for sparse football data.  
2. A **tactical state segmentation model** built on those reconstructions.  
3. A small **open benchmark** (SB360-Tactical-Bench) that anyone can reproduce using open data. [getgoalsideanalytics](https://www.getgoalsideanalytics.com/high-fat-data-for-low-er-fat-costs/)

***

## 2. Constraints and philosophy

### 2.1 Budget and licensing

- 0₹ budget:
  - No paid APIs.
  - No paid data providers (Wyscout, Opta subscriptions, paid SkillCorner, Hudl Sportscode, Metrica, etc.). [perplexity](https://www.perplexity.ai/search/8ba6bb79-f49d-444a-9999-65bb428be32c)
- Only use:
  - **StatsBomb Open Data** (events + 360 freeze-frames where available). [blogarchive.statsbomb](https://blogarchive.statsbomb.com/news/statsbomb-360-freeze-frame-viewer-a-new-release-in-statsbomb-iq/)
  - **SkillCorner Open Data** (~10 matches with full broadcast tracking). [github](https://github.com/SkillCorner/opendata)
  - Any other genuinely open datasets, if needed. [football-analytics-101.readthedocs](https://football-analytics-101.readthedocs.io/en/latest/data.html)
- Only open-source Python ecosystem (PyTorch, NumPy, SciPy, scikit-learn, etc.).

### 2.2 Non-goals

HalfSpace is **not**:

- a commercial clone of Hudl Sportscode or Wyscout;
- primarily an LLM-based AI coach;
- a generic multi-feature scouting product;
- a pure UI/dashboard project.

Those can be added later as layers. The **scientific core** is sparse-data reconstruction + tactical state modeling.

***

## 3. Research problem and thesis

### 3.1 Problem setting

Most sophisticated football tactical models assume:

- rich, continuous 10–25Hz tracking of all 22 players and the ball. [pmc.ncbi.nlm.nih](https://pmc.ncbi.nlm.nih.gov/articles/PMC12163489/)

In practice, especially for open data and smaller clubs, we often have:

- event data (passes, shots, etc.),
- and **sparse positional snapshots** like **StatsBomb 360 freeze-frames**, which capture player positions only around events, not continuously. [blogarchive.statsbomb](https://blogarchive.statsbomb.com/news/statsbomb-data-case-studies-freeze-frames-and-defender-locations/)

Standard sparse trajectory imputation models optimize a point-wise error objective such as:

$$
\mathcal{L}_{\text{MSE}} = \frac{1}{N T} \sum_{t=1}^T \sum_{i=1}^N \lVert p_{i,t} - \hat{p}_{i,t} \rVert^2
$$

They care about **coordinate accuracy**, not whether the reconstructed team still forms a realistic block/press. [getgoalsideanalytics](https://www.getgoalsideanalytics.com/high-fat-data-for-low-er-fat-costs/)

Consequences:

- mean-collapse toward centroids,
- flattened defensive line height fluctuations,
- distorted compactness,
- loss of pressing and block structure. [getgoalsideanalytics](https://www.getgoalsideanalytics.com/high-fat-data-for-low-er-fat-costs/)

These distortions harm downstream tactical classification (e.g., misclassifying a compact mid-block as a stretched low block).

### 3.2 Thesis

> In sparse-data settings, **tactical usefulness** depends more on preserving **collective team topology** than on minimizing pure coordinate RMSE.

HalfSpace will:

- design a reconstruction model that includes **differentiable shape/topology losses** (centroid, width/height variance, hull proxies), and
- evaluate success not only on RMSE, but on **downstream tactical state recognition** and stability. [getgoalsideanalytics](https://www.getgoalsideanalytics.com/high-fat-data-for-low-er-fat-costs/)

We expect:

- Baseline methods may win on RMSE.
- HalfSpace’s topology-preserving reconstruction should significantly improve:
  - macro tactical state classification F1,
  - stability of segmentation (fewer jittery state jumps). [getgoalsideanalytics](https://www.getgoalsideanalytics.com/high-fat-data-for-low-er-fat-costs/)

***

## 4. Data sources and usage regimes

### 4.1 Open datasets

1. **StatsBomb Open Data** (GitHub):
   - Free event data (passes, shots, etc.). [vrwiki.cs.brown](https://www.vrwiki.cs.brown.edu/scientific-data/statsbomb-open-data)
   - Some competitions with **StatsBomb 360** freeze-frames (positional snapshots at events). [soccermatics.readthedocs](https://soccermatics.readthedocs.io/en/latest/gallery/plot_UsingStatsbomb.html)

2. **SkillCorner Open Data**:
   - ~10 matches with complete broadcast-derived tracking data (10–25Hz) for all players and ball. [thesignificantgame](https://www.thesignificantgame.com/portfolio/first-look-at-skillcorner-s-free-tracking-dataset/)
   - Open repository for research use.

3. Optional additional open sources (later / if needed), such as:
   - other research datasets with tracking or event data. [arxiv](https://arxiv.org/html/2502.02785v2)

### 4.2 Operating regimes

We will use two regimes:

#### Regime 1 – Sparse regime (target application)

- Data: StatsBomb events + StatsBomb 360 freeze-frames. [github](https://github.com/statsbomb/open-data)
- No full tracking ground truth.
- Goal:
  - apply HalfSpace to reconstruct trajectories,
  - extract team-level features,
  - run tactical state classification.

#### Regime 2 – Masked full-tracking regime (for validation)

- Data: SkillCorner Open Data full tracking. [github](https://github.com/SkillCorner/opendata)
- We artificially mask the tracking to mimic 360-style sparsity:
  - keep player positions only at “event-like” frames,
  - drop off-ball frames.
- Goal:
  - reconstruct full trajectories using HalfSpace and baselines,
  - compare reconstructions to true full tracking for:
    - coordinate RMSE,
    - shape metrics,
    - tactical state classification.

This regime directly addresses the standard reviewer critique about “speculative” sparse reconstruction by providing actual ground truth. [getgoalsideanalytics](https://www.getgoalsideanalytics.com/high-fat-data-for-low-er-fat-costs/)

***

## 5. System components and modules

We define logical modules; implementation details can vary as long as these responsibilities are met.

### 5.1 Data ingestion module

Responsibilities:

- Load StatsBomb Open Data:
  - events and 360 freeze-frames (via `statsbombpy` or custom parser). [github](https://github.com/statsbomb/statsbombpy)
- Load SkillCorner Open Data:
  - full tracking from official open repo (via `kloppy` or custom parser). [kloppy.pysport](https://kloppy.pysport.org/user-guide/loading-data/skillcorner/)
- Standardize:
  - pitch coordinates (e.g., 105 x 68),
  - team IDs,
  - player IDs and roles,
  - timestamps (map between event and tracking times).

Outputs:

- Unified data structures:
  - `MatchData` object containing teams, players, events, positions over time.

### 5.2 Masking module (for SkillCorner validation)

Responsibilities:

- Given full tracking (SkillCorner), create a **masked sparse version** approximating StatsBomb 360 distribution:
  - select frames around event times (either from SkillCorner’s own event file or synthetic logic),
  - drop non-selected frames for all players,
  - optionally drop some off-ball players to mimic occlusion. [github](https://github.com/SkillCorner/opendata)

Outputs:

- `SparseTraj` (what HalfSpace sees),
- `FullTraj` (ground truth reference),
- alignment metadata.

### 5.3 HalfSpace-Imputer (trajectory reconstruction model)

This is the **core ML model**.

**Inputs:**

- Sparse observations over time for all players (positions where observed, missing otherwise).
- Context features:
  - event type, minute, score, etc. (optional, but helpful).

**Outputs:**

- Reconstructed coordinates $\hat{p}_{i,t}$ for all players at specified time resolution (e.g., each original tracking frame or 1Hz).

**Architecture choices:** (flexible, but aim for something like)

- Encoder:
  - per-time-step set encoder (Set Transformer / GNN) over players to capture spatial relations.
- Temporal modeling:
  - gated RNN/GRU/Transformer over time (sequence modeling).
- Decoder:
  - outputs reconstructed positions for each player and time.

**Loss function:**

We define:

$$
\mathcal{L}
= \lambda_{\text{MSE}} \mathcal{L}_{\text{MSE}}
+ \lambda_{\text{centroid}} \mathcal{L}_{\text{centroid}}
+ \lambda_{\text{shape}} \mathcal{L}_{\text{shape}}
+ \lambda_{\text{hull}} \mathcal{L}_{\text{hull\_proxy}}
$$

Source/context: [getgoalsideanalytics](https://www.getgoalsideanalytics.com/high-fat-data-for-low-er-fat-costs/)

Where:

- $\mathcal{L}_{\text{MSE}}$:
  - Standard coordinate MSE between true and reconstructed positions (only available for masked SkillCorner). [getgoalsideanalytics](https://www.getgoalsideanalytics.com/high-fat-data-for-low-er-fat-costs/)
- $\mathcal{L}_{\text{centroid}}$:
  - Squared difference between true and reconstructed team centroids.
- $\mathcal{L}_{\text{shape}}$:
  - Difference in principal variances (longitudinal and lateral) of player distribution:
    - captures block height and width. [getgoalsideanalytics](https://www.getgoalsideanalytics.com/high-fat-data-for-low-er-fat-costs/)
- $\mathcal{L}_{\text{hull\_proxy}}$:
  - Differentiable proxy for convex hull boundary:
    - e.g., radial distances in multiple angular sectors around centroid, comparing true vs reconstructed radial profiles. [getgoalsideanalytics](https://www.getgoalsideanalytics.com/high-fat-data-for-low-er-fat-costs/)

We want to implement multiple variants:

- Baseline:
  - $\mathcal{L}_{\text{MSE}}$ only.
- HalfSpace full:
  - $\mathcal{L}_{\text{MSE}} + \mathcal{L}_{\text{centroid}} + \mathcal{L}_{\text{shape}} + \mathcal{L}_{\text{hull\_proxy}}$.

### 5.4 Feature extraction module

Given reconstructed trajectories (and/or ground truth), compute team-level features for each timestep or window:

- Defensive line height:
  - average y-coordinate of last line of defenders (or something similar).
- Team width:
  - lateral spread of outfield players.
- Compactness:
  - covariance or total variance.
- Hull-proxy features:
  - radial profile metrics in different angles.
- Possession/pressure proxies:
  - derived from events or relative positions if possible. [pmc.ncbi.nlm.nih](https://pmc.ncbi.nlm.nih.gov/articles/PMC12163489/)

Outputs:

- `TacticalFeatureSequence` for each match:
  - time-indexed vectors of features.

### 5.5 Tactical state decoder (sequence model with constraints)

We model tactical states $z_t$ as discrete labels:

- High Press
- Mid-Block
- Low Block
- Transition

We want a sequence model, e.g.:

- HMM-like / CRF-like decoder:
  - $P(z_t \mid z_{t-1}, x_t)$
- Emissions conditioned on features $x_t$.
- Transitions constrained by:
  - physical limits (how fast line height can change),
  - optional analyst-defined constraints (thresholds on line height, compactness, etc.). [getgoalsideanalytics](https://www.getgoalsideanalytics.com/high-fat-data-for-low-er-fat-costs/)

**Analyst-tunable constraints:**

- Analyst can supply inequality constraints such as:
  - High Press: $h_t > H_{\text{min}}$, compactness within X, etc.
- Decoder respects these by:
  - masking or penalizing states that violate constraints,
  - shaping transition probabilities.

This is not the main novelty, but improves interpretability and alignment with coaching language. [getgoalsideanalytics](https://www.getgoalsideanalytics.com/high-fat-data-for-low-er-fat-costs/)

### 5.6 Benchmark and evaluation scripts

We need scripts to:

- Generate masked SkillCorner data,
- Run imputation and store reconstructions,
- Extract features,
- Train and apply tactical state decoder,
- Compute metrics:
  - RMSE, centroid error, shape errors, hull-proxy error,
  - Macro-F1 for state classification,
  - number of state switches per minute (stability),
  - robustness under different analyst thresholds. [getgoalsideanalytics](https://www.getgoalsideanalytics.com/high-fat-data-for-low-er-fat-costs/)

We also need scripts for:

- Applying HalfSpace to a StatsBomb 360 subset,
- Saving reconstructed trajectories and predicted states.

***

## 6. Benchmark: SB360-Tactical-Bench

### 6.1 Purpose

Create a small but high-value open benchmark that:

- uses only open data,
- tests both reconstruction and tactical state recognition,
- can be reused by other researchers. [github](https://github.com/statsbomb/open-data)

### 6.2 Composition

**Part A – Masked SkillCorner benchmark**

- Select ~5–10 SkillCorner Open Data matches. [thesignificantgame](https://www.thesignificantgame.com/portfolio/first-look-at-skillcorner-s-free-tracking-dataset/)
- For each:
  - full tracking (ground truth),
  - masked sparse observations,
  - computed features,
  - human-labeled macro tactical states on key windows.

**Part B – StatsBomb 360 benchmark**

- Select ~5–10 StatsBomb 360 matches with rich freeze-frames. [blogarchive.statsbomb](https://blogarchive.statsbomb.com/news/statsbomb-360-freeze-frame-viewer-a-new-release-in-statsbomb-iq/)
- For each:
  - events + freeze-frames,
  - features computed from HalfSpace reconstructions,
  - human-labeled macro states based on freeze-frames + video.

### 6.3 Deliverables

- Repository content for the benchmark:
  - Scripts to download or locate open data (no redistribution if license forbids raw data).
  - Masking, feature extraction, label format, evaluation scripts.
  - Label files (JSON/CSV) for states per time window.

Goal: a reader can replicate results using the open data and our scripts.

***

## 7. Evaluation metrics and baselines

### 7.1 Reconstruction metrics (masked SkillCorner)

- **RMSE** per player per frame (baseline objective).
- **Centroid error**:
  - distance between true and reconstructed team centroid.
- **Shape error**:
  - difference in longitudinal and lateral variances.
- **Hull-proxy error**:
  - L2 difference between true and reconstructed radial profiles.

Baselines:

- simple linear interpolation,
- possibly a reimplementation of ARMAX-style or similar from literature if feasible. [getgoalsideanalytics](https://www.getgoalsideanalytics.com/high-fat-data-for-low-er-fat-costs/)

HalfSpace vs baselines:

- Expect HalfSpace to be:
  - similar or slightly worse in RMSE,
  - significantly better in centroid, shape, and hull metrics.

### 7.2 Tactical metrics (SkillCorner + StatsBomb 360)

- **Macro-F1 / accuracy** for state classification:
  - using labeled states.
- **State transition stability**:
  - number of state switches per minute,
  - penalize jitter.
- **Robustness to constraints**:
  - vary analyst thresholds over a reasonable range,
  - measure how often state segments stay consistent.

Goal:

- Show topology-preserving imputation improves tactical metrics compared to baselines, even if RMSE is slightly worse. [getgoalsideanalytics](https://www.getgoalsideanalytics.com/high-fat-data-for-low-er-fat-costs/)

***

## 8. Technical stack and repo structure

### 8.1 Technology stack

- **Language:** Python 3.11+
- **Core libraries:**
  - PyTorch (for imputer),
  - NumPy, SciPy,
  - pandas,
  - scikit-learn. [arxiv](https://arxiv.org/html/2502.02785v2)
- **Football-specific:**
  - `statsbombpy` for StatsBomb data ingestion. [github](https://github.com/statsbomb/statsbombpy)
  - `kloppy` for SkillCorner open data ingestion. [kloppy.pysport](https://kloppy.pysport.org/user-guide/loading-data/skillcorner/)
- **Visualization:**
  - matplotlib, mplsoccer,
  - optional Plotly.

### 8.2 Suggested repo layout

- `PROJECT.md` – this specification.
- `README.md` – short overview for GitHub.
- `data/` – scripts for fetching and preparing open data.
- `halfspace/imputer/` – model architectures and loss definitions.
- `halfspace/features/` – feature extraction.
- `halfspace/decoder/` – tactical state decoder (HMM/CRF, constraints).
- `benchmarks/sb360_tactical/` – benchmark generation and evaluation.
- `notebooks/` – exploration and development notebooks.

***

## 9. Implementation priorities

Before building UI or advanced tooling, focus on:

1. **Core imputation model (HalfSpace-Imputer)**:
   - baseline version + topology-loss version.

2. **Masking and validation pipeline**:
   - SkillCorner masking,
   - evaluation code against ground truth.

3. **Tactical state decoder**:
   - simple HMM/CRF first,
   - then add constraint masks.

4. **Small initial SB360-Tactical-Bench**:
   - label a small set of matches,
   - define the evaluation protocol.

Only after that:

- consider any replay UI or basic visualization beyond necessary plots.

***

## 10. Success criteria

HalfSpace is considered successful if:

1. On masked SkillCorner data:
   - HalfSpace’s topology-loss model is competitive on RMSE and clearly better on shape metrics than baselines. [getgoalsideanalytics](https://www.getgoalsideanalytics.com/high-fat-data-for-low-er-fat-costs/)

2. On SB360-Tactical-Bench:
   - HalfSpace shows higher macro-F1 for tactical states and more stable segments than non-topology-preserving baselines. [getgoalsideanalytics](https://www.getgoalsideanalytics.com/high-fat-data-for-low-er-fat-costs/)

3. Reproducibility:
   - Everyone can reproduce results using only open data and the repo scripts. [github](https://github.com/statsbomb/open-data)

4. Benchmark:
   - SB360-Tactical-Bench is clearly documented and usable by others.

***

You can now:

- Copy this entire document and save it as `PROJECT.md` or any format you like.
- Ask ChatGPT (or another tool) to turn it into a downloadable PDF/Doc/Markdown file.
- Feed this to Antigravity as the authoritative spec for HalfSpace.
