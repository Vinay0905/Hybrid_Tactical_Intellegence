Here’s a `context.md` you can save and reuse later to “remind” any model (including me in a new chat) what HalfSpace is and where we left off.

***

# context.md – HalfSpace Project Context

## 1. Who I am and my constraints

- I’m an ECE student in Chennai aiming at **AI/ML + system design** roles.  
- I’m comfortable with Python, ML frameworks, backend dev, and agentic AI.  
- For this project, my constraints are:
  - **Strict 0₹ budget** – no paid APIs, no Wyscout/Opta/Hudl/SkillCorner subscriptions, no commercial tools. [perplexity](https://www.perplexity.ai/search/8ba6bb79-f49d-444a-9999-65bb428be32c)
  - **Open-source only stack** – Python, PyTorch, pandas, etc.  
  - **Open data only** – especially **StatsBomb Open Data** (events + 360) and **SkillCorner Open Data** tracking. [github](https://github.com/statsbomb/open-data)

I want a project that is:

- technically deep enough for a **paper/preprint**,  
- practically relevant to **smaller football clubs / analysts**,  
- and strong as a **portfolio and interview story**.

***

## 2. Project name and identity

**Project name:** **HalfSpace**

- HalfSpace is **not** a generic dashboard or product clone.  
- It is primarily a **research-grade system** plus a thin prototype tool.

High-level identity:

> HalfSpace is a zero-budget, open-source football analytics project that reconstructs sparse spatial data in a way that preserves **team tactical shape**, and then uses that for **tactical state recognition** (high press / mid-block / low block / transitions).

***

## 3. Data reality

Main data sources:

- **StatsBomb Open Data** – free football events and some **StatsBomb 360** freeze-frames (positional snapshots at on-ball events). [blogarchive.statsbomb](https://blogarchive.statsbomb.com/news/statsbomb-360-freeze-frame-viewer-a-new-release-in-statsbomb-iq/)
- **SkillCorner Open Data** – ~10 matches of full broadcast tracking, open for research. [github](https://github.com/SkillCorner/opendata)

Key points:

- I do **not** assume access to continuous commercial tracking (Stats Perform, Second Spectrum, etc.).  
- My **target setting** is **sparse data**:
  - event + 360 snapshots (for real usage),
  - masked tracking that mimics 360 sparsity (for validation).

***

## 4. The core research idea (what makes it “paper-worthy”)

### 4.1 The problem

- Typical sparse trajectory reconstruction methods focus on minimizing **coordinate RMSE**:
  - they try to get each player’s location as close as possible to the true coordinates. [getgoalsideanalytics](https://www.getgoalsideanalytics.com/high-fat-data-for-low-er-fat-costs/)
- But for tactics, what matters more is **team shape**:
  - centroid (where the block is),
  - width,
  - depth / line height,
  - compactness,
  - overall block topology. [totalfootballanalysis](https://totalfootballanalysis.com/tactical-theory-compactness-tactical-analysis-tactics)

Minimizing only RMSE causes:

- smoothed trajectories (mean collapse),
- flatter line height,
- distorted compactness,
- loss of pressing/defensive structure. [getgoalsideanalytics](https://www.getgoalsideanalytics.com/high-fat-data-for-low-er-fat-costs/)

### 4.2 The thesis

> In sparse-data settings, we should reconstruct trajectories in a way that **preserves the team’s collective topology** (shape), not just individual coordinate accuracy.

HalfSpace’s intended contribution:

- A **topology-preserving trajectory imputer**:
  - combines standard RMSE with:
    - centroid alignment loss,
    - shape/variance loss (width/depth),
    - differentiable hull/radial-profile proxies. [getgoalsideanalytics](https://www.getgoalsideanalytics.com/high-fat-data-for-low-er-fat-costs/)
- Evaluation focused on **downstream tactical state recognition**:
  - high press / mid-block / low block / transitions,
  - state stability (fewer jittery switches),
  - not only on RMSE.

***

## 5. Two operating regimes for HalfSpace

### Regime 1 – Sparse regime (real target)

- Use **StatsBomb events + 360 freeze-frames** only. [github](https://github.com/statsbomb/open-data)
- No full tracking ground truth.
- Goal:
  - reconstruct approximate movement,
  - compute features (line height, width, compactness),
  - classify tactical states over time.

### Regime 2 – Masked full tracking (for validation)

- Use **SkillCorner Open Data** full tracking. [thesignificantgame](https://www.thesignificantgame.com/portfolio/first-look-at-skillcorner-s-free-tracking-dataset/)
- Artificially mask it to a 360-style pattern:
  - keep positions only at event-like times,
  - drop intermediate frames.
- Goal:
  - compare HalfSpace reconstructions against true tracking,
  - show that HalfSpace:
    - is competitive in RMSE,
    - but clearly better in shape metrics and tactical classification. [getgoalsideanalytics](https://www.getgoalsideanalytics.com/high-fat-data-for-low-er-fat-costs/)

***

## 6. Main components we agreed on

Logical modules (not necessarily final code structure):

1. **Data ingestion**
   - Load StatsBomb open events + 360, and SkillCorner tracking using open libraries (`statsbombpy`, `kloppy`). [github](https://github.com/statsbomb/statsbombpy)

2. **Masking module**
   - For SkillCorner tracking: simulate sparse 360-style data via masking.

3. **HalfSpace-Imputer (core model)**
   - Input: sparse trajectories (+ context).
   - Output: reconstructed trajectories.
   - Loss components:
     - MSE (point-wise),
     - centroid loss,
     - shape/variance loss (width/depth),
     - differentiable hull proxy (radial distances). [getgoalsideanalytics](https://www.getgoalsideanalytics.com/high-fat-data-for-low-er-fat-costs/)

4. **Feature extractor**
   - From trajectories:
     - defensive line height,
     - team width,
     - compactness (covariance/variance),
     - hull metrics,
     - possibly simple pressure proxies.

5. **Tactical state decoder**
   - Sequence model (HMM/CRF-style):
     - states: High Press, Mid-Block, Low Block, Transition,
     - emissions: features,
     - transitions: constrained by physics (no instant deep-block → high press).
   - Optional **analyst constraints**:
     - thresholds on line height, compactness, etc., used as masks in decoding. [getgoalsideanalytics](https://www.getgoalsideanalytics.com/high-fat-data-for-low-er-fat-costs/)

6. **SB360-Tactical-Bench (open benchmark)**
   - A small set of:
     - masked SkillCorner matches with labels,
     - StatsBomb 360 matches with macro tactical labels.
   - Released via code + label files, not redistributing raw data. [github](https://github.com/SkillCorner/opendata)

***

## 7. Evaluation philosophy

We explicitly want to show:

1. **Reconstruction:**
   - HalfSpace vs baselines on:
     - RMSE,
     - centroid error,
     - shape error (width/depth covariance),
     - hull proxy error. [getgoalsideanalytics](https://www.getgoalsideanalytics.com/high-fat-data-for-low-er-fat-costs/)

2. **Tactical classification:**
   - Macro-F1 and segment stability for:
     - High Press / Mid-Block / Low Block / Transition.
   - HalfSpace should:
     - beat baselines on tactical metrics,
     - even if its RMSE is slightly worse.

This is the key argument: **tactical utility > pure coordinate RMSE**.

***

## 8. Why this is useful for small clubs

Short story we agreed on:

- Lower-tier clubs often have:
  - one analyst,
  - limited budget,
  - maybe only event-like data and video. [once](https://once.sport/blog/how-do-professionals-analyse/)
- HalfSpace aims to:
  - work with **open/sparse data**,  
  - provide better segmentation of tactical phases,  
  - help analysts find **key phases faster** (pressing phases, block changes, etc.),  
  - without needing expensive tracking or tools. [english-programs.sportsdatacampus](https://english-programs.sportsdatacampus.com/free-football-data-websites/)

HalfSpace is **assistive**:
- not replacing analysts,
- but giving them a head start on “where to look in the match”.

***

## 9. My learning needs (for future conversations)

I asked Gemini (and want future assistants) to help me:

- Understand **football shape metrics** in math:
  - centroid, width, depth, compactness, hull.
- Understand **topology-preserving losses**:
  - why RMSE alone is insufficient,
  - how to define shape-aware terms.
- Understand **sequence models**:
  - HMM/CRF for tactical state segmentation.
- Design a **learning roadmap** to implement HalfSpace step by step.

I prefer:

- **intuitive explanations first**,  
- then formulas,  
- with football-based examples. [youtube](https://www.youtube.com/watch?v=slqZAemcgO4)

***

## 10. Where we are now / what’s next

- We have:
  - a clear **project spec** (HalfSpace),
  - a strong **review from Gemini** confirming it’s “promising but needs one major contribution” (topology-loss + benchmark). [getgoalsideanalytics](https://www.getgoalsideanalytics.com/high-fat-data-for-low-er-fat-costs/)
  - a `PROJECT.md` and prompts for Gemini/deep research.  

- Next phases (for future chats):
  1. Finalize math choices (exact loss definitions, baselines).
  2. Plan milestone roadmap (M1: simple features, M2: imputer baseline, M3: topology losses, M4: benchmark).
  3. Start implementation in a structured way (likely with Antigravity, etc.).

This `context.md` should be enough to “reboot” the conversation in a new chat without re-explaining everything from scratch.