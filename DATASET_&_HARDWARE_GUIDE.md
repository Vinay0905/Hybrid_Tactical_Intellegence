# HalfSpace: Hardware, APIs & Data Acquisition Guide

This guide details the physical hardware requirements, API dependencies (or lack thereof), and step-by-step instructions on how to download the free, open-source football datasets needed to train and test the **HalfSpace** framework.

---

## 1. Hardware Requirements & Performance

Because the project focuses on lightweight, high-utility tactical models (like Bidirectional GRUs and CR-HMM decoders) rather than large-scale foundational models, **you do not need high-end or expensive hardware.**

### 1.1 Local Laptop / Desktop Specs (Minimum & Recommended)
* **CPU:** 
  * *Minimum:* Standard Quad-Core Intel i5 / AMD Ryzen 5 or equivalent.
  * *Recommended:* Apple Silicon (M1/M2/M3/M4) or Intel i7/Ryzen 7 (for faster data parsing).
* **RAM:** 8 GB RAM minimum (16 GB recommended to prevent memory constraints when loading long, raw JSON tracking files).
* **GPU (Optional but helpful):**
  * *NVIDIA:* Any CUDA-compatible GPU (e.g., GTX 1060, RTX 20/30/40 series).
  * *Apple Silicon:* PyTorch uses **MPS (Metal Performance Shaders)** automatically.
  * *If no GPU is present:* Training the baseline BiGRU model on CPU will only take roughly **3 to 10 minutes** per run due to the low dimensionality.

### 1.2 Free Cloud Option (Google Colab)
* If your local machine is slow, you can run the entire pipeline inside a free **Google Colab** notebook using a free T4 GPU instance.

---

## 2. API Dependencies

* **Cost:** 0₹ (Zero Budget Constraint).
* **API Keys Required:** **None**.
* Both StatsBomb and SkillCorner release their public open data under open-source licenses on GitHub, meaning they can be parsed directly using public URLs or downloaded locally without registering for commercial developer accounts.

---

## 3. Data Retrieval Guide

You will use two specific open-source datasets: **SkillCorner Open Data** (for training and absolute ground-truth coordinate validation) and **StatsBomb Open Data** (for sparse freeze-frame testing).

### 3.1 SkillCorner Open Data (Tracking)
SkillCorner provides full 10Hz broadcast-derived tracking coordinates for 9 complete professional matches.

* **GitHub Repository URL:** [SkillCorner/opendata](https://github.com/SkillCorner/opendata)
* **How to acquire:**
  1. Open a terminal in your workspace.
  2. Clone the repository into a temporary folder (or download it as a ZIP file):
     ```bash
     git clone https://github.com/SkillCorner/opendata.git
     ```
  3. Locate the `data/` directory inside the cloned folder.
  4. Create a folder named `data_raw` in your workspace root (`Football/data_raw`).
  5. Copy the following files from the cloned repository into your `data_raw` folder:
     * **Metadata file:** `data/matches/match_data.json` (rename or save as `match_data.json`).
     * **Tracking file:** `data/matches/structured_data.json` (rename or save as `structured_data.json`).

### 3.2 StatsBomb Open Data (Events + 360 Freeze-Frames)
StatsBomb provides extensive event data and 360 freeze-frames (which capture player positions only around on-ball events).

* **GitHub Repository URL:** [statsbomb/open-data](https://github.com/statsbomb/open-data)
* **How to acquire using Python (0 Key Required):**
  StatsBomb provides a free Python library called `statsbombpy` that fetches event JSON structures directly.
  1. Install the library:
     ```bash
     pip install statsbombpy
     ```
  2. Fetch a match using Python:
     ```python
     from statsbombpy import sb
     
     # List all free competitions
     competitions = sb.competitions()
     print(competitions)
     
     # List matches for Euro 2020 (Competition ID: 55, Season ID: 4)
     matches = sb.matches(competition_id=55, season_id=4)
     print(matches[["match_id", "home_team", "away_team"]])
     
     # Load events for a specific match (e.g. Euro 2020 Final, Match ID: 3795506)
     events = sb.events(match_id=3795506)
     print(events.head())
     ```

---

## 4. Immediate Next Step

To test Phase 1 and run the pipeline, download your first raw tracking files:
1. Run `git clone https://github.com/SkillCorner/opendata.git` in your terminal.
2. Locate the folder `data/matches/4039/` inside the cloned repository.
3. Move `match_data.json` and `structured_data.json` into your `Football/data_raw/` directory.
4. You are now ready to run your `data.py` data loader!
