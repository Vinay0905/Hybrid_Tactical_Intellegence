# HalfSpace: Product Strategy & UI Roadmap

This document outlines how to transform the core **HalfSpace** machine learning model into a viable, low-cost commercial product tailored specifically for **lower-level, semi-professional, collegiate, and amateur football clubs** who cannot afford dedicated analysts or expensive software (e.g., Wyscout, Hudl).

---

## 1. The Core Product Vision

Lower-tier clubs lack the budget (0₹/low budget constraints) for commercial optical tracking data or multi-person GPS systems. However, they almost always have:
1. **Match video recordings** (shot from a single camera, iPad on a tripod, or a cheap drone).
2. **Coaches or volunteer parents** who manually record basic stats (passes, shots, goals) during a match.

### HalfSpace's Product Value Proposition
> "Upload your match video or event list, and HalfSpace will automatically reconstruct your team's tactical shape, segment the match into defensive blocks, and let you replay the match as a 2D simulated board—just like Football Manager."

---

## 2. Product Architecture & User Journey

To make the product accessible, we design a 3-step pipeline:

```
+──────────────────────────+     +──────────────────────────+     +──────────────────────────+
│    1. Input Capture      │     │  2. Reconstruction Engine │     │     3. Interactive UI     │
│  - Video Upload          │ ──> │  - HalfSpace Imputer     │ ──> │  - 2D Simulation Replay  │
│  - Camera-to-Coordinates │     │  - CR-HMM State Decoder  │     │  - Tactical Timeline     │
│  - Event Log Ingestion   │     │  - Metrics Extraction    │     │  - Automated Coaching Log│
+──────────────────────────+     +──────────────────────────+     +──────────────────────────+
```

### Step 1: Input Capture (The Video Bridge)
* **The Problem:** How do we get coordinates if they don't have tracking data?
* **The Solution:** 
  * A lightweight, web-based computer vision tool. The coach uploads their raw video file.
  * A pre-trained object-detection model (like a customized YOLOv8 running locally in the browser or on a cheap serverless endpoint) detects player positions relative to the screen.
  * A simple camera calibration step (homography matrix) maps the video pixels onto a standard 2D pitch coordinate frame.
  * Alternatively, a **Manual Click-and-Impute interface**: The coach scrubs to a critical event (e.g., a conceded goal) and clicks on the screen to place the 11 defending players. The HalfSpace imputer automatically fills in the movement sequences before and after that moment.

### Step 2: Reconstruction Engine (The Core Model)
* The PyTorch `BiGRUImputer` (or GNN Autoencoder) takes the sparse coordinates from the video detector and fills in the gaps.
* The `CRHMDecoder` segments the entire match into: *When was the team in a Mid-Block? When did they transition? When were they pressing high?*

### Step 3: Interactive UI (The "Live Match" Simulation)
The user interface should feel like a game simulator (Football Manager / EA FC Career Mode), not a dry spreadsheet:
* **The 2D Pitch Canvas:** Renders smooth, moving colored dots for players and a ball. 
* **The Interactive Timeline:** Color-coded timeline displaying tactical states. A coach can click a "Transitions" block to instantly jump to the moments when transition turnovers occurred.
* **Auto-Coach Alerts:** The UI highlights anomalies:
  * *"Compactness alert at 24:12 – Midfield line was stretched 22m wide, creating space."*
  * *"Defensive line dropped to 12m (Low-Block) too early during the buildup."*

---

## 3. UI Concept & Mockup Structure

The interface contains three synchronized panels designed for tablet or laptop screen formats:

```
┌──────────────────────────────────────────────────────────────────────────────────────┐
│  [Score: 1 - 0]                   HALFSPACE ANALYST HUB               [Clock: 34:12] │
├──────────────────────────────────────────────────────────────────────────────────────┤
│  ┌──────────────────────────────────────────────────┐ ┌────────────────────────────┐  │
│  │                                                  │ │      AUTO-COACH INSIGHTS   │  │
│  │                2D MATCH SIMULATOR                │ │                            │  │
│  │                                                  │ │ ● Transition Turnovers: 4  │  │
│  │                     ● (Away)                     │ │   - Line height dropped    │  │
│  │                                                  │ │     too fast at 14m/s.     │  │
│  │          ● (Home)        ★ (Ball)                │ │                            │  │
│  │                                                  │ │ ● Defensive Compactness:   │  │
│  │                                                  │ │   - gap between lines is   │  │
│  │                                                  │ │     currently 18.5 meters  │  │
│  │                                                  │ │     (Target: < 12m).       │  │
│  │                                                  │ │                            │  │
│  └──────────────────────────────────────────────────┘ └────────────────────────────┘  │
├──────────────────────────────────────────────────────────────────────────────────────┤
│  CONTROLS: [ Play / Pause ]  [ 1x / 2x / 5x Speed ]      ACTIVE STATE: [ MID-BLOCK ] │
├──────────────────────────────────────────────────────────────────────────────────────┤
│  TACTICAL TIMELINE:                                                                  │
│  [  High Press  ] [        Mid-Block        ] [  Transition  ] [     Low-Block     ] │
│  0m               15m                        32m              45m                90m │
└──────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 4. Productization Roadmap (Milestones)

Here is how you can systematically build this project into a native desktop SaaS application:

### 🚀 Phase A: The Scientific Core (Current Stage)
* **Goal:** Build the PyTorch models, custom mathematical losses, and run validations against open datasets.
* **Deliverable:** Command-line pipeline code.

### 💻 Phase B: Standalone Desktop App (Tauri + React + Go Sidecar)
* **Goal:** Compile the app as a native executable for macOS and Windows.
* **Tech Stack:**
  * **Frontend View:** React + TypeScript + WebGL (PixiJS) packaged inside Tauri container.
  * **Local Backend Sidecar:** Go (embedded with SQLite database and ONNX inference runtime).
* **Deliverable:** A `.dmg`/`.exe` file that runs completely offline on the coach's desktop, using local SQLite storage.

### 🔒 Phase C: Authentication & Cloud Synchronization
* **Goal:** Add login validation and allow coaches to save and share tactical playbacks.
* **Tech Stack:**
  * **User Access Control:** Supabase Auth (Sign-in/Sign-up + JWT token validation).
  * **Database Sync:** Sync SQLite data to Supabase Postgres cloud tables when an internet connection is detected.
* **Deliverable:** Secured native login and seamless multi-device cloud synchronization.

### 📸 Phase D: Video Tracking Extraction (The Pro Tool)
* **Goal:** Allow coaches to upload video files directly to extract pitch coordinates.
* **Tech Stack:**
  * **Local Computer Vision:** Go calling YOLOv8 (via OpenCV Go bindings) or uploading to a cloud GPU serverless endpoint (AWS Lambda/RunPod) to output object-detection JSON files.
  * **Homography Calibration:** Interactive frontend overlay mapping camera coordinates to standard pitch dimensions.
* **Deliverable:** Drag-and-drop video uploads inside the desktop app producing instant 2D simulations.
