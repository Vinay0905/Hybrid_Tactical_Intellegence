# HalfSpace: Premium SaaS Product Architecture & "WOW" Factor Tech Stack

This document details the modern, high-performance, industry-standard tech stack and interactive features designed to create a premium, state-of-the-art SaaS product for football clubs.

---

## 1. The High-Performance Tech Stack

To move away from basic scripts and static templates, we utilize a modern, GPU-accelerated, type-safe stack:

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                              FRONTEND (Next.js / TypeScript)                           │
│  - Tailwind CSS + Framer Motion (Glassmorphism & premium dark UI)                      │
│  - Pixi.js / WebGL2 (GPU-accelerated 2D Pitch rendering at 120Hz+)                     │
│  - ONNX Runtime Web (Wasm) (Runs the imputer model directly inside the user's browser) │
└────────────────────────────────────────────────────────┬───────────────────────────────┘
                                                         │
                                                  (Local / Cloud API)
                                                         │
                                                         ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                                 BACKEND (Go / ONNX)                                    │
│  - Go Gin/Fiber API Server (Ultra-high concurrency, simple and fast REST/gRPC API)     │
│  - ONNX Runtime Go (Runs compiled PyTorch model weights at sub-millisecond speeds)      │
│  - SQLite / PostgreSQL (High-performance relational persistence)                       │
└────────────────────────────────────────────────────────────────────────────────────────┘
```

### 1.1 The Frontend (TypeScript, Next.js, WebGL2)
* **Framework:** **Next.js 14+** (React) or **Vite + React** with **TypeScript** for absolute type safety and component-driven modularity.
* **Rendering Engine:** **PixiJS** or raw **WebGL2 Canvas** (instead of standard HTML5 canvas). This enables GPU-accelerated drawing, allowing smooth rendering of player circles, movement vectors, particle trails, and tactical zones at 144Hz.
* **In-Browser Model Execution:** **ONNX Runtime Web (Wasm)**. 
  * *How it works:* We train the imputer model in PyTorch, export the weights to `.onnx` format, and load them directly into the browser. 
  * *The Wow Factor:* When a coach uploads tracking data, the model does not send the data to a server. It runs the neural network *locally on the coach's device (CPU/GPU) in the browser* in milliseconds. This provides zero server costs, absolute data privacy, and instant results.

### 1.2 The Backend (Go + compiled ONNX)
* **Language:** **Go (Golang)** with a high-performance web framework like **Gin** or **Fiber**.
* *Why:* Incredibly fast, extremely simple to learn/code compared to Rust, utilizes a small memory footprint, and handles thousands of concurrent connections natively via Go's scheduler (goroutines).
* **Inference Pipeline:** We run the ONNX model in Go using `onnxruntime-go` bindings. This runs model forward-passes in sub-milliseconds without Python dependency overhead, keeping the production deployment compact and highly performant.

---

## 2. The "WOW" Features (The Aww Factor)

To make coaches and club directors go "wow" at first glance:

### 2.1 Holographic/Neon Pitch Visualizer
* **Aesthetics:** A sleek dark-mode stadium layout. Player circles are glowing neon disks (cyan for home, magenta for away).
* **Motion Vector Trails:** As players run, they leave fading, glowing particle trails behind them showing their speed and direction.
* **Dynamic Defensive Grids:** Draw a translucent, morphing defensive block polygon (using the calculated convex hull) that changes color in real-time as the team shifts from a wide press (red) to a compact block (deep blue).

```
        Neon Vector Trails & Convex Hull Overlay:
        
             ● (Player) ─ ─ ─ ─ ─ (Fading Trail)
            /   \
           /     \  [Translucent glowing polygon matches block shape]
          /       \
         ● ─────── ●
```

### 2.2 3D Camera Rotation (Three.js)
* **The Interaction:** With a single toggle, the flat 2D pitch tilts into a 3D tactical stadium view.
* **Tactical Camera Angles:** Allow coaches to scrub the timeline and view the match from:
  1. *The Press Box:* Standard high-angle tactical view.
  2. *First-Person Player View:* View what the central midfielder saw before making a pass.
  3. *The Defensive Line:* A view looking down the defensive line to inspect offside traps.

### 2.3 Automatic Voice Commentator (Web Audio API)
* **The Feature:** When the match timeline plays, a speech synthesis voice describes tactical changes:
  * *"14:32 - Beta FC collapses into a compact Low-Block as the opponents progress the ball."*
  * *"34:10 - High-press triggered. Line height pushes up by 15 meters."*
* **Impact:** Provides an automated, engaging narrative that helps non-technical parents and players understand tactical patterns instantly.

### 2.4 Video Annotation Overlay (The Professional Touch)
* The coach uploads match footage. The WebGL overlay matches the pitch lines in the video and draws the **glowing player tracking circles directly on top of the real video feed**.
* The coach can draw lines, highlight space, and measure compactness distances directly on top of the real video in their browser.

---

## 3. Implementation Plan

To build this premium stack:
1. **Model Export:** Add a step in **Phase 4** to export the trained PyTorch model to `halfspace_imputer.onnx`.
2. **Next.js Boot-up:** Initialize a Next.js 14 project using TypeScript and Tailwind CSS.
3. **PixiJS canvas player:** Build the canvas player component inside the Next.js app, loading the `match_simulation.json` dataset.
4. **ONNX integration:** Load the `.onnx` model inside the browser via `onnxruntime-web` to support local browser-based tracking imputation.
5. **Go API Server:** Set up a Go server using Gin/Fiber to manage user authentication, video uploads, and metadata queries.
