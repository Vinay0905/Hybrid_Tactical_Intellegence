# Phase 5: 2D Pitch Simulation Player

This phase details how to construct the lightweight 2D Pitch Simulation visualizer. The simulator loads the exported `match_simulation.json` file and plays back player movements using 60fps coordinate interpolation, replicating a Career Mode simulation view (as seen in FM26 / FC26).

---

## 1. File Setup Guide

Follow these simple steps to instantiate and run your visualizer:

1. **Create the HTML file:**
   Create a new file named `index.html` in the root of your workspace (`Football/index.html`).
2. **Add the simulation data:**
   Place the exported `match_simulation.json` file (generated in Phase 4) in the same directory (`Football/match_simulation.json`).
3. **Run the local visualizer:**
   Double-click the `index.html` file to open it in a browser, or run a simple local server to avoid browser CORS policy blocking the fetch of the JSON file:
   ```bash
   python -m http.server 8000
   ```
   Open your browser and navigate to `http://localhost:8000`.

---

## 2. Complete Phase 5 Code

Add the following code to `Football/index.html`:

```html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>HalfSpace: 2D Live Match Simulation</title>
    <style>
        :root {
            --bg-color: #0f172a;
            --pitch-color: #1e3a24;
            --pitch-lines: rgba(255, 255, 255, 0.4);
            --card-bg: #1e293b;
            --text-main: #f8fafc;
            --text-sub: #94a3b8;
            --primary: #38bdf8;
            
            /* Tactical State Colors */
            --color-press: #ef4444; /* Red */
            --color-mid: #f59e0b;   /* Amber */
            --color-low: #3b82f6;   /* Blue */
            --color-trans: #8b5cf6; /* Purple */
        }

        body {
            background-color: var(--bg-color);
            color: var(--text-main);
            font-family: 'Outfit', sans-serif;
            margin: 0;
            padding: 20px;
            display: flex;
            flex-direction: column;
            align-items: center;
        }

        .container {
            width: 100%;
            max-width: 900px;
            background: var(--card-bg);
            padding: 20px;
            border-radius: 12px;
            box-shadow: 0 4px 20px rgba(0,0,0,0.3);
        }

        header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 15px;
        }

        .scoreboard {
            font-size: 20px;
            font-weight: 700;
        }

        .clock {
            font-family: monospace;
            font-size: 22px;
            background: #0f172a;
            padding: 4px 12px;
            border-radius: 6px;
            color: var(--primary);
        }

        .pitch-container {
            position: relative;
            width: 100%;
            aspect-ratio: 1.54;
            background-color: var(--pitch-color);
            border-radius: 8px;
            overflow: hidden;
            border: 2px solid #334155;
        }

        canvas {
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
        }

        .controls {
            display: flex;
            gap: 15px;
            margin-top: 15px;
            align-items: center;
        }

        button {
            background: var(--primary);
            color: #0f172a;
            border: none;
            padding: 8px 16px;
            border-radius: 6px;
            font-weight: 600;
            cursor: pointer;
            transition: opacity 0.2s;
        }

        button:hover {
            opacity: 0.9;
        }

        .speed-btn {
            background: #334155;
            color: var(--text-main);
        }

        .phase-indicator {
            margin-left: auto;
            display: flex;
            align-items: center;
            gap: 8px;
        }

        .state-badge {
            padding: 4px 12px;
            border-radius: 20px;
            font-weight: 600;
            font-size: 13px;
            text-transform: uppercase;
        }

        .timeline-container {
            margin-top: 20px;
            position: relative;
        }

        .timeline {
            display: flex;
            height: 16px;
            border-radius: 4px;
            overflow: hidden;
            cursor: pointer;
            background: #334155;
        }

        .timeline-segment {
            height: 100%;
        }

        .playhead {
            position: absolute;
            top: -4px;
            width: 3px;
            height: 24px;
            background: var(--text-main);
            box-shadow: 0 0 6px rgba(255, 255, 255, 0.8);
            pointer-events: none;
            transition: left 0.1s linear;
        }

        .time-labels {
            display: flex;
            justify-content: space-between;
            font-size: 12px;
            color: var(--text-sub);
            margin-top: 4px;
        }
    </style>
</head>
<body>

<div class="container">
    <header>
        <div class="scoreboard">ALPHA FC vs BETA C</div>
        <div class="clock" id="match-clock">00:00.0</div>
    </header>

    <div class="pitch-container">
        <canvas id="pitch-canvas"></canvas>
    </div>

    <div class="controls">
        <button id="play-pause-btn">Play</button>
        <button id="speed-btn" class="speed-btn">1x Speed</button>
        
        <div class="phase-indicator">
            <span style="color: var(--text-sub)">Tactical Phase:</span>
            <span id="active-state" class="state-badge" style="background: #334155;">LOADING</span>
        </div>
    </div>

    <div class="timeline-container">
        <div class="timeline" id="scrub-timeline">
            <!-- Timeline segments will be loaded dynamically based on states -->
        </div>
        <div class="playhead" id="playhead" style="left: 0%;"></div>
        <div class="time-labels">
            <span>00:00</span>
            <span>30:00</span>
            <span>60:00</span>
            <span>90:00</span>
        </div>
    </div>
</div>

<script>
    // Local state variables
    let matchData = null;
    const canvas = document.getElementById('pitch-canvas');
    const ctx = canvas.getContext('2d');
    
    let isPlaying = false;
    let playheadTime = 0.0; // Elapsed match seconds
    let playbackSpeed = 1.0; // Playback speed multiplier (1x, 2x, 5x)
    let lastAnimFrameTime = null;

    // Rescale Canvas to fit container
    function resizeCanvas() {
        canvas.width = canvas.offsetWidth;
        canvas.height = canvas.offsetHeight;
    }
    window.addEventListener('resize', resizeCanvas);
    resizeCanvas();

    // Map physical pitch coordinates (meters) to canvas coordinate space
    function mapCoords(x, y) {
        if (!matchData) return [0, 0];
        const len = matchData.meta.pitch_length;
        const wid = matchData.meta.pitch_width;
        
        // Convert range [-len/2, len/2] to [0, canvas.width]
        const cx = ((x + len/2) / len) * canvas.width;
        // Convert range [-wid/2, wid/2] to [0, canvas.height]
        const cy = ((y + wid/2) / wid) * canvas.height;
        return [cx, cy];
    }

    // Drawing Pitch Lines
    function drawPitch() {
        ctx.fillStyle = '#1e3a24';
        ctx.fillRect(0, 0, canvas.width, canvas.height);
        
        ctx.strokeStyle = 'rgba(255,255,255,0.4)';
        ctx.lineWidth = 1.5;
        
        // Touchline and Goal-line boundary
        ctx.strokeRect(10, 10, canvas.width - 20, canvas.height - 20);
        
        // Halfway line
        ctx.beginPath();
        ctx.moveTo(canvas.width / 2, 10);
        ctx.lineTo(canvas.width / 2, canvas.height - 10);
        ctx.stroke();
        
        // Center Circle
        ctx.beginPath();
        ctx.arc(canvas.width / 2, canvas.height / 2, canvas.height * 0.15, 0, 2 * Math.PI);
        ctx.stroke();
    }

    // 60fps Dynamic Linear Interpolation
    function getInterpolatedFrame(elapsedSec) {
        if (!matchData) return null;
        const frames = matchData.frames;
        const elapsedMs = elapsedSec * 1000;
        
        // Clamp to match start/end
        if (elapsedMs <= frames[0].timestamp_ms) return frames[0];
        if (elapsedMs >= frames[frames.length - 1].timestamp_ms) return frames[frames.length - 1];
        
        // Locate matching interval
        for (let i = 0; i < frames.length - 1; i++) {
            const f_k = frames[i];
            const f_kp1 = frames[i + 1];
            
            if (elapsedMs >= f_k.timestamp_ms && elapsedMs <= f_kp1.timestamp_ms) {
                const fraction = (elapsedMs - f_k.timestamp_ms) / (f_kp1.timestamp_ms - f_k.timestamp_ms);
                
                // Interpolate ball coordinates
                const ball = [
                    f_k.ball[0] + fraction * (f_kp1.ball[0] - f_k.ball[0]),
                    f_k.ball[1] + fraction * (f_kp1.ball[1] - f_k.ball[1])
                ];
                
                // Interpolate Home players
                const home = f_k.home_players.map((p, idx) => [
                    p[0] + fraction * (f_kp1.home_players[idx][0] - p[0]),
                    p[1] + fraction * (f_kp1.home_players[idx][1] - p[1])
                ]);
                
                // Interpolate Away players
                const away = f_k.away_players.map((p, idx) => [
                    p[0] + fraction * (f_kp1.away_players[idx][0] - p[0]),
                    p[1] + fraction * (f_kp1.away_players[idx][1] - p[1])
                ]);
                
                return {
                    clock: f_k.clock,
                    decoded_state: f_k.decoded_state,
                    ball: ball,
                    home_players: home,
                    away_players: away
                };
            }
        }
    }

    function renderLoop(now) {
        if (!lastAnimFrameTime) lastAnimFrameTime = now;
        const deltaSec = (now - lastAnimFrameTime) / 1000.0;
        lastAnimFrameTime = now;
        
        // Render empty pitch if data isn't loaded
        if (!matchData) {
            drawPitch();
            requestAnimationFrame(renderLoop);
            return;
        }

        if (isPlaying) {
            playheadTime += deltaSec * playbackSpeed;
            const maxTime = matchData.frames[matchData.frames.length - 1].timestamp_ms / 1000.0;
            if (playheadTime >= maxTime) {
                playheadTime = maxTime;
                isPlaying = false;
                document.getElementById('play-pause-btn').textContent = "Play";
            }
        }
        
        // Update HUD
        const frame = getInterpolatedFrame(playheadTime);
        if (!frame) return;
        
        document.getElementById('match-clock').textContent = frame.clock;
        
        // Update State Badge
        const badge = document.getElementById('active-state');
        badge.textContent = frame.decoded_state.replace('_', ' ');
        badge.style.background = `var(--color-${frame.decoded_state.split('_')[0].toLowerCase()})`;
        
        // Update timeline progress bar
        const totalDuration = matchData.frames[matchData.frames.length - 1].timestamp_ms / 1000.0;
        const pct = (playheadTime / totalDuration) * 100;
        document.getElementById('playhead').style.left = `${pct}%`;
        
        // Clear and draw pitch lines
        drawPitch();
        
        // Draw Home Players (Teal circles)
        frame.home_players.forEach(p => {
            const [cx, cy] = mapCoords(p[0], p[1]);
            ctx.beginPath();
            ctx.arc(cx, cy, 8, 0, 2*Math.PI);
            ctx.fillStyle = '#06b6d4';
            ctx.fill();
            ctx.strokeStyle = '#ffffff';
            ctx.lineWidth = 1.5;
            ctx.stroke();
        });
        
        // Draw Away Players (Pink circles)
        frame.away_players.forEach(p => {
            const [cx, cy] = mapCoords(p[0], p[1]);
            ctx.beginPath();
            ctx.arc(cx, cy, 8, 0, 2*Math.PI);
            ctx.fillStyle = '#ec4899';
            ctx.fill();
            ctx.strokeStyle = '#ffffff';
            ctx.lineWidth = 1.5;
            ctx.stroke();
        });
        
        // Draw Match Ball (White circle)
        const [bcx, bcy] = mapCoords(frame.ball[0], frame.ball[1]);
        ctx.beginPath();
        ctx.arc(bcx, bcy, 5, 0, 2*Math.PI);
        ctx.fillStyle = '#ffffff';
        ctx.fill();
        ctx.strokeStyle = '#000000';
        ctx.stroke();
        
        requestAnimationFrame(renderLoop);
    }

    // --- Interactive Control Listeners ---
    document.getElementById('play-pause-btn').addEventListener('click', (e) => {
        if (!matchData) return;
        isPlaying = !isPlaying;
        e.target.textContent = isPlaying ? "Pause" : "Play";
        lastAnimFrameTime = performance.now();
    });

    document.getElementById('speed-btn').addEventListener('click', (e) => {
        if (playbackSpeed === 1.0) playbackSpeed = 2.0;
        else if (playbackSpeed === 2.0) playbackSpeed = 5.0;
        else playbackSpeed = 1.0;
        e.target.textContent = `${playbackSpeed}x Speed`;
    });

    // Scrub timeline
    document.getElementById('scrub-timeline').addEventListener('click', (e) => {
        if (!matchData) return;
        const rect = e.currentTarget.getBoundingClientRect();
        const clickedFraction = (e.clientX - rect.left) / rect.width;
        const totalDuration = matchData.frames[matchData.frames.length - 1].timestamp_ms / 1000.0;
        playheadTime = clickedFraction * totalDuration;
    });

    // Build timeline blocks visually
    function initTimeline() {
        const timeline = document.getElementById('scrub-timeline');
        timeline.innerHTML = '';
        
        const frames = matchData.frames;
        const totalDuration = frames[frames.length - 1].timestamp_ms;
        
        for (let i = 0; i < frames.length; i++) {
            const start = frames[i].timestamp_ms;
            const end = (i < frames.length - 1) ? frames[i+1].timestamp_ms : totalDuration;
            const widthPct = ((end - start) / totalDuration) * 100;
            
            const segment = document.createElement('div');
            segment.className = 'timeline-segment';
            segment.style.width = `${widthPct}%`;
            
            // Assign color per state
            const state = frames[i].decoded_state.split('_')[0].toLowerCase();
            segment.style.background = `var(--color-${state})`;
            
            timeline.appendChild(segment);
        }
    }

    // --- Async Data Fetching ---
    fetch('match_simulation.json')
        .then(response => response.json())
        .then(data => {
            matchData = data;
            initTimeline();
            console.log("Match simulation successfully loaded!");
        })
        .catch(err => {
            console.error("Could not find match_simulation.json. Make sure to generate it in Phase 4 and place it in the same directory.", err);
        });

    requestAnimationFrame(renderLoop);
</script>
</body>
</html>
```

---

## 3. Snippet-by-Snippet Explanation

### 3.1 Coordinate Space Translation
```javascript
    function mapCoords(x, y) {
        const len = matchData.meta.pitch_length;
        const wid = matchData.meta.pitch_width;
        const cx = ((x + len/2) / len) * canvas.width;
        const cy = ((y + wid/2) / wid) * canvas.height;
        return [cx, cy];
    }
```
* **Explanation:** Translates metric positions from center-aligned $[-52.5, 52.5]$ meter coordinates directly to local HTML5 canvas boundary coordinates `[0, width]` and `[0, height]` for correct scaling and projection onto the screen.

### 3.2 Client-Side Coordinate Interpolation
```javascript
        const fraction = (elapsedMs - f_k.timestamp_ms) / (f_kp1.timestamp_ms - f_k.timestamp_ms);
        const ball = [
            f_k.ball[0] + fraction * (f_kp1.ball[0] - f_k.ball[0]),
            f_k.ball[1] + fraction * (f_kp1.ball[1] - f_k.ball[1])
        ];
```
* **Explanation:** Runs in the 60fps render loop. If the active playhead falls between export timestamps $k$ and $k+1$ (e.g. at 1.4 seconds), we calculate the exact fraction between the frames (0.4) and calculate coordinates as:
  $$\mathbf{x}_{\text{interp}} = \mathbf{x}_k + 0.4 \cdot (\mathbf{x}_{\text{k}+1} - \mathbf{x}_k)$$
  This generates extremely smooth 60fps moving circles, even though the backend only outputs coordinates at 1Hz/2Hz.

### 3.3 Dynamic Async File Fetching
```javascript
    fetch('match_simulation.json')
        .then(response => response.json())
        .then(data => {
            matchData = data;
            initTimeline();
        })
```
* **Explanation:** Loads the simulation file exported by Python via an AJAX request. Once the JSON finishes transferring, it parses the metadata configurations, populates the color-coded tactical segments timeline, and enables control buttons to start playback.

---

## 4. Testing Suggestions

To check your client-side visualizer:
1. **Local Server CORS Verification:** Open the page using `http://localhost:8000/index.html` after spinning up python's local server. Assert that the visualizer loads successfully and the developer console does not show any CORS warnings.
2. **Speed Multiplier Verification:** Switch playback speed to 5x. Ensure the clock progresses 5x faster and players move proportionally faster.
3. **Boundary Check:** Check that players are drawn strictly within the boundary lines of the pitch and never exceed the green canvas box edges.
4. **Scrub Playhead Synchronization:** Click the 50% mark on the timeline and verify that the playhead dot snaps immediately to the center, and the game clock displays approximately 45:00.
