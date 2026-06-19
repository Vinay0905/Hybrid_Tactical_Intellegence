# HalfSpace: Standalone Desktop App & System Design Specification

This document specifies the system design, authentication flow, local database architecture, and standalone desktop compilation wrapper to build **HalfSpace** as a native desktop application for macOS and Windows.

---

## 1. Desktop App Architecture (Tauri Wrapper)

Instead of running as a browser-hosted website, we compile the Next.js/React + WebGL frontend into a native standalone desktop app using **Tauri** (or **Electron**).

```
┌────────────────────────────────────────────────────────────────────────┐
│                   NATIVE OS DESKTOP CONTAINER (Tauri)                   │
│                                                                        │
│   ┌────────────────────────────────────────────────────────────────┐   │
│   │                 FRONTEND VIEW (Next.js / WebGL)                │   │
│   │  - Neon Pitch Replay (PixiJS running on OS Native Webview)     │   │
│   │  - Local state management & Timeline controls                  │   │
│   └──────────────────────────────┬─────────────────────────────────┘   │
│                                  │ (Tauri IPC Bridge / Go Sidecar)     │
│                                  ▼                                     │
│   ┌────────────────────────────────────────────────────────────────┐   │
│   │                 LOCAL DESKTOP BACKEND (Go API)                 │   │
│   │  - Local SQLite Database (Offline-first data storage)          │   │
│   │  - Local File System Access (Bypasses upload limits for video)  │   │
│   │  - ONNX Runtime Go (Runs local trajectory imputer model)       │   │
│   └──────────────────────────────┬─────────────────────────────────┘   │
└──────────────────────────────────┼─────────────────────────────────────┘
                                   │ (Cloud Synchronization when online)
                                   ▼
┌────────────────────────────────────────────────────────────────────────┐
│                          CLOUD SERVICES (SaaS API)                     │
│  - Supabase/Firebase Auth (Secure user log-in and token validation)    │
│  - Supabase PostgreSQL (Cloud backup and team sharing database)        │
└────────────────────────────────────────────────────────────────────────┘
```

### 1.1 Why Tauri over Electron?
* **File Size:** Tauri binaries are **~10–15 MB** (because Tauri uses the operating system's native Webview (Webkit on macOS, WebView2 on Windows), whereas Electron bundles the entire Chromium browser, producing **150 MB+** files).
* **Performance:** Uses significantly less RAM and CPU, which is vital when rendering high-framerate WebGL tactical animations.
* **Native File System Access:** Allows the app to read large, raw match video files directly from the coach's local hard drive instantly without browser-based HTTP upload constraints.

---

## 2. Authentication & Security Flow

For user onboarding, security, and access control:

1. **Auth Provider:** Use **Supabase Auth** or **Firebase Auth** (SaaS standard, secure, free tier).
2. **Login Flow:**
   * User opens the desktop app. If no token is cached, they are presented with a premium Dark UI Login screen.
   * On login, the app requests credentials from Supabase Auth and receives a secure **JSON Web Token (JWT)**.
   * The JWT is securely cached in the desktop OS Keyring (using Tauri's secure storage plugin) to keep the coach logged in.
3. **Protected API Access:** Every request from the desktop client to the cloud database includes the JWT in the `Authorization: Bearer <TOKEN>` header.

---

## 3. Database Architecture (Offline-First Sync)

Coaches are frequently on training pitches or stadiums with weak or non-existent internet connections. Therefore, the app is designed with an **Offline-First** model.

### 3.1 Local Storage (SQLite)
* **Technology:** **SQLite** embedded inside the desktop Go backend.
* **Schema:** Matches the Parquet and Event database schemas exactly.
* **Data Flow:** When a coach imports a match, the data is instantly written to the local SQLite file. The 2D player simulator runs completely offline using local data.

### 3.2 Cloud Synchronization (Supabase PostgreSQL)
* When the app detects an active internet connection, it runs a background sync worker.
* It uploads new match records and tactical labels from the local SQLite database to a centralized cloud PostgreSQL database.
* This allows coaches to sync coordinates and share tactical replay files seamlessly with other staff and players.

---

## 4. Native Integration Features

By deploying as a standalone app, we unlock native operating system features:

* **File Drag-and-Drop:** Drag a match video directly from the desktop into the window. The Go sidecar triggers the camera tracking pipeline immediately.
* **System Tray Notifications:** Runs background processing and notifies the coach when a 90-minute video has finished coordinate extraction: *"Match Analysis Complete: Euro 2020 Final is ready for simulation."*
* **Local Video Player Sync:** Plays the real match video in a small picture-in-picture window that is frame-synced to the 2D neon dot movement on the canvas.
