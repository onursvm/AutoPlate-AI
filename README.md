# ⚡ AutoPlate AI - YOLOv11 & Advanced OCR License Plate Recognition & Target Tracking System

An AI-powered desktop application that detects distant vehicle license plates with high precision using **YOLOv11**, recognizes alphanumeric characters via high-accuracy OCR, and performs real-time matching against a user-defined **watchlist / target plate list**, triggering instant audio and visual alerts.

---

## 🌟 Key Features

1. **YOLOv11 License Plate Detection (`license-plate-finetune-v1n.pt`)**:
   - Custom fine-tuned YOLOv11 architecture specifically trained on vehicle license plates.
   - **High-Resolution Support for Distant Vehicles**: Standard 640px or enhanced 1280px high-resolution scanning modes.
   - Dynamic boundary padding to prevent edge characters from being cropped out.

2. **Advanced OCR & Image Preprocessing (Enhancement Engine)**:
   - Automatic **Bicubic upscaling** for small and distant plate crops.
   - **CLAHE (Contrast Limited Adaptive Histogram Equalization)** to eliminate glare, shadows, and low-light degradation.
   - **Bilateral filtering** and sharpening to accentuate character edges.
   - Intelligent normalization that automatically strips unnecessary symbols and the European/Turkish blue stripe artifacts (`TR` / `R`).

3. **🎯 Target Plate Watchlist & Smart Matching (Target Matcher)**:
   - Add single or comma-separated target plates to the watchlist.
   - Supports exact match, substring match, and **Fuzzy matching (similarity tolerance)**.
   - When a target vehicle is spotted:
     - 🚨 **Visual Alert:** Futuristic red HUD bounding box and header warning banner on the live video stream.
     - 🔊 **Audio Alarm:** Dual-tone high-priority alert sound.
     - 💾 **Automatic Evidence Archiving:** Both the cropped plate and the full-frame timestamped camera snapshot are saved automatically under `records/alerts/`.

4. **📹 Multi-Source Video Input Support**:
   - **Video Files**: Play, pause, seek, and loop `.mp4`, `.avi`, `.mkv`, and `.mov` video files.
   - **Webcams**: Direct live streaming from any connected USB camera (0, 1, 2...).
   - **IP / Security Cameras (RTSP Stream)**: Real-time analysis from surveillance cameras using RTSP (`rtsp://user:pass@ip:port/stream`) or HTTP video feeds.

5. **📋 Live Detection Dashboard & Excel/CSV Reporting**:
   - Real-time tabular listing of all detected plates with timestamps and confidence scores.
   - Double-click any record to inspect high-resolution plate crops and full scene snapshots.
   - One-click export to Excel-compatible UTF-8 CSV.

---

## 🚀 Installation & Quick Start

### 1. Launching the Application
Run the application using the local virtual environment (`.venv`):

```bash
run.bat
