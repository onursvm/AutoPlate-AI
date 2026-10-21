"""
AutoPlate AI - OCR ve Performans Benchmark Aracı
PP-OCRv6, FastPlateOCR ve Çift Motor karşılaştırması
1080p ve 4K test videoları üzerinde otomatik değerlendirme.
"""

import os
import sys
import time
import argparse
import csv
import cv2
import numpy as np
from datetime import datetime
from typing import Dict, Any, List

from detector import (
    PlateDetector, PlateOCR, VehicleDetector,
    clean_plate_text, is_valid_plate_format,
    OCR_ENGINE_PPOCR, OCR_ENGINE_FASTPLATE, OCR_ENGINE_DUAL
)


def run_benchmark(
    video_path: str,
    detector: PlateDetector,
    ocr: PlateOCR,
    engine_name: str,
    max_frames: int = 300,
    frame_interval: int = 2,
    conf_threshold: float = 0.35,
    imgsz: int = 640
) -> Dict[str, Any]:
    """Tek bir video ve OCR motoru için benchmark çalıştırır."""
    ocr.set_engine(engine_name)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[!] Video açılamadı: {video_path}")
        return {}

    total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    is_4k = width >= 2500
    effective_imgsz = 1280 if is_4k else imgsz

    frame_count = 0
    processed_count = 0
    total_detections = 0
    unique_plates = set()
    ocr_latencies: List[float] = []
    plate_results: List[Dict[str, Any]] = []

    t_start = time.perf_counter()

    while cap.isOpened() and frame_count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        if frame_count % frame_interval != 0:
            continue

        processed_count += 1

        # 4K için ROI: Üst %25'i atla (gökyüzü)
        if is_4k:
            crop_y = int(height * 0.25)
            detect_frame = frame[crop_y:, :]
            y_offset = crop_y
        else:
            detect_frame = frame
            y_offset = 0

        # YOLO ile plaka tespiti
        plates = detector.detect_plates(detect_frame, conf_threshold=conf_threshold, imgsz=effective_imgsz)

        for p in plates:
            total_detections += 1
            crop = p['crop']
            if crop.size == 0:
                continue

            # OCR Ölçümü
            t_ocr0 = time.perf_counter()
            text, conf = ocr.read_plate(crop)
            t_ocr_ms = (time.perf_counter() - t_ocr0) * 1000
            ocr_latencies.append(t_ocr_ms)

            clean = clean_plate_text(text)
            is_valid = is_valid_plate_format(clean)

            if is_valid and conf >= 0.35:
                unique_plates.add(clean)

            plate_results.append({
                'frame': frame_count,
                'raw_text': text,
                'clean_text': clean,
                'conf': conf,
                'valid_format': is_valid,
                'latency_ms': t_ocr_ms
            })

    total_time = time.perf_counter() - t_start
    cap.release()

    fps = processed_count / max(0.001, total_time)

    avg_latency = float(np.mean(ocr_latencies)) if ocr_latencies else 0.0
    median_latency = float(np.median(ocr_latencies)) if ocr_latencies else 0.0
    p95_latency = float(np.percentile(ocr_latencies, 95)) if len(ocr_latencies) >= 2 else avg_latency

    return {
        'video': os.path.basename(video_path),
        'resolution': f"{width}x{height}",
        'engine': engine_name,
        'frames_processed': processed_count,
        'total_time_sec': total_time,
        'fps': fps,
        'yolo_detections': total_detections,
        'unique_valid_plates': len(unique_plates),
        'plates_list': sorted(list(unique_plates)),
        'ocr_calls': len(ocr_latencies),
        'latency_avg_ms': avg_latency,
        'latency_median_ms': median_latency,
        'latency_p95_ms': p95_latency,
        'details': plate_results
    }


def main():
    parser = argparse.ArgumentParser(description="AutoPlate AI Benchmark Script")
    parser.add_argument("--frames", type=int, default=200, help="Test edilecek maksimum kare sayısı (varsayılan: 200)")
    parser.add_argument("--interval", type=int, default=3, help="Kare örnekleme aralığı (varsayılan: 3)")
    parser.add_argument("--save-csv", action="store_true", default=True, help="Sonuçları CSV'ye kaydet")
    args = parser.parse_args()

    print("=" * 75)
    print("⚡ AutoPlate AI - OCR & Performans Benchmark Motoru")
    print("=" * 75)

    base_dir = os.path.dirname(os.path.abspath(__file__))
    video_1080p = os.path.join(base_dir, "pexels-taryn-elliott-5309381 (1080p).mp4")
    video_4k = os.path.join(base_dir, "pexels-george-morina-6719160 (2160p).mp4")

    videos = [v for v in [video_1080p, video_4k] if os.path.exists(v)]
    if not videos:
        print("[!] Proje ana dizininde test videoları bulunamadı!")
        return

    print(f"[*] Bulunan test videoları: {len(videos)} adet")
    for v in videos:
        print(f"    - {os.path.basename(v)}")

    print("\n[*] Modeller yükleniyor...")
    detector = PlateDetector(use_gpu=False)
    ocr = PlateOCR(use_gpu=False)

    engines = [
        (OCR_ENGINE_FASTPLATE, "FastPlateOCR (CCT-S V2)"),
        (OCR_ENGINE_PPOCR, "PP-OCRv6_medium_rec"),
        (OCR_ENGINE_DUAL, "Çift Motor (Dual Mode)")
    ]

    results = []

    print(f"\n[*] Benchmark başlatılıyor (Kare limiti: {args.frames}, Aralık: {args.interval})...\n")

    for vpath in videos:
        vname = os.path.basename(vpath)
        print(f"🎬 Video: {vname}")
        print("-" * 75)

        for eng_code, eng_label in engines:
            print(f"   ► Motor: {eng_label:28s} ... ", end="", flush=True)
            res = run_benchmark(
                vpath, detector, ocr, eng_code,
                max_frames=args.frames,
                frame_interval=args.interval
            )
            results.append(res)
            print(f"Tamamlandı! (FPS: {res['fps']:.1f} | Plaka: {res['unique_valid_plates']} | OCR Gecikme: {res['latency_avg_ms']:.1f}ms)")

        print()

    # Özet Tablo
    print("\n" + "=" * 90)
    print("📊 BENCHMARK SONUÇLARI ÖZET TABLOSU")
    print("=" * 90)
    print(f"{'Video':26s} | {'Motor':16s} | {'FPS':6s} | {'Tespit':7s} | {'Tekil Plaka':12s} | {'Ort. OCR (ms)':14s} | {'p95 (ms)':9s}")
    print("-" * 90)

    for r in results:
        v_short = r['video'][:24]
        eng_short = r['engine']
        print(f"{v_short:26s} | {eng_short:16s} | {r['fps']:6.1f} | {r['yolo_detections']:7d} | {r['unique_valid_plates']:12d} | {r['latency_avg_ms']:14.1f} | {r['latency_p95_ms']:9.1f}")

    print("=" * 90)

    # Bulunan plakalar detayı
    print("\n📋 Bulunan Plakalar:")
    for r in results:
        plates_str = ", ".join(r['plates_list']) if r['plates_list'] else "(Yok)"
        print(f" • [{r['video'][:20]}] [{r['engine']}]: {plates_str}")

    # CSV Raporu
    if args.save_csv:
        rec_dir = os.path.join(base_dir, "records")
        os.makedirs(rec_dir, exist_ok=True)
        csv_path = os.path.join(rec_dir, f"benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")

        with open(csv_path, mode='w', newline='', encoding='utf-8-sig') as f:
            fieldnames = [
                'Video', 'Cozunurluk', 'OCR_Motoru', 'Islenen_Kare',
                'Toplam_Sure_Sn', 'FPS', 'YOLO_Tespit_Sayisi',
                'Tekil_Gecerli_Plaka', 'Plaka_Listesi', 'Ortalama_OCR_ms', 'p95_OCR_ms'
            ]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in results:
                writer.writerow({
                    'Video': r['video'],
                    'Cozunurluk': r['resolution'],
                    'OCR_Motoru': r['engine'],
                    'Islenen_Kare': r['frames_processed'],
                    'Toplam_Sure_Sn': f"{r['total_time_sec']:.2f}",
                    'FPS': f"{r['fps']:.1f}",
                    'YOLO_Tespit_Sayisi': r['yolo_detections'],
                    'Tekil_Gecerli_Plaka': r['unique_valid_plates'],
                    'Plaka_Listesi': "; ".join(r['plates_list']),
                    'Ortalama_OCR_ms': f"{r['latency_avg_ms']:.1f}",
                    'p95_OCR_ms': f"{r['latency_p95_ms']:.1f}"
                })

        print(f"\n[✔] Ayrıntılı benchmark raporu CSV olarak kaydedildi:\n    {csv_path}\n")


if __name__ == '__main__':
    main()
