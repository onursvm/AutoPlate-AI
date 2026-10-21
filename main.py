import sys
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from detector import PlateDetector, PlateOCR, VehicleDetector
from app_ui import AutoPlateApp



def main():
    print("=" * 60)
    print("⚡ AutoPlate AI - YOLOv11 & FastPlateOCR + PP-OCRv6 Plaka Tanıma Sistemi")
    print("=" * 60)
    print("[1/3] YOLOv11 Plaka Tespit Modeli Yükleniyor...")
    detector = PlateDetector(use_gpu=False)

    print("[2/3] OCR Motorları (FastPlateOCR & PP-OCRv6) Hazırlanıyor...")
    ocr = PlateOCR(use_gpu=False)

    print("[3/3] YOLOv11 Araç Tespit Modeli Hazırlanıyor...")
    try:
        vehicle_detector = VehicleDetector(use_gpu=False)
    except Exception as e:
        print(f"[!] Araç tespit modeli yüklenemedi: {e}")
        vehicle_detector = None

    print("\n[✔] Tüm yapay zeka modelleri hazır! Arayüz başlatılıyor...")
    app = AutoPlateApp(detector, ocr, vehicle_detector=vehicle_detector)
    app.mainloop()


if __name__ == '__main__':
    main()
