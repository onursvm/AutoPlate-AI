import os
import re
import csv
import cv2
import winsound
from datetime import datetime
from typing import Dict, Any, List, Optional


class PlateHistoryManager:
    """Tespit edilen plakaları kaydeden, alarm veren ve CSV raporu oluşturan yönetici sınıf."""

    def __init__(self, base_dir: str = "records"):
        self.base_dir = base_dir
        self.detections_dir = os.path.join(self.base_dir, "detections")
        self.alerts_dir = os.path.join(self.base_dir, "alerts")
        os.makedirs(self.detections_dir, exist_ok=True)
        os.makedirs(self.alerts_dir, exist_ok=True)

        self.records: List[Dict[str, Any]] = []
        self.sound_enabled: bool = True

    def set_sound_enabled(self, enabled: bool):
        self.sound_enabled = enabled

    def trigger_sound_alert(self):
        """Windows sistem alarm sesi çalar."""
        if not self.sound_enabled:
            return
        try:
            # Dikkat çekici çift tonlu uyarı sesi (Hz, ms)
            winsound.Beep(1400, 160)
            winsound.Beep(1800, 220)
        except Exception:
            try:
                winsound.MessageBeep(winsound.MB_ICONEXCLAMATION)
            except Exception:
                pass

    def add_record(self, event_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Tespit edilen plakayı kaydeder, görselini diske yazar ve kayıt objesini döndürür.
        """
        now = datetime.now()
        timestamp_str = now.strftime("%Y-%m-%d %H:%M:%S")
        file_prefix = now.strftime("%Y%m%d_%H%M%S_%f")[:19]

        clean_name = re.sub(r'[^A-Za-z0-9_]', '', event_data['text'].replace(" ", "_")) or "plaka"
        is_match = event_data.get('is_match', False)

        save_folder = self.alerts_dir if is_match else self.detections_dir

        crop_filename = f"{file_prefix}_{clean_name}_crop.jpg"
        crop_path = os.path.join(save_folder, crop_filename)

        # Kırpılmış plaka görselini kaydet
        crop_img = event_data.get('crop')
        if crop_img is not None and crop_img.size > 0:
            cv2.imwrite(crop_path, crop_img)
        else:
            crop_path = ""

        # Hedef eşleşmesinde tam kare görselini de kaydet (delil amaçlı)
        full_path = ""
        if is_match and 'full_frame' in event_data:
            full_frame = event_data['full_frame']
            if full_frame is not None and full_frame.size > 0:
                full_filename = f"{file_prefix}_{clean_name}_full.jpg"
                full_path = os.path.join(save_folder, full_filename)
                cv2.imwrite(full_path, full_frame)

        record_entry = {
            'timestamp': timestamp_str,
            'time_only': event_data.get('time_str', now.strftime("%H:%M:%S")),
            'plate': event_data['text'],
            'conf': event_data.get('conf', 0.0),
            'yolo_conf': event_data.get('yolo_conf', 0.0),
            'is_match': is_match,
            'target_query': event_data.get('target_query', ''),
            'crop_path': crop_path,
            'full_path': full_path,
            'crop_img': crop_img,
            'vehicle_type': event_data.get('vehicle_type'),
            'vehicle_conf': event_data.get('vehicle_conf')
        }

        self.records.append(record_entry)

        if is_match:
            self.trigger_sound_alert()

        return record_entry

    def export_csv(self, file_path: Optional[str] = None) -> str:
        """Kayıtları CSV formatında dışa aktarır."""
        if not file_path:
            date_str = datetime.now().strftime("%Y%m%d_%H%M%S")
            file_path = os.path.join(self.base_dir, f"plaka_kayitlari_{date_str}.csv")

        fieldnames = [
            'Tarih/Saat', 'Plaka No', 'OCR Guven Skoru', 'YOLO Guven Skoru',
            'Hedef Eslesti Mi', 'Hedef Arama Metni', 'Plaka Gorseli Yolu', 'Tam Kare Gorseli Yolu',
            'Arac Tipi'
        ]

        with open(file_path, mode='w', newline='', encoding='utf-8-sig') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in self.records:
                writer.writerow({
                    'Tarih/Saat': r['timestamp'],
                    'Plaka No': r['plate'],
                    'OCR Guven Skoru': f"%{int(r['conf'] * 100)}",
                    'YOLO Guven Skoru': f"%{int(r['yolo_conf'] * 100)}",
                    'Hedef Eslesti Mi': 'EVET (HEDEF ALARM)' if r['is_match'] else 'HAYIR',
                    'Hedef Arama Metni': r['target_query'] or '-',
                    'Plaka Gorseli Yolu': r['crop_path'],
                    'Tam Kare Gorseli Yolu': r['full_path'],
                    'Arac Tipi': r.get('vehicle_type') or '-'
                })

        return file_path
