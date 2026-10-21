import os
import sys
import re
import cv2
import numpy as np
from difflib import SequenceMatcher
from typing import List, Tuple, Dict, Any, Optional
from huggingface_hub import hf_hub_download
from ultralytics import YOLO
try:
    import easyocr
except Exception:
    easyocr = None


def get_resource_path(relative_path: str) -> str:
    """PyInstaller ve normal çalıştırma uyumlu dosya yolu çözümleyici."""
    if hasattr(sys, '_MEIPASS'):
        base_path = sys._MEIPASS
    else:
        base_path = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base_path, relative_path)


def clean_plate_text(text: str) -> str:
    """Plaka metnindeki boşluk, tire ve gereksiz karakterleri temizler ve büyük harfe çevirir."""
    if not text:
        return ""
    # Sadece alfanümerik karakterleri tut
    cleaned = re.sub(r'[^A-Za-z0-9]', '', text).upper()
    # TR karakter dönüştürme (İ -> I, Ğ -> G vs.)
    tr_map = str.maketrans({
        'İ': 'I', 'I': 'I', 'ı': 'I',
        'Ğ': 'G', 'ğ': 'G',
        'Ü': 'U', 'ü': 'U',
        'Ş': 'S', 'ş': 'S',
        'Ö': 'O', 'ö': 'O',
        'Ç': 'C', 'ç': 'C'
    })
    cleaned = cleaned.translate(tr_map)

    # Türk plakalarındaki sol mavi bandın (TR) okunmasından kaynaklanan baştaki TR veya R'yi temizle
    # Örn: TR34ABC123 -> 34ABC123 veya R16BURSA16 -> 16BURSA16
    cleaned = re.sub(r'^(TR|R)(\d{2}[A-Z])', r'\2', cleaned)
    return cleaned


def normalize_plate_text(text: str) -> str:
    """
    Plaka metnindeki harf/rakam karışıklıklarını ve ülke formatlarını
    (Türkiye, İngiltere/UK ve Uluslararası/AB) otomatik akıllı düzeltir.
    """
    if not text:
        return ""

    raw = clean_plate_text(text)
    if len(raw) < 3:
        return raw

    CHAR_TO_DIGIT = {'O': '0', 'I': '1', 'L': '1', 'Z': '2', 'S': '5', 'B': '8', 'G': '6', 'A': '4'}
    CHAR_TO_LETTER = {'0': 'O', '1': 'I', '2': 'Z', '5': 'S', '8': 'B', '6': 'G', '4': 'A'}

    # 1. TÜRKİYE PLAKA FORMATI (01-81 ile başlayan)
    # Örn: 34ABC123, 06A1234
    c0 = CHAR_TO_DIGIT.get(raw[0], raw[0])
    c1 = CHAR_TO_DIGIT.get(raw[1], raw[1])
    if c0.isdigit() and c1.isdigit():
        city_code = int(c0 + c1)
        if 1 <= city_code <= 81:
            rest = raw[2:]
            m_tr = re.match(r'^([A-Z]{1,3})([0-9]{2,4})$', rest)
            if m_tr:
                letters, numbers = m_tr.groups()
                return f"{c0}{c1} {letters} {numbers}"

    # 2. İNGİLTERE (UK) PLAKA FORMATI (2 Harf + 2 Rakam + 3 Harf)
    # Örn: LK60 AEG, FB19 VLR, MF16 JZR, KY16 GUJ
    # Son 3 karakterin en az 2 tanesi harf olmalı (CA45582 gibi sonu rakam olan AB plakalarıyla karışmaması için)
    if len(raw) == 7:
        raw_last3 = raw[4:]
        num_letters = sum(1 for c in raw_last3 if c.isalpha())
        if num_letters >= 2:
            p0 = CHAR_TO_LETTER.get(raw[0], raw[0])
            p1 = CHAR_TO_LETTER.get(raw[1], raw[1])
            p2 = CHAR_TO_DIGIT.get(raw[2], raw[2])
            p3 = CHAR_TO_DIGIT.get(raw[3], raw[3])
            p4 = CHAR_TO_LETTER.get(raw[4], raw[4])
            p5 = CHAR_TO_LETTER.get(raw[5], raw[5])
            p6 = CHAR_TO_LETTER.get(raw[6], raw[6])

            if p0.isalpha() and p1.isalpha() and p2.isdigit() and p3.isdigit() and p4.isalpha() and p5.isalpha() and p6.isalpha():
                return f"{p0}{p1}{p2}{p3} {p4}{p5}{p6}"

    # 3. İKİ HARF + 5 RAKAM (Örn: CA 45582)
    m_eu = re.match(r'^([A-Z]{1,3})([0-9]{3,5})$', raw)
    if m_eu:
        letters, numbers = m_eu.groups()
        return f"{letters} {numbers}"

    # 4. Genel Gruplama
    groups = re.findall(r'[A-Z]+|[0-9]+', raw)
    if len(groups) > 1:
        return " ".join(groups)

    return raw


def format_tr_plate(text: str) -> str:
    """Geriye dönük uyumluluk için normalize_plate_text çağırır."""
    return normalize_plate_text(text)


# ─── Bilinen yanlış pozitif kara liste (reklam, tabela, marka vb.) ─────────
_BLACKLIST_WORDS = frozenset({
    'TAXI', 'TAKSI', 'POLICE', 'POLIS', 'AMBULANCE', 'AMBULANS',
    'BUS', 'OTOBUS', 'SCHOOL', 'OKUL', 'FIRE', 'ITFAIYE',
    'RENT', 'RENTAL', 'KIRALIK', 'SATILIK', 'SALE',
    'STOP', 'EXIT', 'GIRIS', 'CIKIS', 'PARK', 'PARKING',
    'SPEED', 'LIMIT', 'ZONE', 'AREA', 'WARNING', 'DANGER',
    'COCA', 'COLA', 'PEPSI', 'FORD', 'BMW', 'AUDI', 'OPEL',
    'MERCEDES', 'TOYOTA', 'HONDA', 'HYUNDAI', 'RENAULT',
    'FIAT', 'DACIA', 'SKODA', 'VOLVO', 'IVECO', 'SCANIA',
    'IETT', 'EGO', 'ESHOT', 'KENT', 'METRO', 'METROBUS',
    'BOC', 'CIANTY', 'QUALITY', 'SERVICE', 'SERVIS',
    'HOTEL', 'MOTEL', 'MARKET', 'RESTORAN', 'CAFE',
})


def is_valid_plate_format(clean_text: str) -> bool:
    """
    Gerçek plaka formatı doğrulaması:
    Dünyadaki ve Türkiye'deki yasal araç plakalarının tamamı
    HEM harf (en az 1 adet) HEM de rakam (en az 1 adet) içerir.
    Sadece harflerden oluşan kelimeler (reklam, marka logosu, yol tabelası)
    veya sadece rakamlardan oluşan sayılar (otobüs hat no, hız sınırı) elenir.
    Minimum plaka uzunluğu 5 karakterdir.
    Kara listedeki bilinen yanlış pozitif kelimeler de elenir.
    """
    if not clean_text or len(clean_text) < 5 or len(clean_text) > 10:
        return False

    has_letters = any(c.isalpha() for c in clean_text)
    has_digits = any(c.isdigit() for c in clean_text)
    if not (has_letters and has_digits):
        return False

    # Kara liste kontrolü
    upper = clean_text.upper()
    for word in _BLACKLIST_WORDS:
        if word in upper:
            return False

    return True


def similarity_ratio(a: str, b: str) -> float:
    """İki plaka metni arasındaki benzerlik oranını döndürür (0.0 - 1.0)."""
    clean_a = clean_plate_text(a)
    clean_b = clean_plate_text(b)
    if not clean_a or not clean_b:
        return 0.0
    return SequenceMatcher(None, clean_a, clean_b).ratio()


def is_target_match(detected: str, targets: List[str], fuzzy_threshold: float = 0.82) -> Tuple[bool, Optional[str]]:
    """
    Tespit edilen plakanın hedef plaka listesinde olup olmadığını kontrol eder.
    Birebir eşleşme, alt-dize eşleşmesi ve karakter benzerliği (fuzzy) kontrolü yapar.
    """
    clean_det = clean_plate_text(detected)
    if not clean_det or len(clean_det) < 3:
        return False, None

    for target in targets:
        clean_target = clean_plate_text(target)
        if not clean_target:
            continue

        # 1. Birebir tam eşleşme
        if clean_det == clean_target:
            return True, target

        # 2. Biri diğerinin alt dizesi ise (uzaktan okumada başta/sonda 1 harf eksik/fazla olabilir)
        if len(clean_target) >= 4:
            if clean_target in clean_det or clean_det in clean_target:
                # Boyut farkı 2 karakterden fazla değilse
                if abs(len(clean_target) - len(clean_det)) <= 2:
                    return True, target

        # 3. Benzerlik oranı (örn: 0 ile O, 8 ile B gibi OCR benzerlikleri)
        sim = similarity_ratio(clean_det, clean_target)
        if sim >= fuzzy_threshold:
            return True, target

    return False, None


# ─── Akıllı Ön İşleme Motoru ──────────────────────────────────────────────

class SmartPreprocessor:
    """
    Koşullu görüntü ön işleme motoru:
    Plaka kırpıntısının parlaklık, kontrast ve gürültü seviyesini analiz ederek
    gece/gündüz/far yansıması durumlarına göre farklı ön işleme stratejileri uygular.
    """

    @staticmethod
    def analyze_brightness(crop: np.ndarray) -> Dict[str, float]:
        """Kırpıntının parlaklık ve kontrast istatistiklerini hesaplar."""
        if crop is None or crop.size == 0:
            return {'mean': 128.0, 'std': 40.0, 'is_dark': False, 'is_bright': False, 'has_glare': False}

        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if len(crop.shape) == 3 else crop
        mean_val = float(np.mean(gray))
        std_val = float(np.std(gray))
        max_val = float(np.max(gray))
        min_val = float(np.min(gray))

        # Parlaklık bölgeleri
        bright_ratio = float(np.sum(gray > 220)) / max(1, gray.size)
        dark_ratio = float(np.sum(gray < 40)) / max(1, gray.size)

        return {
            'mean': mean_val,
            'std': std_val,
            'max': max_val,
            'min': min_val,
            'is_dark': mean_val < 80,              # Gece / karanlık
            'is_bright': mean_val > 200,            # Aşırı parlak
            'has_glare': bright_ratio > 0.25,       # Far yansıması
            'low_contrast': std_val < 30,           # Düşük kontrast
            'bright_ratio': bright_ratio,
            'dark_ratio': dark_ratio
        }

    @staticmethod
    def preprocess(crop: np.ndarray, target_h: int = 70, min_w: int = 200) -> np.ndarray:
        """
        Akıllı ön işleme: Parlaklık analizine göre en uygun stratejiyi seçer.
        Dönüş: OCR'a hazır 3 kanallı BGR görüntü.
        """
        if crop is None or crop.size == 0:
            return crop

        h, w = crop.shape[:2]
        stats = SmartPreprocessor.analyze_brightness(crop)

        # 1. Boyutlandırma
        scale = target_h / float(max(1, h))
        target_w = max(min_w, int(w * scale))
        resized = cv2.resize(crop, (target_w, target_h), interpolation=cv2.INTER_CUBIC)

        # 2. Kenar dolgusu (padding)
        padded = cv2.copyMakeBorder(resized, 6, 6, 10, 10, cv2.BORDER_REPLICATE)

        # 3. Gri tonlama
        gray = cv2.cvtColor(padded, cv2.COLOR_BGR2GRAY) if len(padded.shape) == 3 else padded

        # 4. Koşullu ön işleme stratejisi
        if stats['is_dark']:
            # GECE MODU: Agresif kontrast + gürültü azaltma + gamma düzeltme
            gamma = 1.8
            inv_gamma = 1.0 / gamma
            table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in range(256)]).astype("uint8")
            gray = cv2.LUT(gray, table)
            gray = cv2.fastNlMeansDenoising(gray, h=8, templateWindowSize=7, searchWindowSize=21)
            clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(4, 4))
            gray = clahe.apply(gray)
        elif stats['has_glare']:
            # FAR YANSIMASI MODU: Adaptif eşikleme + morfolojik operasyonlar
            gray = cv2.normalize(gray, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
            clahe = cv2.createCLAHE(clipLimit=3.5, tileGridSize=(6, 6))
            gray = clahe.apply(gray)
            # Parlak bölgeleri bastır
            _, mask = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            mask = cv2.dilate(mask, kernel, iterations=1)
            inpainted = cv2.inpaint(cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR), mask, 3, cv2.INPAINT_TELEA)
            gray = cv2.cvtColor(inpainted, cv2.COLOR_BGR2GRAY)
        elif stats['low_contrast']:
            # DÜŞÜK KONTRAST MODU: Kontrast germe + CLAHE
            gray = cv2.normalize(gray, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(6, 6))
            gray = clahe.apply(gray)
        else:
            # NORMAL / GÜNDÜZ MODU: Hafif keskinleştirme + standart CLAHE
            gray = cv2.normalize(gray, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
            clahe = cv2.createCLAHE(clipLimit=2.8, tileGridSize=(6, 6))
            gray = clahe.apply(gray)

        # 5. Son yumuşatma (OCR'da kenar parazitini azaltır)
        smoothed = cv2.GaussianBlur(gray, (3, 3), 0)

        # 3 kanallı BGR formatına dönüştür (PP-OCRv6 ve FastPlateOCR uyumlu)
        return cv2.cvtColor(smoothed, cv2.COLOR_GRAY2BGR)


# ─── Araç Tespit Motoru ───────────────────────────────────────────────────

class VehicleDetector:
    """
    YOLOv11 COCO pretrained model ile araç tespiti.
    Desteklenen sınıflar: car(2), motorcycle(3), bus(5), truck(7)
    Plaka ↔ Araç eşleştirmesi yaparak plaka sahipliğini belirler.
    """

    VEHICLE_CLASSES = {2: 'Otomobil', 3: 'Motosiklet', 5: 'Otobüs', 7: 'Kamyon'}

    def __init__(self, use_gpu: bool = False):
        self.model = None
        self.enabled = False
        self._load_model(use_gpu)

    def _load_model(self, use_gpu: bool):
        try:
            print("[VehicleDetector] YOLOv11n COCO modeli yükleniyor...")
            local_model = get_resource_path(os.path.join("models", "yolo11n.pt"))
            if not os.path.exists(local_model):
                local_model = get_resource_path("yolo11n.pt")
            if not os.path.exists(local_model):
                local_model = "yolo11n.pt"

            self.model = YOLO(local_model)
            device = "cuda:0" if use_gpu else "cpu"
            self.model.to(device)
            self.enabled = True
            print(f"[VehicleDetector] Araç tespit modeli hazır! Aygıt: {device}")
        except Exception as e:
            print(f"[VehicleDetector] Model yükleme hatası: {e}")
            self.enabled = False

    def detect_vehicles(self, frame: np.ndarray, conf_threshold: float = 0.40) -> List[Dict[str, Any]]:
        """
        Karede araçları tespit eder.
        Dönüş: [{'bbox': [x1,y1,x2,y2], 'conf': float, 'class_id': int, 'class_name': str}, ...]
        """
        if not self.enabled or self.model is None or frame is None:
            return []

        results = self.model.predict(
            source=frame,
            conf=conf_threshold,
            classes=list(self.VEHICLE_CLASSES.keys()),
            imgsz=640,
            verbose=False
        )

        vehicles = []
        for r in results:
            if r.boxes is None:
                continue
            for box in r.boxes:
                xyxy = box.xyxy[0].cpu().numpy().astype(int)
                conf = float(box.conf[0].cpu().numpy())
                cls_id = int(box.cls[0].cpu().numpy())
                vehicles.append({
                    'bbox': xyxy.tolist(),
                    'conf': conf,
                    'class_id': cls_id,
                    'class_name': self.VEHICLE_CLASSES.get(cls_id, 'Bilinmeyen')
                })
        return vehicles

    @staticmethod
    def match_plate_to_vehicle(plate_bbox: List[int], vehicles: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """
        Plaka kutusunun hangi araca ait olduğunu belirler.
        Kural: Plaka kutusu, araç kutusunun içinde (veya büyük oranda örtüşüyorsa) o araca aittir.
        """
        if not vehicles:
            return None

        px1, py1, px2, py2 = plate_bbox
        plate_cx = (px1 + px2) / 2.0
        plate_cy = (py1 + py2) / 2.0

        best_match = None
        best_overlap = 0.0

        for vehicle in vehicles:
            vx1, vy1, vx2, vy2 = vehicle['bbox']

            # Plaka merkezi araç kutusunun içinde mi?
            if vx1 <= plate_cx <= vx2 and vy1 <= plate_cy <= vy2:
                # Örtüşme oranı
                inter_x1 = max(px1, vx1)
                inter_y1 = max(py1, vy1)
                inter_x2 = min(px2, vx2)
                inter_y2 = min(py2, vy2)
                inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
                plate_area = max(1, (px2 - px1) * (py2 - py1))
                overlap = inter_area / plate_area

                if overlap > best_overlap:
                    best_overlap = overlap
                    best_match = vehicle

        return best_match


# ─── Plaka Tespit Motoru (YOLOv11) ────────────────────────────────────────

class PlateDetector:
    """YOLOv11 tabanlı araç ve plaka tespit motoru."""

    DEFAULT_REPO = "morsetechlab/yolov11-license-plate-detection"
    DEFAULT_WEIGHTS = "license-plate-finetune-v1n.pt"

    def __init__(self, model_path: Optional[str] = None, use_gpu: bool = False):
        self.model = None
        self.model_path = model_path
        self.use_gpu = use_gpu
        self._load_model()

    def _load_model(self):
        try:
            if not self.model_path or not os.path.exists(self.model_path):
                local_path = get_resource_path(os.path.join("models", self.DEFAULT_WEIGHTS))
                if not os.path.exists(local_path):
                    local_path = get_resource_path(self.DEFAULT_WEIGHTS)

                if os.path.exists(local_path):
                    self.model_path = local_path
                else:
                    print(f"[PlateDetector] HuggingFace'ten YOLOv11 plaka modeli kontrol ediliyor: {self.DEFAULT_WEIGHTS}...")
                    self.model_path = hf_hub_download(
                        repo_id=self.DEFAULT_REPO,
                        filename=self.DEFAULT_WEIGHTS
                    )
            print(f"[PlateDetector] YOLOv11 modeli yükleniyor: {self.model_path}")
            self.model = YOLO(self.model_path)
            device = "cuda:0" if self.use_gpu else "cpu"
            self.model.to(device)
            print(f"[PlateDetector] YOLOv11 hazır! Çalışma aygıtı: {device}")
        except Exception as e:
            print(f"[PlateDetector] Model yükleme hatası: {e}")
            raise e

    def detect_plates(self, frame: np.ndarray, conf_threshold: float = 0.35, imgsz: int = 640) -> List[Dict[str, Any]]:
        """
        Gelen video karesinde araç plakalarını tespit eder.
        Dönüş: [{'bbox': [x1, y1, x2, y2], 'conf': float, 'crop': np.ndarray, 'crop_enhanced': np.ndarray}, ...]
        """
        if self.model is None or frame is None:
            return []

        h, w = frame.shape[:2]
        results = self.model.predict(source=frame, conf=conf_threshold, imgsz=imgsz, verbose=False)
        detections = []

        for r in results:
            boxes = r.boxes
            if boxes is None:
                continue

            for box in boxes:
                xyxy = box.xyxy[0].cpu().numpy().astype(int)
                conf = float(box.conf[0].cpu().numpy())

                x1, y1, x2, y2 = xyxy

                # Plaka sınırlarına biraz pay (margin) ekleyerek kırp
                # Uzaktaki küçük plakalarda karakterlerin kenarlardan kesilmesini önler
                box_w = x2 - x1
                box_h = y2 - y1

                # Çok küçük veya anormal boyutları filtrele
                if box_w < 10 or box_h < 5:
                    continue

                pad_x = int(box_w * 0.08)
                pad_y = int(box_h * 0.12)

                crop_x1 = max(0, x1 - pad_x)
                crop_y1 = max(0, y1 - pad_y)
                crop_x2 = min(w, x2 + pad_x)
                crop_y2 = min(h, y2 + pad_y)

                crop = frame[crop_y1:crop_y2, crop_x1:crop_x2]
                if crop.size == 0:
                    continue

                detections.append({
                    'bbox': [x1, y1, x2, y2],
                    'conf': conf,
                    'crop': crop
                })

        return detections


# ─── OCR Motor Seçenekleri ─────────────────────────────────────────────────

# OCR motor sabitleri
OCR_ENGINE_PPOCR = "ppocr"
OCR_ENGINE_FASTPLATE = "fastplate"
OCR_ENGINE_DUAL = "dual"


class PlateOCR:
    """
    Çoklu OCR Motor Yöneticisi:
    - PP-OCRv6_medium_rec (ONNX Runtime) — Yüksek doğruluk, Çince/Türkçe/Latin desteği
    - FastPlateOCR (CCT-S V2 Global) — Ultra hızlı, 65+ ülke plaka formatı
    - Çift Motor (Dual) — İki motoru da çalıştırıp en iyi sonucu seçer
    - EasyOCR — Yedek (fallback) motor

    Kullanıcı ayarlardan motor değiştirebilir (UI'da seçenek).
    """

    def __init__(self, use_gpu: bool = False, engine: str = OCR_ENGINE_FASTPLATE):
        self.predictor = None           # PP-OCRv6
        self.fast_recognizer = None     # FastPlateOCR
        self.easy_reader = None         # EasyOCR (yedek)
        self.preprocessor = SmartPreprocessor()
        self.engine = engine

        # OCR gecikme metrikleri
        self._ocr_times: List[float] = []
        self._max_ocr_history = 100

        # 1. PP-OCRv6_medium_rec (ONNX Runtime)
        print("[PlateOCR] PP-OCRv6_medium_rec (ONNXRuntime) başlatılıyor...")
        try:
            os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"
            from paddlex.inference.models import create_predictor
            self.predictor = create_predictor(model_name="PP-OCRv6_medium_rec", engine="onnxruntime")
            print("[PlateOCR] PP-OCRv6_medium_rec (ONNXRuntime) hazır!")
        except Exception as e:
            print(f"[PlateOCR] PP-OCRv6 başlatılamadı: {e}")

        # 2. FastPlateOCR (CCT-S V2 Global)
        print("[PlateOCR] FastPlateOCR (CCT-S V2 Global) başlatılıyor...")
        try:
            from fast_plate_ocr import LicensePlateRecognizer
            local_onnx = get_resource_path(os.path.join("models", "cct_s_v2_global.onnx"))
            local_yaml = get_resource_path(os.path.join("models", "cct_s_v2_global_plate_config.yaml"))
            if os.path.exists(local_onnx) and os.path.exists(local_yaml):
                self.fast_recognizer = LicensePlateRecognizer(
                    onnx_model_path=local_onnx,
                    plate_config_path=local_yaml
                )
            else:
                self.fast_recognizer = LicensePlateRecognizer("cct-s-v2-global-model")
            print("[PlateOCR] FastPlateOCR (CCT-S V2 Global) hazır!")
        except Exception as e:
            print(f"[PlateOCR] FastPlateOCR başlatılamadı: {e}")

        # 3. İkincil yedek (Fallback) olarak EasyOCR
        if easyocr is not None:
            try:
                self.easy_reader = easyocr.Reader(['en'], gpu=use_gpu)
                print("[PlateOCR] EasyOCR yedek olarak hazır!")
            except Exception as e:
                print(f"[PlateOCR] EasyOCR yedek başlatılamadı: {e}")
                self.easy_reader = None
        else:
            self.easy_reader = None


    def set_engine(self, engine: str):
        """OCR motorunu değiştirir. Değerler: 'ppocr', 'fastplate', 'dual'"""
        if engine in (OCR_ENGINE_PPOCR, OCR_ENGINE_FASTPLATE, OCR_ENGINE_DUAL):
            self.engine = engine
            print(f"[PlateOCR] OCR motoru değiştirildi: {engine}")

    def get_engine_name(self) -> str:
        """Aktif OCR motor adını döndürür."""
        names = {
            OCR_ENGINE_PPOCR: "PP-OCRv6",
            OCR_ENGINE_FASTPLATE: "FastPlateOCR",
            OCR_ENGINE_DUAL: "Çift Motor (En İyi)"
        }
        return names.get(self.engine, self.engine)

    def get_latency_stats(self) -> Dict[str, float]:
        """OCR gecikme istatistiklerini döndürür (ms cinsinden)."""
        if not self._ocr_times:
            return {'avg': 0.0, 'median': 0.0, 'p95': 0.0, 'count': 0}
        times = self._ocr_times[-self._max_ocr_history:]
        sorted_t = sorted(times)
        return {
            'avg': float(np.mean(sorted_t)),
            'median': float(np.median(sorted_t)),
            'p95': float(sorted_t[int(len(sorted_t) * 0.95)]) if len(sorted_t) >= 2 else float(sorted_t[-1]),
            'count': len(sorted_t)
        }

    def _record_latency(self, ms: float):
        """OCR gecikme süresini kaydeder."""
        self._ocr_times.append(ms)
        if len(self._ocr_times) > self._max_ocr_history * 2:
            self._ocr_times = self._ocr_times[-self._max_ocr_history:]

    def enhance_crop_for_ocr(self, crop: np.ndarray) -> np.ndarray:
        """
        Plakayı OCR için netleştirir (SmartPreprocessor kullanır):
        Parlaklık analizi → koşullu strateji (gece/gündüz/yansıma)
        """
        return self.preprocessor.preprocess(crop)

    def _read_ppocr(self, crop: np.ndarray) -> Tuple[str, float]:
        """PP-OCRv6 ile plaka okur."""
        if self.predictor is None:
            return "", 0.0

        try:
            # 1. Doğrudan okuma
            preds = list(self.predictor(crop))
            if preds:
                for r in preds:
                    raw_text = r.get('rec_text', '')
                    conf = float(r.get('rec_score', 0.0))
                    clean = clean_plate_text(raw_text)
                    if clean and len(clean) >= 3 and conf >= 0.25:
                        return format_tr_plate(clean), conf

            # 2. Akıllı ön işleme ile ikinci geçiş
            enhanced = self.enhance_crop_for_ocr(crop)
            preds = list(self.predictor(enhanced))
            if preds:
                for r in preds:
                    raw_text = r.get('rec_text', '')
                    conf = float(r.get('rec_score', 0.0))
                    clean = clean_plate_text(raw_text)
                    if clean and len(clean) >= 3 and conf >= 0.25:
                        return format_tr_plate(clean), conf
        except Exception as e:
            print(f"[PlateOCR:PP-OCR] Hata: {e}")

        return "", 0.0

    def _read_fastplate(self, crop: np.ndarray) -> Tuple[str, float]:
        """FastPlateOCR (CCT-S V2) ile plaka okur."""
        if self.fast_recognizer is None:
            return "", 0.0

        try:
            # BGR → RGB (FastPlateOCR RGB bekler)
            crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB) if len(crop.shape) == 3 else crop

            # 1. Doğrudan okuma
            preds = self.fast_recognizer.run(crop_rgb, return_confidence=True)
            if preds:
                p = preds[0]
                if p.char_probs is not None:
                    avg_conf = float(np.mean(p.char_probs))
                    min_conf = float(np.min(p.char_probs))
                else:
                    avg_conf = 0.0
                    min_conf = 0.0

                clean = clean_plate_text(p.plate)
                if clean and len(clean) >= 3 and avg_conf >= 0.20:
                    return format_tr_plate(clean), avg_conf

            # 2. Akıllı ön işleme ile ikinci geçiş
            enhanced = self.enhance_crop_for_ocr(crop)
            enhanced_rgb = cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB)
            preds = self.fast_recognizer.run(enhanced_rgb, return_confidence=True)
            if preds:
                p = preds[0]
                if p.char_probs is not None:
                    avg_conf = float(np.mean(p.char_probs))
                else:
                    avg_conf = 0.0

                clean = clean_plate_text(p.plate)
                if clean and len(clean) >= 3 and avg_conf >= 0.20:
                    return format_tr_plate(clean), avg_conf

        except Exception as e:
            print(f"[PlateOCR:FastPlate] Hata: {e}")

        return "", 0.0

    def _read_easyocr(self, crop: np.ndarray) -> Tuple[str, float]:
        """EasyOCR ile plaka okur (yedek motor)."""
        if self.easy_reader is None:
            return "", 0.0

        try:
            enhanced = self.enhance_crop_for_ocr(crop)
            results = self.easy_reader.readtext(
                enhanced,
                allowlist='0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ',
                detail=1,
                paragraph=False,
                low_text=0.25,
                text_threshold=0.35,
                link_threshold=0.30
            )
            if results:
                texts = []
                confs = []
                for item in results:
                    _, text, conf = item
                    clean = clean_plate_text(text)
                    if clean:
                        texts.append(clean)
                        confs.append(float(conf))
                full_text = "".join(texts)
                avg_conf = float(np.mean(confs)) if confs else 0.0
                if len(full_text) >= 3:
                    return format_tr_plate(full_text), avg_conf
        except Exception as e:
            print(f"[PlateOCR:EasyOCR] Hata: {e}")

        return "", 0.0

    def read_plate(self, crop: np.ndarray) -> Tuple[str, float]:
        """
        Kırpılmış plaka görselini okur ve (temiz_plaka_metni, güven_skoru) döndürür.
        Seçilen OCR motoruna göre okuma yapar; Dual modda en iyi sonucu seçer.
        """
        if crop is None or crop.size == 0:
            return "", 0.0

        import time
        t0 = time.perf_counter()

        try:
            if self.engine == OCR_ENGINE_PPOCR:
                text, conf = self._read_ppocr(crop)
                if text and conf > 0:
                    return text, conf
                # Yedek: EasyOCR
                return self._read_easyocr(crop)

            elif self.engine == OCR_ENGINE_FASTPLATE:
                text, conf = self._read_fastplate(crop)
                if text and conf > 0:
                    return text, conf
                # Yedek: EasyOCR
                return self._read_easyocr(crop)

            elif self.engine == OCR_ENGINE_DUAL:
                # İki motoru da çalıştır, en iyi sonucu seç
                ppocr_text, ppocr_conf = self._read_ppocr(crop)
                fast_text, fast_conf = self._read_fastplate(crop)

                # Sonuç kalite puanı: güven × karakter sayısı / 7 × format bonusu
                def quality_score(text: str, conf: float) -> float:
                    if not text or conf <= 0:
                        return 0.0
                    clean = clean_plate_text(text)
                    if not is_valid_plate_format(clean):
                        return conf * 0.3  # Geçersiz format → düşük puan
                    length_factor = min(1.0, len(clean) / 7.0)
                    format_bonus = 1.2 if len(clean) == 7 else 1.0
                    return conf * length_factor * format_bonus

                ppocr_score = quality_score(ppocr_text, ppocr_conf)
                fast_score = quality_score(fast_text, fast_conf)

                if ppocr_score >= fast_score and ppocr_text:
                    return ppocr_text, ppocr_conf
                elif fast_text:
                    return fast_text, fast_conf
                elif ppocr_text:
                    return ppocr_text, ppocr_conf

                # Hiçbiri bulamadıysa EasyOCR yedek
                return self._read_easyocr(crop)

            return "", 0.0

        except Exception as e:
            print(f"[PlateOCR] OCR okuma hatası: {e}")
            return "", 0.0

        finally:
            elapsed_ms = (time.perf_counter() - t0) * 1000
            self._record_latency(elapsed_ms)
