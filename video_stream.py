import os
import time
import cv2
import threading
import numpy as np
from datetime import datetime
from typing import List, Dict, Any, Optional, Callable

from detector import (
    PlateDetector, PlateOCR, VehicleDetector,
    is_target_match, clean_plate_text,
    similarity_ratio, is_valid_plate_format, normalize_plate_text
)

# Varsayılan minimum OCR güven eşiği (%35 dengeli; hem gece hem gündüz çalışır)
DEFAULT_MIN_OCR_CONFIDENCE = 0.35


def calculate_iou(boxA, boxB) -> float:
    """İki sınır kutusu (bbox) arasındaki Intersection over Union (IoU) oranını hesaplar."""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    denom = float(boxAArea + boxBArea - interArea)
    if denom <= 0:
        return 0.0
    return interArea / denom


def box_distance_and_iou(boxA, boxB):
    """
    İki plaka kutusunun aynı araca ait olup olmadığını hem örtüşme (IoU)
    hem de merkez noktası mesafesi ile hesaplar.
    4K videolarda veya hızlı hareket eden araçlarda takip kaybını önler.
    """
    iou = calculate_iou(boxA, boxB)

    # Merkez noktaları
    cA_x = (boxA[0] + boxA[2]) / 2.0
    cA_y = (boxA[1] + boxA[3]) / 2.0
    cB_x = (boxB[0] + boxB[2]) / 2.0
    cB_y = (boxB[1] + boxB[3]) / 2.0

    dist = np.hypot(cA_x - cB_x, cA_y - cB_y)
    diagA = np.hypot(boxA[2] - boxA[0], boxA[3] - boxA[1])

    # IoU > 0.12 veya merkezler plakanın 2.8 katı mesafesindeyse aynı araçtır (4K hızlı araçları korur)
    is_same = (iou > 0.12) or (dist < max(150, diagA * 2.8))
    return is_same, iou, dist


def is_recent_flicker_duplicate(clean_text: str, recent_dict: Dict[str, float], now: float, cooldown: float = 4.0) -> bool:
    """
    Sadece anlık takip kopmalarını (örn. 1-2 saniye direk/ağaç arkasında kalma) engeller.
    - Kadrajda olduğu sürece araç zaten YOLO takibi (tdata['logged']) ile yalnızca 1 kez loglanır.
    - Araç kadrajdan tamamen çıkıp dönüp tekrar gelirse (>4 saniye sonra) YENİDEN OKUNUR.
    - Farklı araçlar (farklı plakalar) ASLA birbirini engellemez.
    """
    if not clean_text or len(clean_text) < 3 or clean_text.startswith("PLAKA"):
        return False

    for logged_clean, logged_time in list(recent_dict.items()):
        if now - logged_time < cooldown:
            # Sadece birebir aynı plaka veya %90+ çok yüksek benzerlik (flicker koruması)
            if clean_text == logged_clean:
                return True
            if len(clean_text) == len(logged_clean) and similarity_ratio(clean_text, logged_clean) >= 0.90:
                return True
    return False


class VideoStreamEngine:
    """
    Çift İş Parçacıklı (Dual-Thread) Akıllı Video Motoru:
    1. Oynatma İş Parçacığı (Playback Thread): Videoyu gecikmesiz, tam 1.0x hızında ve 0 donma ile oynatır.
    2. Yapay Zeka İş Parçacığı (AI Worker Thread): Arka planda YOLO ve OCR'ı çalıştırır.
       4K ve Gece/Zorlu ışık şartlarına otomatik adapte olur.
    """

    def __init__(self, detector: PlateDetector, ocr: PlateOCR):
        self.detector = detector
        self.ocr = ocr
        self.vehicle_detector: Optional[VehicleDetector] = None  # Opsiyonel araç tespiti
        self.vehicle_detection_enabled: bool = False

        self.source_type: str = "file"  # "file", "webcam", "ip"
        self.source_path: Any = ""
        self.target_plates: List[str] = []

        self.is_running: bool = False
        self.is_paused: bool = False
        self.loop_video: bool = True
        self.conf_threshold: float = 0.35
        self.imgsz: int = 640
        self.effective_imgsz: int = 640
        self.frame_skip: int = 1
        self.fuzzy_threshold: float = 0.82
        self.min_ocr_conf: float = DEFAULT_MIN_OCR_CONFIDENCE
        self.log_unreadable: bool = True

        # 4K Optimizasyon: ROI (Region of Interest) – üst bölge atla
        self.use_roi: bool = True  # 4K'da otomatik aktif
        self.roi_top_skip: float = 0.25  # Üst %25 atla (gökyüzü/bina)

        # Threading
        self.playback_thread: Optional[threading.Thread] = None
        self.ai_thread: Optional[threading.Thread] = None

        # Callbacks
        self.callback_frame: Optional[Callable[[np.ndarray, np.ndarray], None]] = None
        self.callback_plate: Optional[Callable[[Dict[str, Any]], None]] = None
        self.callback_alert: Optional[Callable[[Dict[str, Any]], None]] = None
        self.callback_stats: Optional[Callable[[float, int, int], None]] = None
        self.callback_error: Optional[Callable[[str], None]] = None
        self.callback_completed: Optional[Callable[[], None]] = None

        # Takip & Tekrar bildirim önleme hafızası
        self.active_tracks: Dict[int, Dict[str, Any]] = {}
        self.display_tracks: List[Dict[str, Any]] = []  # Kilit beklemeden anında çizim için atomik liste
        self.next_track_id: int = 0
        self.alerted_plates = set()
        self.recent_logged_plates: Dict[str, float] = {}  # clean_text -> last_logged_time

        # Thread senkronizasyon araçları
        self.track_lock = threading.Lock()
        self.ai_event = threading.Event()
        self.latest_frame_for_ai: Optional[np.ndarray] = None
        self.ai_busy = False

    def set_source(self, source_type: str, source_path: Any):
        self.source_type = source_type
        self.source_path = source_path

    def set_targets(self, targets: List[str]):
        with self.track_lock:
            self.target_plates = [t.strip().upper() for t in targets if t.strip()]

    def set_vehicle_detection(self, enabled: bool, detector: Optional[VehicleDetector] = None):
        """Araç tespitini açar/kapatır."""
        self.vehicle_detection_enabled = enabled
        if detector is not None:
            self.vehicle_detector = detector

    def set_settings(self, conf: float, imgsz: int, frame_skip: int, min_ocr_conf: float = DEFAULT_MIN_OCR_CONFIDENCE, log_unreadable: bool = True):
        with self.track_lock:
            self.conf_threshold = conf
            self.imgsz = imgsz
            self.frame_skip = max(1, frame_skip)
            self.min_ocr_conf = min_ocr_conf
            self.log_unreadable = log_unreadable

    def clear_tracking_history(self):
        """Arayüzde 'Temizle' butonuna basıldığında motorun tüm takip ve kayıt hafızasını sıfırlar."""
        with self.track_lock:
            self.recent_logged_plates.clear()
            self.alerted_plates.clear()
            for tdata in self.active_tracks.values():
                tdata['logged'] = False
                tdata['vote_counts'] = {}
                tdata['vote_raw_counts'] = {}

    def start(self):
        if self.is_running:
            return
        self.is_running = True
        self.is_paused = False
        self.ai_busy = False
        self.latest_frame_for_ai = None
        self.ai_event.clear()

        # 1. Oynatma Thread'i (Video akışını kesintisiz 1.0x hızda tutar)
        self.playback_thread = threading.Thread(target=self._run_playback_loop, daemon=True)
        self.playback_thread.start()

        # 2. AI Worker Thread'i (YOLO ve OCR arka planda çalışır, videoyu asla dondurmaz)
        self.ai_thread = threading.Thread(target=self._run_ai_worker, daemon=True)
        self.ai_thread.start()

    def pause(self):
        self.is_paused = True

    def resume(self):
        self.is_paused = False

    def toggle_pause(self):
        self.is_paused = not self.is_paused

    def stop(self):
        self.is_running = False
        self.ai_event.set()
        if self.playback_thread and self.playback_thread.is_alive():
            self.playback_thread.join(timeout=1.5)
        if self.ai_thread and self.ai_thread.is_alive():
            self.ai_thread.join(timeout=1.5)
        self.playback_thread = None
        self.ai_thread = None

    def _run_playback_loop(self):
        """
        Videonun tam normal hızında (1.0x) ve SIFIR DONMA ile akmasını sağlayan
        yüksek performanslı oynatıcı döngüsü.
        """
        cap = None
        try:
            if self.source_type == "webcam":
                try:
                    cam_id = int(self.source_path)
                except ValueError:
                    cam_id = 0
                cap = cv2.VideoCapture(cam_id, cv2.CAP_DSHOW)
            elif self.source_type == "ip":
                os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"
                cap = cv2.VideoCapture(str(self.source_path))
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            else:
                cap = cv2.VideoCapture(str(self.source_path))

            if not cap or not cap.isOpened():
                if self.callback_error:
                    self.callback_error(f"Görüntü kaynağı açılamadı:\n{self.source_path}")
                self.is_running = False
                return

            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if self.source_type == "file" else 0
            fps_target = cap.get(cv2.CAP_PROP_FPS)
            if fps_target <= 0 or np.isnan(fps_target):
                fps_target = 30.0

            # 4K veya Ultra-HD video kontrolü (Otomatik model boyutu optimizasyonu)
            cap_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            with self.track_lock:
                if cap_w >= 2500 and (self.imgsz == 640 or self.imgsz == 0):
                    self.effective_imgsz = 1280
                else:
                    self.effective_imgsz = self.imgsz if self.imgsz > 0 else 640

            frame_delay = 1.0 / fps_target
            frame_count = 0
            fps_timer = time.time()
            fps_calc_counter = 0
            current_fps = 0.0

            # Gerçek saat referansı
            playback_start_time = time.time()

            while self.is_running:
                if self.is_paused:
                    time.sleep(0.04)
                    playback_start_time = time.time() - (frame_count * frame_delay)
                    continue

                ret, frame = cap.read()

                if not ret:
                    if self.source_type == "file" and self.loop_video:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        frame_count = 0
                        playback_start_time = time.time()
                        continue
                    else:
                        if self.callback_completed:
                            self.callback_completed()
                        break

                frame_count += 1
                fps_calc_counter += 1

                now = time.time()
                if now - fps_timer >= 1.0:
                    current_fps = fps_calc_counter / (now - fps_timer)
                    fps_calc_counter = 0
                    fps_timer = now

                # AI Worker boşta ise bu kareyi arka plandaki yapay zekaya devret
                if not self.ai_busy and self.latest_frame_for_ai is None:
                    self.latest_frame_for_ai = frame.copy()
                    self.ai_event.set()

                # Kare üzerine mevcut plaka kutularını çiz (AI'ı beklemeden anında çizer)
                annotated_frame = self._draw_annotations(frame)
                rgb_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)

                if self.callback_frame:
                    self.callback_frame(rgb_frame, frame)

                # İstatistikleri her karede değil, saniyede ~5-6 kez güncelle (arayüzde titremeyi ve kasılmayı önler)
                if self.callback_stats and (frame_count % 5 == 0 or frame_count == total_frames):
                    self.callback_stats(current_fps, frame_count, total_frames)

                # Video dosyaları için tam 1.0x normal hız sınırlaması
                if self.source_type == "file":
                    expected_time = frame_count * frame_delay
                    elapsed_time = time.time() - playback_start_time
                    time_to_sleep = expected_time - elapsed_time
                    if time_to_sleep > 0.002:
                        time.sleep(time_to_sleep)

        except Exception as e:
            if self.callback_error:
                self.callback_error(f"Oynatma hatası: {str(e)}")
        finally:
            if cap:
                cap.release()
            self.is_running = False

    def _run_ai_worker(self):
        """
        Yapay Zeka İş Parçacığı:
        Arka planda bağımsız çalışır. YOLO ve OCR burada koşar.
        Ana video oynatmayı asla yavaşlatmaz veya dondurmaz.
        """
        while self.is_running:
            self.ai_event.wait(timeout=0.1)
            if not self.is_running:
                break

            if self.latest_frame_for_ai is None:
                self.ai_event.clear()
                continue

            self.ai_busy = True
            frame_to_process = self.latest_frame_for_ai
            self.latest_frame_for_ai = None
            self.ai_event.clear()

            try:
                with self.track_lock:
                    conf = self.conf_threshold
                    imgsz = getattr(self, 'effective_imgsz', self.imgsz)

                frame_h, frame_w = frame_to_process.shape[:2]

                # 4K Optimizasyon: AI worker'a gönderilen kareyi küçült
                ai_frame = frame_to_process
                scale_factor = 1.0
                if frame_w > 2500:
                    scale_factor = 1920.0 / frame_w
                    new_w = 1920
                    new_h = int(frame_h * scale_factor)
                    ai_frame = cv2.resize(frame_to_process, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

                # 1. YOLOv11 Plaka Tespiti
                detected_items = self.detector.detect_plates(
                    ai_frame,
                    conf_threshold=conf,
                    imgsz=imgsz
                )

                # 4K'da bbox koordinatlarını orijinal boyuta geri ölçekle
                if scale_factor < 1.0:
                    inv_scale = 1.0 / scale_factor
                    for item in detected_items:
                        item['bbox'] = [int(c * inv_scale) for c in item['bbox']]
                        # Crop'u orijinal çözünürlüklü kareden al (daha net OCR)
                        x1, y1, x2, y2 = item['bbox']
                        bw = x2 - x1
                        bh = y2 - y1
                        px = int(bw * 0.08)
                        py = int(bh * 0.12)
                        cx1 = max(0, x1 - px)
                        cy1 = max(0, y1 - py)
                        cx2 = min(frame_w, x2 + px)
                        cy2 = min(frame_h, y2 + py)
                        crop = frame_to_process[cy1:cy2, cx1:cx2]
                        if crop.size > 0:
                            item['crop'] = crop

                # 2. Araç Tespiti (opsiyonel)
                vehicles = []
                if self.vehicle_detection_enabled and self.vehicle_detector is not None:
                    vehicles = self.vehicle_detector.detect_vehicles(ai_frame)
                    if scale_factor < 1.0:
                        inv_scale = 1.0 / scale_factor
                        for v in vehicles:
                            v['bbox'] = [int(c * inv_scale) for c in v['bbox']]

                # 3. Plaka Takip & Tekil OCR Okuma
                self._process_tracked_plates(frame_to_process, detected_items, vehicles)

            except Exception as e:
                print(f"[AI Worker Error]: {e}")
            finally:
                self.ai_busy = False

    def _process_tracked_plates(self, frame: np.ndarray, detected_items: List[Dict[str, Any]], vehicles: Optional[List[Dict[str, Any]]] = None):
        """
        Çoklu Kare En İyi Aday (Best-Candidate Multi-Frame) & Konsensüs Takip Motoru:
        - Uzaktaki (küçük) plakalarda erken ve hatalı okuma yapmaz.
        - Araç kameraya yaklaşırken her karede OCR adaylarını değerlendirir.
        - Sadece geçerli plaka formatına (hem harf hem rakam içeren) sahip okumaları kabul eder;
          reklam yazıları, tabela ve çöp kelimeleri (BOC, 101, CIANTY vs.) otomatik eler.
        - Araç için en yüksek puanlı ve en net okumayı tekil olarak kaydeder.
        - Confidence-weighted voting ile konsensüs sağlar.
        """
        now = time.time()
        updated_tracks = {}
        frame_h, frame_w = frame.shape[:2]
        min_crop_w = 85 if frame_w >= 2500 else 50

        # 1.5 saniyeden uzun süredir görülmeyen (ekrandan çıkan) araçlar
        expired_tracks = []
        for tid, tdata in list(self.active_tracks.items()):
            if now - tdata['last_seen'] < 1.5:
                updated_tracks[tid] = tdata
            else:
                expired_tracks.append(tdata)

        # 5 saniyeden eski loglanmış plakaları temizle (böylece geri dönen araçlar tekrar okunabilir)
        for p_clean, log_t in list(self.recent_logged_plates.items()):
            if now - log_t > 5.0:
                del self.recent_logged_plates[p_clean]

        # Ekrandan çıkan araçlar için son kontrol (Eğer henüz loglanmadıysa en iyi adayı kaydet)
        for tdata in expired_tracks:
            if not tdata.get('logged', False):
                best_clean = clean_plate_text(tdata.get('best_valid_text', ''))
                # Konsensüs kontrolü: En az 2 karede aynı plaka okunmuş olmalı
                votes = tdata.get('vote_counts', {})
                if votes:
                    top_vote = max(votes.values())
                else:
                    top_vote = 0

                if len(best_clean) >= 6 and tdata.get('best_ocr_conf', 0) >= 0.70 and top_vote >= 2:
                    tdata['text'] = tdata['best_valid_text']
                    tdata['ocr_conf'] = tdata['best_ocr_conf']
                    tdata['crop'] = tdata['best_crop']
                    self._check_target_and_emit(tdata, tdata['best_bbox'], tdata.get('best_full_frame', frame))
                elif len(best_clean) >= 6 and tdata.get('best_ocr_conf', 0) >= 0.85:
                    # Çok yüksek güvenle tek karede bile kabul et
                    tdata['text'] = tdata['best_valid_text']
                    tdata['ocr_conf'] = tdata['best_ocr_conf']
                    tdata['crop'] = tdata['best_crop']
                    self._check_target_and_emit(tdata, tdata['best_bbox'], tdata.get('best_full_frame', frame))
                elif self.log_unreadable and tdata.get('seen_count', 0) >= 15 and tdata.get('max_crop_w', 0) >= min_crop_w:
                    tdata['text'] = f"Plaka (%{int(tdata['conf']*100)})"
                    tdata['ocr_conf'] = 0.0
                    self._check_target_and_emit(tdata, tdata['bbox'], frame)

        for item in detected_items:
            bbox = item['bbox']
            conf = item['conf']
            crop = item['crop']
            crop_h, crop_w = crop.shape[:2]

            matched_tid = None
            best_match_score = -1.0

            # Mevcut araçlarla eşleştir
            for tid, tdata in updated_tracks.items():
                is_same, iou, dist = box_distance_and_iou(bbox, tdata['bbox'])
                if is_same:
                    score = iou - (dist / 2500.0)
                    if score > best_match_score:
                        best_match_score = score
                        matched_tid = tid

            if matched_tid is not None:
                # EKRANDA ZATEN TAKİP EDİLEN MEVCUT BİR ARAÇ
                tdata = updated_tracks[matched_tid]
                tdata['bbox'] = bbox
                tdata['last_seen'] = now
                tdata['seen_count'] = tdata.get('seen_count', 0) + 1
                if crop_w > tdata.get('max_crop_w', 0):
                    tdata['max_crop_w'] = crop_w
                if conf > tdata.get('conf', 0.0):
                    tdata['conf'] = conf
            else:
                # EKRANA YENİ GİREN BİR ARAÇ
                new_tid = self.next_track_id
                self.next_track_id += 1

                # Araç eşleştirme
                vehicle_info = None
                if vehicles:
                    vehicle_info = VehicleDetector.match_plate_to_vehicle(bbox, vehicles)

                tdata = {
                    'track_id': new_tid,
                    'bbox': bbox,
                    'conf': conf,
                    'text': '',
                    'ocr_conf': 0.0,
                    'crop': crop,
                    'first_seen': now,
                    'last_seen': now,
                    'seen_count': 1,
                    'max_crop_w': crop_w,
                    'best_valid_text': '',
                    'best_ocr_conf': 0.0,
                    'best_score': 0.0,
                    'best_crop': crop,
                    'best_full_frame': None,  # Sadece loglama anında kopyalanacak
                    'best_bbox': bbox,
                    'vote_counts': {},           # {norm_text: weighted_votes}
                    'vote_raw_counts': {},       # {norm_text: raw_count}
                    'is_match': False,
                    'target_query': None,
                    'alerted': False,
                    'logged': False,
                    'vehicle_type': vehicle_info['class_name'] if vehicle_info else None,
                    'vehicle_conf': vehicle_info['conf'] if vehicle_info else None
                }
                updated_tracks[new_tid] = tdata

            # Sadece yeterli çözünürlüğe sahip kırpıntılarda OCR oku (Uzaktaki gürültüyü engeller)
            if crop_w >= min_crop_w:
                text, ocr_conf = self.ocr.read_plate(crop)
                clean_t = clean_plate_text(text)
                norm_cand = normalize_plate_text(text)
                clean_norm = clean_plate_text(norm_cand)

                # Yasal araç plakası formatı kontrolü (Hem harf hem rakam içerir)
                if is_valid_plate_format(clean_norm) and ocr_conf >= self.min_ocr_conf:
                    # Kalite puanı: karakter sayısı ve format uyumu
                    format_bonus = 1.2 if len(clean_norm) == 7 else 1.0
                    score = ocr_conf * (len(clean_norm) / 7.0) * format_bonus

                    # Confidence-weighted voting: Yüksek güvenli okumalar daha fazla oy alır
                    votes = tdata.setdefault('vote_counts', {})
                    raw_votes = tdata.setdefault('vote_raw_counts', {})
                    weight = ocr_conf  # Oy ağırlığı = OCR güven skoru
                    votes[norm_cand] = votes.get(norm_cand, 0.0) + weight
                    raw_votes[norm_cand] = raw_votes.get(norm_cand, 0) + 1

                    if score > tdata.get('best_score', 0.0):
                        tdata['best_score'] = score
                        tdata['best_valid_text'] = norm_cand
                        tdata['best_ocr_conf'] = ocr_conf
                        tdata['best_crop'] = crop
                        tdata['best_full_frame'] = frame.copy()  # Sadece en iyi kare kopyalanır
                        tdata['best_bbox'] = bbox
                        tdata['text'] = norm_cand
                        tdata['ocr_conf'] = ocr_conf

                    # Yüksek güvenilirlikte anında tekil kayıt (%80+ ve en az 6 karakter)
                    if not tdata.get('logged', False) and ocr_conf >= 0.80 and len(clean_norm) >= 6:
                        # Konsensüs: Ağırlıklı oy lideri (en az 2 ham oy veya %90+ güven)
                        if raw_votes:
                            top_cand = max(votes.items(), key=lambda kv: kv[1])[0]
                            top_raw = raw_votes.get(top_cand, 0)
                        else:
                            top_cand = norm_cand
                            top_raw = 1

                        if top_raw >= 2 or ocr_conf >= 0.90:
                            tdata['text'] = top_cand
                            tdata['ocr_conf'] = ocr_conf
                            tdata['crop'] = crop
                            if tdata.get('best_full_frame') is None:
                                tdata['best_full_frame'] = frame.copy()
                            self._check_target_and_emit(tdata, bbox, tdata['best_full_frame'])

        self.active_tracks = updated_tracks
        # Oynatıcı thread'in kilit beklemeden 0 ms'de çizmesi için atomik referans güncelle
        self.display_tracks = list(updated_tracks.values())

    def _check_target_and_emit(self, tdata: Dict[str, Any], bbox: List[int], full_frame: Optional[np.ndarray] = None):
        """Plakayı tekil olarak kaydeder ve hedeflerle eşleştirir."""
        if tdata.get('logged', False):
            return

        text = tdata.get('text', '')
        clean_text = clean_plate_text(text)
        now = time.time()

        is_unreadable_log = not clean_text or text.startswith("Plaka")

        if not is_unreadable_log:
            # Sadece anlık takip kopmalarını (örn. 1-2 sn direk arkasında kalma) engeller (4 saniye)
            # Araç kadrajdan çıkıp dönüp tekrar gelirse YENİDEN OKUNUR.
            # Farklı araçlar (farklı plakalar) ASLA birbirini engellemez.
            if is_recent_flicker_duplicate(clean_text, self.recent_logged_plates, now, cooldown=4.0):
                tdata['logged'] = True
                return
            self.recent_logged_plates[clean_text] = now

        targets = list(self.target_plates)
        is_match = False
        matched_target = None
        if clean_text and not is_unreadable_log:
            is_match, matched_target = is_target_match(
                text, targets, fuzzy_threshold=self.fuzzy_threshold
            )

        tdata['is_match'] = is_match
        tdata['target_query'] = matched_target
        tdata['logged'] = True

        time_str = datetime.now().strftime("%H:%M:%S")

        display_text = text if text else f"Plaka (%{int(tdata['conf']*100)})"

        # full_frame kopyası sadece loglama anında alınır (performans optimizasyonu)
        frame_for_log = full_frame.copy() if full_frame is not None else np.zeros((100, 100, 3), dtype=np.uint8)

        event_data = {
            'text': display_text,
            'conf': tdata.get('ocr_conf', 0.0),
            'yolo_conf': tdata.get('conf', 0.0),
            'crop': tdata['crop'].copy(),
            'full_frame': frame_for_log,
            'bbox': bbox,
            'time_str': time_str,
            'is_match': is_match,
            'target_query': matched_target,
            'vehicle_type': tdata.get('vehicle_type'),
            'vehicle_conf': tdata.get('vehicle_conf')
        }

        # 1 ARAÇ = 1 SATIR (Listeye sadece 1 kez gönder)
        if self.callback_plate:
            self.callback_plate(event_data)

        # Hedef eşleşmesi varsa 1 kez alarm çal
        if is_match and not tdata['alerted'] and clean_text and clean_text not in self.alerted_plates:
            tdata['alerted'] = True
            self.alerted_plates.add(clean_text)
            if self.callback_alert:
                self.callback_alert(event_data)

    def _draw_annotations(self, frame: np.ndarray) -> np.ndarray:
        """Kare üzerine plaka kutularını gecikmesiz, anlık olarak çizer."""
        annotated = frame.copy()
        now = time.time()

        # Kilit beklemeden sıfır mikrosaniyede çizim listesini al
        tracks_snapshot = self.display_tracks

        for tdata in tracks_snapshot:
            # 0.6 saniyeden eski pozisyonu çizme
            if now - tdata['last_seen'] > 0.6:
                continue

            x1, y1, x2, y2 = tdata['bbox']
            is_match = tdata.get('is_match', False)
            plate_text = tdata.get('text', '')
            best_plate = tdata.get('best_valid_text') or plate_text
            ocr_conf = tdata.get('best_ocr_conf') if tdata.get('best_valid_text') else tdata.get('ocr_conf', 0.0)
            yolo_conf = tdata.get('conf', 0.0)

            if is_match:
                box_color = (0, 0, 255)       # Kırmızı
                bg_color = (0, 0, 220)
                text_color = (255, 255, 255)
                label = f"HEDEF PLAKA: {best_plate}"
            else:
                box_color = (50, 255, 100)     # Neon Yeşil
                bg_color = (20, 180, 70)
                text_color = (0, 0, 0)
                if best_plate and ocr_conf >= self.min_ocr_conf:
                    label = f"{best_plate} (%{int(ocr_conf * 100)})"
                elif best_plate and not best_plate.startswith("Plaka"):
                    label = f"{best_plate} (%{int(ocr_conf * 100)})"
                else:
                    label = f"Plaka (%{int(yolo_conf * 100)})"

            line_thickness = 3 if is_match else 2
            cv2.rectangle(annotated, (x1, y1), (x2, y2), box_color, line_thickness)

            # HUD Köşeleri
            corner_len = max(10, min(18, (x2 - x1) // 4))
            cv2.line(annotated, (x1, y1), (x1 + corner_len, y1), (255, 255, 255), line_thickness + 1)
            cv2.line(annotated, (x1, y1), (x1, y1 + corner_len), (255, 255, 255), line_thickness + 1)
            cv2.line(annotated, (x2, y1), (x2 - corner_len, y1), (255, 255, 255), line_thickness + 1)
            cv2.line(annotated, (x2, y1), (x2, y1 + corner_len), (255, 255, 255), line_thickness + 1)
            cv2.line(annotated, (x1, y2), (x1 + corner_len, y2), (255, 255, 255), line_thickness + 1)
            cv2.line(annotated, (x1, y2), (x1, y2 - corner_len), (255, 255, 255), line_thickness + 1)
            cv2.line(annotated, (x2, y2), (x2 - corner_len, y2), (255, 255, 255), line_thickness + 1)
            cv2.line(annotated, (x2, y2), (x2 - corner_len, y2), (255, 255, 255), line_thickness + 1)

            # Badge Metni
            (w_label, h_label), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_DUPLEX, 0.65, 1)
            badge_y1 = max(0, y1 - h_label - 10)
            badge_y2 = y1
            badge_x1 = x1
            badge_x2 = min(frame.shape[1], x1 + w_label + 14)

            cv2.rectangle(annotated, (badge_x1, badge_y1), (badge_x2, badge_y2), bg_color, -1)
            cv2.putText(
                annotated,
                label,
                (badge_x1 + 6, badge_y2 - 6),
                cv2.FONT_HERSHEY_DUPLEX,
                0.65,
                text_color,
                1,
                cv2.LINE_AA
            )

        return annotated
