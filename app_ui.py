import os
import cv2
import threading
import numpy as np
from datetime import datetime
from typing import List, Dict, Any, Optional

import customtkinter as ctk
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import filedialog, messagebox

from detector import PlateDetector, PlateOCR, VehicleDetector, clean_plate_text, OCR_ENGINE_PPOCR, OCR_ENGINE_FASTPLATE, OCR_ENGINE_DUAL
from video_stream import VideoStreamEngine
from plate_history import PlateHistoryManager

# CustomTkinter Genel Tema Ayarları
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")


class PlateDetailWindow(ctk.CTkToplevel):
    """Tespit edilen plakanın detaylarını ve yüksek çözünürlüklü görselini gösteren modal pencere."""

    def __init__(self, parent, record: Dict[str, Any]):
        super().__init__(parent)
        self.title(f"Plaka Detayı - {record['plate']}")
        self.geometry("720, 580")
        self.attributes("-topmost", True)
        self.configure(fg_color="#0d1117")

        # Başlık ve Rozet
        header_frame = ctk.CTkFrame(self, fg_color="#161b22", corner_radius=10)
        header_frame.pack(fill="x", padx=16, pady=12)

        lbl_title = ctk.CTkLabel(
            header_frame,
            text=f"Plaka: {record['plate']}",
            font=ctk.CTkFont(size=22, weight="bold"),
            text_color="#58a6ff"
        )
        lbl_title.pack(side="left", padx=16, pady=12)

        if record['is_match']:
            badge = ctk.CTkLabel(
                header_frame,
                text="🚨 ARANAN HEDEF PLAKA",
                font=ctk.CTkFont(size=12, weight="bold"),
                fg_color="#da3633",
                text_color="#ffffff",
                corner_radius=6,
                padx=10,
                pady=4
            )
            badge.pack(side="right", padx=16)
        else:
            badge = ctk.CTkLabel(
                header_frame,
                text="NORMAL PLAKA",
                font=ctk.CTkFont(size=12, weight="bold"),
                fg_color="#238636",
                text_color="#ffffff",
                corner_radius=6,
                padx=10,
                pady=4
            )
            badge.pack(side="right", padx=16)

        # Bilgi Metinleri
        info_frame = ctk.CTkFrame(self, fg_color="#161b22", corner_radius=10)
        info_frame.pack(fill="x", padx=16, pady=6)

        vehicle_str = f"\n🚗 Araç Tipi: {record['vehicle_type']}" if record.get('vehicle_type') else ""
        info_text = (
            f"📅 Tespit Zamanı: {record['timestamp']}\n"
            f"🎯 OCR Güven Skoru: %{int(record['conf'] * 100)}\n"
            f"⚡ YOLO Tespit Skoru: %{int(record['yolo_conf'] * 100)}\n"
            f"🔍 Eşleşen Arama: {record['target_query'] if record['target_query'] else '-'}"
            f"{vehicle_str}"
        )
        lbl_info = ctk.CTkLabel(
            info_frame,
            text=info_text,
            font=ctk.CTkFont(size=13),
            justify="left",
            text_color="#c9d1d9"
        )
        lbl_info.pack(anchor="w", padx=16, pady=10)

        # Kırpılmış Görsel
        img_frame = ctk.CTkFrame(self, fg_color="#161b22", corner_radius=10)
        img_frame.pack(fill="both", expand=True, padx=16, pady=8)

        lbl_img_title = ctk.CTkLabel(img_frame, text="Kırpılmış Plaka Görseli:", font=ctk.CTkFont(weight="bold"))
        lbl_img_title.pack(anchor="w", padx=14, pady=6)

        self.img_label = ctk.CTkLabel(img_frame, text="Görsel Yükleniyor...")
        self.img_label.pack(expand=True, pady=10)

        crop_path = record.get('crop_path')
        if crop_path and os.path.exists(crop_path):
            pil_img = Image.open(crop_path)
            # Orantılı boyutlandırma
            w, h = pil_img.size
            new_w = 400
            new_h = int(h * (400.0 / w))
            ctk_img = ctk.CTkImage(light_image=pil_img, dark_image=pil_img, size=(new_w, new_h))
            self.img_label.configure(image=ctk_img, text="")

        # Kapat Butonu
        btn_close = ctk.CTkButton(self, text="Kapat", command=self.destroy, fg_color="#21262d", hover_color="#30363d")
        btn_close.pack(pady=10)


class AutoPlateApp(ctk.CTk):
    """AutoPlate AI Ana Uygulama Arayüzü."""

    def __init__(self, detector: PlateDetector, ocr: PlateOCR, vehicle_detector: Optional[VehicleDetector] = None):
        super().__init__()

        self.detector = detector
        self.ocr = ocr
        self.vehicle_detector = vehicle_detector
        self.history_mgr = PlateHistoryManager()
        self.engine = VideoStreamEngine(detector, ocr)

        # Durum değişkenleri
        self.current_frame_rgb = None
        self.current_raw_frame = None
        self.target_plates_list = []
        self.total_detections_count = 0
        self.total_matched_count = 0
        self.latest_tk_img = None
        self.photo_items = []  # Bellekte çöp toplayıcıdan (GC) koruma

        self._setup_window()
        self._build_ui()
        self._connect_engine()
        self._refresh_target_tags()

    def _setup_window(self):
        self.title("⚡ AutoPlate AI - YOLOv11 & PP-OCRv6 + FastPlateOCR Plaka Tanıma Sistemi")
        self.geometry("1380, 860")
        self.minsize(1100, 720)
        self.configure(fg_color="#0d1117")

    def _build_ui(self):
        # 1. Üst Başlık & Durum Çubuğu
        self.header_frame = self._create_header()
        self.header_frame.pack(fill="x", padx=14, pady=(10, 4))

        # 2. Hedef Plaka Alarm Banner'ı (Eşleştiğinde parlar)
        self.alert_banner = self._create_alert_banner()
        self.alert_banner.pack(fill="x", padx=14, pady=4)
        self.alert_banner.pack_forget()  # Başlangıçta gizli

        # 3. Ana Gövde (Sol: Video Oynatıcı, Sağ: Paneller & Hedefler)
        main_content = ctk.CTkFrame(self, fg_color="transparent")
        main_content.pack(fill="both", expand=True, padx=14, pady=6)

        left_panel = self._create_video_panel(main_content)
        left_panel.pack(side="left", fill="both", expand=True, padx=(0, 8))

        right_panel = self._create_sidebar(main_content)
        right_panel.pack(side="right", fill="both", expand=False, padx=(8, 0))

        # 4. Alt Durum Çubuğu
        self.status_bar = ctk.CTkLabel(
            self,
            text="Hazır. Video dosyası seçin, web kamerası veya IP kamera bağlayın ve 'Başlat'a tıklayın.",
            font=ctk.CTkFont(size=12),
            text_color="#8b949e",
            anchor="w"
        )
        self.status_bar.pack(fill="x", padx=18, pady=(2, 6))

    def _create_header(self) -> ctk.CTkFrame:
        frame = ctk.CTkFrame(self, fg_color="#161b22", corner_radius=12, border_width=1, border_color="#30363d")

        # Sol: Başlık & Açıklama
        title_box = ctk.CTkFrame(frame, fg_color="transparent")
        title_box.pack(side="left", padx=16, pady=8)

        lbl_app_title = ctk.CTkLabel(
            title_box,
            text="⚡ AutoPlate AI",
            font=ctk.CTkFont(size=20, weight="bold"),
            text_color="#58a6ff"
        )
        lbl_app_title.pack(anchor="w")

        lbl_app_sub = ctk.CTkLabel(
            title_box,
            text="YOLOv11 Plaka & Araç Tespiti + FastPlateOCR & PP-OCRv6 Çift Motor + Akıllı Konsensüs",
            font=ctk.CTkFont(size=12),
            text_color="#8b949e"
        )
        lbl_app_sub.pack(anchor="w")

        # Sağ: Model & Durum Rozetleri
        badges_box = ctk.CTkFrame(frame, fg_color="transparent")
        badges_box.pack(side="right", padx=16, pady=8)

        badge_yolo = ctk.CTkLabel(
            badges_box,
            text="MODEL: YOLOv11-Plate",
            font=ctk.CTkFont(size=11, weight="bold"),
            fg_color="#1f242c",
            text_color="#58a6ff",
            corner_radius=6,
            padx=10,
            pady=4
        )
        badge_yolo.pack(side="left", padx=4)

        badge_ocr = ctk.CTkLabel(
            badges_box,
            text=f"OCR: {self.ocr.get_engine_name()}",
            font=ctk.CTkFont(size=11, weight="bold"),
            fg_color="#1f242c",
            text_color="#3fb950",
            corner_radius=6,
            padx=10,
            pady=4
        )
        badge_ocr.pack(side="left", padx=4)
        self.badge_ocr = badge_ocr

        self.badge_status = ctk.CTkLabel(
            badges_box,
            text="DURUM: BEKLEMEDE",
            font=ctk.CTkFont(size=11, weight="bold"),
            fg_color="#21262d",
            text_color="#8b949e",
            corner_radius=6,
            padx=10,
            pady=4
        )
        self.badge_status.pack(side="left", padx=4)

        return frame

    def _create_alert_banner(self) -> ctk.CTkFrame:
        banner = ctk.CTkFrame(self, fg_color="#da3633", corner_radius=10, border_width=1, border_color="#ff7b72")

        self.alert_text = ctk.CTkLabel(
            banner,
            text="🚨 HEDEF PLAKA TESPİT EDİLDİ!",
            font=ctk.CTkFont(size=15, weight="bold"),
            text_color="#ffffff"
        )
        self.alert_text.pack(side="left", padx=16, pady=8)

        btn_dismiss = ctk.CTkButton(
            banner,
            text="Kapat",
            width=80,
            height=28,
            fg_color="#21262d",
            hover_color="#30363d",
            command=lambda: self.alert_banner.pack_forget()
        )
        btn_dismiss.pack(side="right", padx=16, pady=8)

        return banner

    def _create_video_panel(self, parent) -> ctk.CTkFrame:
        panel = ctk.CTkFrame(parent, fg_color="transparent")

        # 1. Video Ekranı (Canvas / Siyah Zemin)
        self.video_container = ctk.CTkFrame(
            panel,
            fg_color="#05070a",
            corner_radius=12,
            border_width=1,
            border_color="#30363d"
        )
        self.video_container.pack(fill="both", expand=True, pady=(0, 8))

        self.video_canvas = tk.Canvas(
            self.video_container,
            bg="#05070a",
            highlightthickness=0
        )
        self.video_canvas.pack(fill="both", expand=True, padx=4, pady=4)
        self.video_canvas.bind("<Configure>", lambda e: self._on_canvas_resize())

        # Başlangıç metni
        self.canvas_text = self.video_canvas.create_text(
            400, 250,
            text="Canlı Video / Kamera Akışı Burada Görünecek\n\nSağ taraftan video seçin veya webcam/IP kamera bağlayıp Başlat'a tıklayın.",
            fill="#484f58",
            font=("Segoe UI", 14),
            justify="center"
        )

        # 2. Oynatma Kontrolleri
        ctrl_card = ctk.CTkFrame(panel, fg_color="#161b22", corner_radius=10, border_width=1, border_color="#30363d")
        ctrl_card.pack(fill="x", pady=4)

        self.btn_play = ctk.CTkButton(
            ctrl_card,
            text="▶ Başlat",
            width=110,
            height=34,
            fg_color="#238636",
            hover_color="#2ea043",
            font=ctk.CTkFont(weight="bold"),
            command=self.start_stream
        )
        self.btn_play.pack(side="left", padx=8, pady=8)

        self.btn_pause = ctk.CTkButton(
            ctrl_card,
            text="⏸ Duraklat",
            width=100,
            height=34,
            fg_color="#21262d",
            hover_color="#30363d",
            state="disabled",
            command=self.toggle_pause
        )
        self.btn_pause.pack(side="left", padx=4, pady=8)

        self.btn_stop = ctk.CTkButton(
            ctrl_card,
            text="⏹ Durdur",
            width=100,
            height=34,
            fg_color="#da3633",
            hover_color="#f85149",
            state="disabled",
            command=self.stop_stream
        )
        self.btn_stop.pack(side="left", padx=4, pady=8)

        self.chk_loop = ctk.CTkCheckBox(
            ctrl_card,
            text="Döngü (Tekrarla)",
            font=ctk.CTkFont(size=12)
        )
        self.chk_loop.select()
        self.chk_loop.pack(side="left", padx=12)

        self.btn_snapshot = ctk.CTkButton(
            ctrl_card,
            text="📸 Anlık Kare Yakala",
            width=140,
            height=34,
            fg_color="#1f6feb",
            hover_color="#388bfd",
            command=self.take_snapshot
        )
        self.btn_snapshot.pack(side="right", padx=8, pady=8)

        # 3. İstatistik Çubuğu (Sabit genişliklerle titremeyi ve kaymayı engeller)
        stats_frame = ctk.CTkFrame(panel, fg_color="#161b22", corner_radius=8, border_width=1, border_color="#30363d")
        stats_frame.pack(fill="x", pady=(4, 0))

        self.lbl_fps = ctk.CTkLabel(stats_frame, text="⚡ FPS: 0.0", width=105, font=ctk.CTkFont(weight="bold"), text_color="#58a6ff", anchor="w")
        self.lbl_fps.pack(side="left", padx=(14, 4), pady=6)

        self.lbl_total = ctk.CTkLabel(stats_frame, text="🚗 Toplam Plaka: 0", width=165, font=ctk.CTkFont(weight="bold"), text_color="#3fb950", anchor="w")
        self.lbl_total.pack(side="left", padx=4, pady=6)

        self.lbl_matched = ctk.CTkLabel(stats_frame, text="🎯 Eşleşen Hedef: 0", width=165, font=ctk.CTkFont(weight="bold"), text_color="#ff7b72", anchor="w")
        self.lbl_matched.pack(side="left", padx=4, pady=6)

        self.lbl_frame_info = ctk.CTkLabel(stats_frame, text="Kare: 0", width=145, font=ctk.CTkFont(size=12), text_color="#8b949e", anchor="e")
        self.lbl_frame_info.pack(side="right", padx=(4, 14), pady=6)

        self.lbl_ocr_latency = ctk.CTkLabel(stats_frame, text="OCR: 0ms", width=170, font=ctk.CTkFont(weight="bold"), text_color="#d2a8ff", anchor="e")
        self.lbl_ocr_latency.pack(side="right", padx=4, pady=6)

        return panel

    def _create_sidebar(self, parent) -> ctk.CTkFrame:
        sidebar = ctk.CTkFrame(parent, fg_color="transparent", width=460)
        sidebar.pack_propagate(False)  # Sağ panelin genişliğini 460px'e kilitler, video ve butonların zıplamasını engeller

        # Sekmeli Yönetim (Hedef Plaka / Giriş Kaynağı / Ayarlar)
        self.tabs = ctk.CTkTabview(sidebar, fg_color="#161b22", corner_radius=10)
        self.tabs.pack(fill="x", expand=False, pady=(0, 8))

        tab_target = self.tabs.add("🎯 Hedef Plaka")
        tab_source = self.tabs.add("📹 Giriş Kaynağı")
        tab_settings = self.tabs.add("⚙ Ayarlar")

        self._build_target_tab(tab_target)
        self._build_source_tab(tab_source)
        self._build_settings_tab(tab_settings)

        # Alt Kısım: Canlı Plaka Tespit Geçmişi Listesi
        history_box = ctk.CTkFrame(sidebar, fg_color="#161b22", corner_radius=10, border_width=1, border_color="#30363d")
        history_box.pack(fill="both", expand=True)

        h_top = ctk.CTkFrame(history_box, fg_color="transparent")
        h_top.pack(fill="x", padx=10, pady=8)

        lbl_hist_title = ctk.CTkLabel(
            h_top,
            text="📋 Canlı Tespit Geçmişi",
            font=ctk.CTkFont(size=14, weight="bold"),
            text_color="#58a6ff"
        )
        lbl_hist_title.pack(side="left")

        btn_csv = ctk.CTkButton(
            h_top,
            text="📥 CSV Aktar",
            width=90,
            height=26,
            fg_color="#21262d",
            hover_color="#30363d",
            font=ctk.CTkFont(size=11),
            command=self.export_csv
        )
        btn_csv.pack(side="right", padx=4)

        btn_clear = ctk.CTkButton(
            h_top,
            text="Temizle",
            width=70,
            height=26,
            fg_color="#21262d",
            hover_color="#30363d",
            font=ctk.CTkFont(size=11),
            command=self.clear_history
        )
        btn_clear.pack(side="right", padx=4)

        # Kaydırılabilir Tespit Listesi
        self.history_scroll = ctk.CTkScrollableFrame(
            history_box,
            fg_color="#0d1117",
            corner_radius=8
        )
        self.history_scroll.pack(fill="both", expand=True, padx=8, pady=(0, 8))

        tip_lbl = ctk.CTkLabel(
            history_box,
            text="💡 Detayları ve kırpılmış görseli görmek için karta tıklayın.",
            font=ctk.CTkFont(size=11),
            text_color="#8b949e"
        )
        tip_lbl.pack(pady=(0, 6))

        return sidebar

    def _build_target_tab(self, parent):
        lbl_desc = ctk.CTkLabel(
            parent,
            text="Tespit edildiğinde alarm çalınacak hedef plakaları girin:",
            font=ctk.CTkFont(size=12),
            text_color="#8b949e",
            wraplength=380,
            justify="left"
        )
        lbl_desc.pack(anchor="w", padx=4, pady=2)

        input_box = ctk.CTkFrame(parent, fg_color="transparent")
        input_box.pack(fill="x", padx=4, pady=6)

        self.txt_target_input = ctk.CTkEntry(
            input_box,
            placeholder_text="Örn: 34 ABC 123 veya 06 XYZ 99",
            border_color="#ff7b72",
            height=34
        )
        self.txt_target_input.pack(side="left", fill="x", expand=True, padx=(0, 6))
        self.txt_target_input.bind("<Return>", lambda e: self.add_target_plate())

        btn_add = ctk.CTkButton(
            input_box,
            text="Ekle",
            width=70,
            height=34,
            fg_color="#da3633",
            hover_color="#f85149",
            font=ctk.CTkFont(weight="bold"),
            command=self.add_target_plate
        )
        btn_add.pack(side="right")

        # Aktif Hedefler Listesi
        lbl_tags = ctk.CTkLabel(parent, text="Takip Edilen Hedefler:", font=ctk.CTkFont(size=12, weight="bold"))
        lbl_tags.pack(anchor="w", padx=4, pady=(6, 2))

        self.targets_container = ctk.CTkScrollableFrame(parent, fg_color="#0d1117", height=90, corner_radius=8)
        self.targets_container.pack(fill="x", padx=4, pady=4)

        # Seçenekler
        self.chk_sound = ctk.CTkCheckBox(parent, text="🔊 Eşleşmede Sesli Alarm Çal", font=ctk.CTkFont(size=12))
        self.chk_sound.select()
        self.chk_sound.pack(anchor="w", padx=4, pady=4)

        self.chk_fuzzy = ctk.CTkCheckBox(parent, text="🔍 Benzerlik / Fuzzy Eşleşme Toleransı", font=ctk.CTkFont(size=12))
        self.chk_fuzzy.select()
        self.chk_fuzzy.pack(anchor="w", padx=4, pady=4)

    def _build_source_tab(self, parent):
        self.source_segmented = ctk.CTkSegmentedButton(
            parent,
            values=["📁 Video", "📷 Webcam", "🌐 IP Kamera"],
            command=self._on_source_segment_changed
        )
        self.source_segmented.set("📁 Video")
        self.source_segmented.pack(fill="x", padx=4, pady=8)

        # 1. Video Dosyası Alanı
        self.frame_src_video = ctk.CTkFrame(parent, fg_color="transparent")
        self.frame_src_video.pack(fill="x", padx=4, pady=4)

        lbl_v = ctk.CTkLabel(self.frame_src_video, text="Video Dosyası (.mp4, .avi, .mkv, .mov):", font=ctk.CTkFont(size=12))
        lbl_v.pack(anchor="w")

        v_input_box = ctk.CTkFrame(self.frame_src_video, fg_color="transparent")
        v_input_box.pack(fill="x", pady=4)

        self.txt_video_path = ctk.CTkEntry(v_input_box, placeholder_text="Video dosyasının yolu...", height=32)
        self.txt_video_path.pack(side="left", fill="x", expand=True, padx=(0, 6))

        btn_browse = ctk.CTkButton(v_input_box, text="Gözat...", width=70, height=32, command=self.browse_video_file)
        btn_browse.pack(side="right")

        # 2. Webcam Alanı
        self.frame_src_webcam = ctk.CTkFrame(parent, fg_color="transparent")
        lbl_w = ctk.CTkLabel(self.frame_src_webcam, text="Kamera Cihaz Numarası:", font=ctk.CTkFont(size=12))
        lbl_w.pack(anchor="w")

        self.cmb_webcam = ctk.CTkComboBox(
            self.frame_src_webcam,
            values=["0 - Dahili Webcam", "1 - Harici Kamera", "2 - Harici Kamera 2"],
            height=32
        )
        self.cmb_webcam.set("0 - Dahili Webcam")
        self.cmb_webcam.pack(fill="x", pady=4)

        # 3. IP Kamera Alanı
        self.frame_src_ip = ctk.CTkFrame(parent, fg_color="transparent")
        lbl_ip = ctk.CTkLabel(self.frame_src_ip, text="RTSP / HTTP Canlı Akış Adresi:", font=ctk.CTkFont(size=12))
        lbl_ip.pack(anchor="w")

        self.txt_ip_url = ctk.CTkEntry(
            self.frame_src_ip,
            placeholder_text="rtsp://admin:sifre@192.168.1.100:554/stream1",
            height=32
        )
        self.txt_ip_url.pack(fill="x", pady=4)

        lbl_ip_hint = ctk.CTkLabel(
            self.frame_src_ip,
            text="Örnek: rtsp://192.168.1.50:554/h264 veya http://ip:port/video",
            font=ctk.CTkFont(size=11),
            text_color="#8b949e"
        )
        lbl_ip_hint.pack(anchor="w")

    def _build_settings_tab(self, parent):
        # 1. YOLO Güven Eşiği
        self.lbl_conf = ctk.CTkLabel(parent, text="YOLOv11 Güven Eşiği: %35", font=ctk.CTkFont(weight="bold"))
        self.lbl_conf.pack(anchor="w", padx=4, pady=(4, 2))

        self.sld_conf = ctk.CTkSlider(
            parent,
            from_=15,
            to=90,
            number_of_steps=75,
            command=self._on_conf_slider
        )
        self.sld_conf.set(35)
        self.sld_conf.pack(fill="x", padx=4, pady=4)

        # 2. OCR Minimum Güven Eşiği
        self.lbl_ocr_conf = ctk.CTkLabel(parent, text="OCR Minimum Güven Eşiği: %35", font=ctk.CTkFont(weight="bold"))
        self.lbl_ocr_conf.pack(anchor="w", padx=4, pady=(8, 2))

        self.sld_ocr_conf = ctk.CTkSlider(
            parent,
            from_=15,
            to=85,
            number_of_steps=70,
            command=self._on_ocr_conf_slider
        )
        self.sld_ocr_conf.set(35)
        self.sld_ocr_conf.pack(fill="x", padx=4, pady=4)

        # 3. Çözünürlük (Uzaktaki Araçlar & 4K)
        lbl_res = ctk.CTkLabel(parent, text="Model Çözünürlüğü:", font=ctk.CTkFont(weight="bold"))
        lbl_res.pack(anchor="w", padx=4, pady=(8, 2))

        self.cmb_resolution = ctk.CTkComboBox(
            parent,
            values=[
                "Otomatik (4K ve HD Video Uyumlu)",
                "Standart (640px - Hızlı ve Akıcı)",
                "Yüksek Hassasiyet (1280px - Uzaktaki Plakalar)",
                "Ultra Hızlı (320px)"
            ],
            height=32,
            command=lambda v: self._apply_engine_settings()
        )
        self.cmb_resolution.set("Otomatik (4K ve HD Video Uyumlu)")
        self.cmb_resolution.pack(fill="x", padx=4, pady=4)

        # 4. Kare Atlama
        lbl_skip = ctk.CTkLabel(parent, text="Performans / Kare Atlama:", font=ctk.CTkFont(weight="bold"))
        lbl_skip.pack(anchor="w", padx=4, pady=(8, 2))

        self.cmb_skip = ctk.CTkComboBox(
            parent,
            values=[
                "Her Kareyi İncele (Maksimum Hassasiyet)",
                "2 Karede Bir İncele (Dengeli)",
                "3 Karede Bir İncele (Yüksek FPS)"
            ],
            height=32,
            command=lambda v: self._apply_engine_settings()
        )
        self.cmb_skip.set("Her Kareyi İncele (Maksimum Hassasiyet)")
        self.cmb_skip.pack(fill="x", padx=4, pady=4)

        # 5. Gece / Düşük Işık Desteği
        self.chk_unreadable = ctk.CTkCheckBox(
            parent,
            text="Gece / Yansımalı Plakaları Kaydet",
            font=ctk.CTkFont(size=12),
            command=self._apply_engine_settings
        )
        self.chk_unreadable.select()
        self.chk_unreadable.pack(anchor="w", padx=4, pady=(10, 2))

        lbl_unreadable_hint = ctk.CTkLabel(
            parent,
            text="Işık parlaması veya hareket bulanıklığında plaka metni okunamasa bile araç görselini ve kaydını korur.",
            font=ctk.CTkFont(size=10),
            text_color="#8b949e",
            wraplength=380,
            justify="left"
        )
        lbl_unreadable_hint.pack(anchor="w", padx=4, pady=(0, 4))

        # 6. OCR Motor Seçimi
        lbl_ocr_engine = ctk.CTkLabel(parent, text="OCR Motoru:", font=ctk.CTkFont(weight="bold"))
        lbl_ocr_engine.pack(anchor="w", padx=4, pady=(10, 2))

        self.cmb_ocr_engine = ctk.CTkComboBox(
            parent,
            values=[
                "FastPlateOCR (Ultra Hızlı, 65+ Ülke - Varsayılan)",
                "Çift Motor (FastPlate + PP-OCRv6 En İyi Sonuç)",
                "PP-OCRv6 (Klasik Motor)"
            ],
            height=32,
            command=lambda v: self._on_ocr_engine_changed(v)
        )
        self.cmb_ocr_engine.set("FastPlateOCR (Ultra Hızlı, 65+ Ülke - Varsayılan)")
        self.cmb_ocr_engine.pack(fill="x", padx=4, pady=4)

        lbl_ocr_hint = ctk.CTkLabel(
            parent,
            text="FastPlateOCR: 20-60ms ultra hızlı yanıt süresi ve zorlu açılarda yüksek doğruluk sağlar.",
            font=ctk.CTkFont(size=10),
            text_color="#8b949e",
            wraplength=380,
            justify="left"
        )
        lbl_ocr_hint.pack(anchor="w", padx=4, pady=(0, 4))

        # 7. Araç Tespiti Toggle
        self.chk_vehicle_detection = ctk.CTkCheckBox(
            parent,
            text="🚗 Araç Tespiti (Plaka-Araç Eşleştirme)",
            font=ctk.CTkFont(size=12),
            command=self._apply_engine_settings
        )
        if self.vehicle_detector is not None:
            self.chk_vehicle_detection.select()
        self.chk_vehicle_detection.pack(anchor="w", padx=4, pady=(6, 2))

        lbl_vehicle_hint = ctk.CTkLabel(
            parent,
            text="Aktifken plaka ile aracı eşleştirir ve araç tipini (otomobil, kamyon, otobüs) kaydeder.",
            font=ctk.CTkFont(size=10),
            text_color="#8b949e",
            wraplength=380,
            justify="left"
        )
        lbl_vehicle_hint.pack(anchor="w", padx=4, pady=(0, 4))

    def _on_source_segment_changed(self, value):
        self.frame_src_video.pack_forget()
        self.frame_src_webcam.pack_forget()
        self.frame_src_ip.pack_forget()

        if value == "📁 Video":
            self.frame_src_video.pack(fill="x", padx=4, pady=4)
        elif value == "📷 Webcam":
            self.frame_src_webcam.pack(fill="x", padx=4, pady=4)
        elif value == "🌐 IP Kamera":
            self.frame_src_ip.pack(fill="x", padx=4, pady=4)

    def _on_conf_slider(self, val):
        val_int = int(val)
        self.lbl_conf.configure(text=f"YOLOv11 Güven Eşiği: %{val_int}")
        self._apply_engine_settings()

    def _on_ocr_conf_slider(self, val):
        val_int = int(val)
        self.lbl_ocr_conf.configure(text=f"OCR Minimum Güven Eşiği: %{val_int}")
        self._apply_engine_settings()

    def _on_ocr_engine_changed(self, value: str):
        """Kullanıcı ayarlar menüsünden OCR motoru seçtiğinde çalışır."""
        if "PP-OCRv6" in value:
            self.ocr.set_engine(OCR_ENGINE_PPOCR)
        elif "FastPlateOCR" in value:
            self.ocr.set_engine(OCR_ENGINE_FASTPLATE)
        else:
            self.ocr.set_engine(OCR_ENGINE_DUAL)

        if hasattr(self, 'badge_ocr'):
            self.badge_ocr.configure(text=f"OCR: {self.ocr.get_engine_name()}")
        self.status_bar.configure(text=f"OCR motoru değiştirildi: {self.ocr.get_engine_name()}")

    def _connect_engine(self):
        self.engine.callback_frame = self._on_frame_received
        self.engine.callback_plate = self._on_plate_detected
        self.engine.callback_alert = self._on_match_alert
        self.engine.callback_stats = self._on_stats_received
        self.engine.callback_error = self._on_error_received
        self.engine.callback_completed = self._on_stream_completed

    def _apply_engine_settings(self):
        conf = self.sld_conf.get() / 100.0
        ocr_conf = self.sld_ocr_conf.get() / 100.0 if hasattr(self, 'sld_ocr_conf') else 0.35
        log_unreadable = (self.chk_unreadable.get() == 1) if hasattr(self, 'chk_unreadable') else True

        res_str = self.cmb_resolution.get()
        imgsz = 640
        if "Otomatik" in res_str:
            imgsz = 0  # Video akışında genişliğe göre otomatik belirlenir (4K için 1280px)
        elif "1280px" in res_str:
            imgsz = 1280
        elif "320px" in res_str:
            imgsz = 320

        skip_str = self.cmb_skip.get()
        skip = 1
        if "2 Karede" in skip_str:
            skip = 2
        elif "3 Karede" in skip_str:
            skip = 3

        self.engine.set_settings(conf, imgsz, skip, min_ocr_conf=ocr_conf, log_unreadable=log_unreadable)
        self.engine.set_targets(self.target_plates_list)
        self.engine.loop_video = (self.chk_loop.get() == 1)
        self.engine.fuzzy_threshold = 0.82 if self.chk_fuzzy.get() == 1 else 1.0

        # Araç tespiti
        vehicle_enabled = (self.chk_vehicle_detection.get() == 1) if hasattr(self, 'chk_vehicle_detection') else False
        if vehicle_enabled and self.vehicle_detector is not None:
            self.engine.set_vehicle_detection(True, self.vehicle_detector)
        else:
            self.engine.set_vehicle_detection(False)

    def add_target_plate(self):
        text = self.txt_target_input.get().strip()
        if not text:
            return

        plates = [p.strip() for p in text.split(",") if p.strip()]
        for p in plates:
            clean = clean_plate_text(p)
            if clean and clean not in [clean_plate_text(t) for t in self.target_plates_list]:
                self.target_plates_list.append(p.upper())

        self.txt_target_input.delete(0, "end")
        self._refresh_target_tags()
        self.engine.set_targets(self.target_plates_list)

    def remove_target_plate(self, target: str):
        if target in self.target_plates_list:
            self.target_plates_list.remove(target)
            self._refresh_target_tags()
            self.engine.set_targets(self.target_plates_list)

    def _refresh_target_tags(self):
        for widget in self.targets_container.winfo_children():
            widget.destroy()

        if not self.target_plates_list:
            lbl = ctk.CTkLabel(self.targets_container, text="Henüz hedef eklenmedi.", text_color="#8b949e")
            lbl.pack(pady=4)
            return

        for target in self.target_plates_list:
            tag = ctk.CTkFrame(self.targets_container, fg_color="#220e11", border_width=1, border_color="#ff7b72", corner_radius=6)
            tag.pack(fill="x", pady=2)

            lbl = ctk.CTkLabel(tag, text=f"🎯 {target}", font=ctk.CTkFont(weight="bold"), text_color="#ff7b72")
            lbl.pack(side="left", padx=8, pady=2)

            btn_del = ctk.CTkButton(
                tag,
                text="✕",
                width=24,
                height=24,
                fg_color="transparent",
                hover_color="#30363d",
                text_color="#ff7b72",
                command=lambda t=target: self.remove_target_plate(t)
            )
            btn_del.pack(side="right", padx=4)

    def browse_video_file(self):
        file_path = filedialog.askopenfilename(
            title="Video Dosyası Seç",
            filetypes=[("Video Dosyaları", "*.mp4 *.avi *.mkv *.mov"), ("Tüm Dosyalar", "*.*")]
        )
        if file_path:
            self.txt_video_path.delete(0, "end")
            self.txt_video_path.insert(0, file_path)

    def start_stream(self):
        seg = self.source_segmented.get()
        source_type = "file"
        source_path = ""

        if seg == "📁 Video":
            source_type = "file"
            source_path = self.txt_video_path.get().strip()
            if not source_path or not os.path.exists(source_path):
                messagebox.showwarning("Kaynak Hatası", "Lütfen geçerli bir video dosyası seçin!")
                return
        elif seg == "📷 Webcam":
            source_type = "webcam"
            cam_str = self.cmb_webcam.get()
            try:
                source_path = int(cam_str.split("-")[0].strip())
            except Exception:
                source_path = 0
        elif seg == "🌐 IP Kamera":
            source_type = "ip"
            source_path = self.txt_ip_url.get().strip()
            if not source_path:
                messagebox.showwarning("Kaynak Hatası", "Lütfen geçerli bir IP/RTSP kamera URL'si girin!")
                return

        self.engine.stop()
        self.engine.clear_tracking_history()
        self.engine.set_source(source_type, source_path)
        self._apply_engine_settings()
        self.engine.start()

        self.btn_play.configure(state="disabled")
        self.btn_pause.configure(state="normal", text="⏸ Duraklat")
        self.btn_stop.configure(state="normal")

        source_name = "Video" if source_type == "file" else ("Webcam" if source_type == "webcam" else "IP Kamera")
        self.badge_status.configure(text=f"DURUM: AKTİF ({source_name})", fg_color="#238636", text_color="#ffffff")
        self.status_bar.configure(text=f"Yayın başlatıldı: {source_path}")

    def toggle_pause(self):
        self.engine.toggle_pause()
        if self.engine.is_paused:
            self.btn_pause.configure(text="▶ Devam Et")
            self.status_bar.configure(text="Yayın duraklatıldı.")
        else:
            self.btn_pause.configure(text="⏸ Duraklat")
            self.status_bar.configure(text="Yayın devam ediyor.")

    def stop_stream(self):
        self.engine.stop()
        self.btn_play.configure(state="normal")
        self.btn_pause.configure(state="disabled", text="⏸ Duraklat")
        self.btn_stop.configure(state="disabled")

        self.badge_status.configure(text="DURUM: DURDURULDU", fg_color="#21262d", text_color="#8b949e")
        self.status_bar.configure(text="Yayın durduruldu.")

    def _on_frame_received(self, rgb_frame: np.ndarray, raw_bgr: np.ndarray):
        self.current_frame_rgb = rgb_frame
        self.current_raw_frame = raw_bgr

        # GUI ana thread'inde donmayı ve kuyruk birikmesini önleyen akıllı çizim
        if not getattr(self, '_is_rendering', False):
            self._is_rendering = True
            self.after(1, self._render_frame)

    def _render_frame(self):
        try:
            if self.current_frame_rgb is None:
                return

            cw = self.video_canvas.winfo_width()
            ch = self.video_canvas.winfo_height()

            if cw < 50 or ch < 50:
                return

            h, w = self.current_frame_rgb.shape[:2]
            scale = min(cw / w, ch / h)
            nw, nh = max(1, int(w * scale)), max(1, int(h * scale))

            resized = cv2.resize(self.current_frame_rgb, (nw, nh), interpolation=cv2.INTER_LINEAR)
            pil_img = Image.fromarray(resized)
            tk_img = ImageTk.PhotoImage(pil_img)

            self.latest_tk_img = tk_img  # GC koruma

            # Canvas öğesini her karede silip baştan oluşturmak yerine var olanı güncelle (Çok daha akıcı)
            canvas_img = getattr(self, '_canvas_img_id', None)
            if canvas_img is None:
                self.video_canvas.delete("all")
                self._canvas_img_id = self.video_canvas.create_image(cw // 2, ch // 2, image=tk_img, anchor="center")
            else:
                self.video_canvas.coords(self._canvas_img_id, cw // 2, ch // 2)
                self.video_canvas.itemconfig(self._canvas_img_id, image=tk_img)
        finally:
            self._is_rendering = False

    def _on_canvas_resize(self):
        """Pencere büyütüldüğünde veya boyutu değiştiğinde videoyu veya karşılama metnini anında merkeze alır ve ölçekler."""
        cw = self.video_canvas.winfo_width()
        ch = self.video_canvas.winfo_height()
        if cw < 50 or ch < 50:
            return

        if self.current_frame_rgb is not None:
            self._render_frame()
        elif hasattr(self, 'canvas_text') and self.canvas_text:
            self.video_canvas.coords(self.canvas_text, cw // 2, ch // 2)

    def _on_plate_detected(self, event_data: Dict[str, Any]):
        record = self.history_mgr.add_record(event_data)
        self.after(0, lambda: self._add_history_card(record))

    def _add_history_card(self, record: Dict[str, Any]):
        self.total_detections_count += 1
        self.lbl_total.configure(text=f"🚗 Toplam Plaka: {self.total_detections_count}")

        card = ctk.CTkFrame(
            self.history_scroll,
            fg_color="#161b22",
            border_width=1,
            border_color="#ff7b72" if record['is_match'] else "#30363d",
            corner_radius=8
        )
        card.pack(fill="x", pady=3)

        # Tıklanabilirlik (Detay modalı açma)
        card.bind("<Button-1>", lambda e, r=record: PlateDetailWindow(self, r))

        # Sol: Kırpılmış Görsel İkonu
        crop_img = record.get('crop_img')
        if crop_img is not None and crop_img.size > 0:
            pil_crop = Image.fromarray(cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB))
            crop_w, crop_h = pil_crop.size
            thumb_w = 64
            thumb_h = max(18, int(crop_h * (64.0 / max(1, crop_w))))
            ctk_thumb = ctk.CTkImage(light_image=pil_crop, dark_image=pil_crop, size=(thumb_w, thumb_h))
            lbl_thumb = ctk.CTkLabel(card, text="", image=ctk_thumb)
            lbl_thumb.pack(side="left", padx=6, pady=4)
            lbl_thumb.bind("<Button-1>", lambda e, r=record: PlateDetailWindow(self, r))

        # Orta: Plaka No ve Saat
        info_box = ctk.CTkFrame(card, fg_color="transparent")
        info_box.pack(side="left", fill="both", expand=True, padx=4)
        info_box.bind("<Button-1>", lambda e, r=record: PlateDetailWindow(self, r))

        p_lbl = ctk.CTkLabel(
            info_box,
            text=record['plate'],
            font=ctk.CTkFont(size=13, weight="bold"),
            text_color="#ff7b72" if record['is_match'] else "#ffffff"
        )
        p_lbl.pack(anchor="w")
        p_lbl.bind("<Button-1>", lambda e, r=record: PlateDetailWindow(self, r))

        time_lbl = ctk.CTkLabel(
            info_box,
            text=f"Saat: {record['time_only']} | Güven: %{int(record['conf']*100)}",
            font=ctk.CTkFont(size=10),
            text_color="#8b949e"
        )
        time_lbl.pack(anchor="w")
        time_lbl.bind("<Button-1>", lambda e, r=record: PlateDetailWindow(self, r))

        # Araç tipi göstergesi
        vehicle_type = record.get('vehicle_type')
        if vehicle_type:
            v_lbl = ctk.CTkLabel(
                info_box,
                text=f"🚗 {vehicle_type}",
                font=ctk.CTkFont(size=10),
                text_color="#7ee787"
            )
            v_lbl.pack(anchor="w")
            v_lbl.bind("<Button-1>", lambda e, r=record: PlateDetailWindow(self, r))

        # Sağ: Rozet
        if record['is_match']:
            b_lbl = ctk.CTkLabel(card, text="🚨 HEDEF", font=ctk.CTkFont(size=10, weight="bold"), fg_color="#da3633", text_color="#ffffff", corner_radius=4, padx=6, pady=2)
        else:
            b_lbl = ctk.CTkLabel(card, text="NORMAL", font=ctk.CTkFont(size=10, weight="bold"), fg_color="#238636", text_color="#ffffff", corner_radius=4, padx=6, pady=2)
        b_lbl.pack(side="right", padx=8)
        b_lbl.bind("<Button-1>", lambda e, r=record: PlateDetailWindow(self, r))

    def _on_match_alert(self, event_data: Dict[str, Any]):
        self.after(0, lambda: self._show_match_alert_ui(event_data))

    def _show_match_alert_ui(self, event_data: Dict[str, Any]):
        self.total_matched_count += 1
        self.lbl_matched.configure(text=f"🎯 Eşleşen Hedef: {self.total_matched_count}")

        plate = event_data.get('text', '')
        time_str = event_data.get('time_str', '')
        target_query = event_data.get('target_query', '')

        self.alert_text.configure(text=f"🚨 HEDEF PLAKA TESPİT EDİLDİ: {plate} (Hedef: {target_query} | Saat: {time_str})")
        self.alert_banner.pack(fill="x", padx=14, pady=4, before=self.header_frame)

    def _on_stats_received(self, fps: float, current_frame: int, total_frames: int):
        self.after(0, lambda: self._update_stats_ui(fps, current_frame, total_frames))

    def _update_stats_ui(self, fps: float, current_frame: int, total_frames: int):
        self.lbl_fps.configure(text=f"⚡ FPS: {fps:.1f}")
        if total_frames > 0:
            self.lbl_frame_info.configure(text=f"Kare: {current_frame} / {total_frames}")
        else:
            self.lbl_frame_info.configure(text=f"Kare: {current_frame}")

        # OCR gecikme istatistiklerini güncelle
        latency = self.ocr.get_latency_stats()
        if latency['count'] > 0:
            self.lbl_ocr_latency.configure(text=f"OCR: {latency['avg']:.0f}ms (p95:{latency['p95']:.0f}ms)")

    def _on_error_received(self, err_msg: str):
        self.after(0, lambda: messagebox.showerror("Hata", err_msg))
        self.after(0, self.stop_stream)

    def _on_stream_completed(self):
        self.after(0, self._handle_stream_completed_ui)

    def _handle_stream_completed_ui(self):
        self.btn_play.configure(state="normal")
        self.btn_pause.configure(state="disabled", text="⏸ Duraklat")
        self.btn_stop.configure(state="disabled")
        self.badge_status.configure(text="DURUM: TAMAMLANDI", fg_color="#21262d", text_color="#8b949e")
        self.status_bar.configure(text="Video akışı tamamlandı. Yeniden oynatmak için 'Başlat'a tıklayın.")

    def take_snapshot(self):
        if self.current_raw_frame is None:
            messagebox.showinfo("Bilgi", "Aktif görüntü karesi bulunamadı.")
            return

        date_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        folder = "snapshots"
        os.makedirs(folder, exist_ok=True)
        path = os.path.join(folder, f"snapshot_{date_str}.jpg")
        cv2.imwrite(path, self.current_raw_frame)
        messagebox.showinfo("Başarılı", f"Anlık kare kaydedildi:\n{os.path.abspath(path)}")

    def export_csv(self):
        if not self.history_mgr.records:
            messagebox.showinfo("Bilgi", "Henüz dışa aktarılacak plaka kaydı yok.")
            return

        file_path = filedialog.asksaveasfilename(
            title="CSV Raporunu Kaydet",
            initialfile=f"plaka_kayitlari_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
            defaultextension=".csv",
            filetypes=[("CSV Dosyaları", "*.csv")]
        )
        if file_path:
            exported = self.history_mgr.export_csv(file_path)
            messagebox.showinfo("Başarılı", f"Kayıtlar CSV dosyasına aktarıldı:\n{exported}")

    def clear_history(self):
        for widget in self.history_scroll.winfo_children():
            widget.destroy()
        self.history_mgr.records.clear()
        self.total_detections_count = 0
        self.total_matched_count = 0
        self.lbl_total.configure(text="🚗 Toplam Plaka: 0")
        self.lbl_matched.configure(text="🎯 Eşleşen Hedef: 0")
        self.engine.clear_tracking_history()
        self.status_bar.configure(text="Tespit geçmişi ve takip hafızası temizlendi.")

    def destroy(self):
        self.engine.stop()
        super().destroy()
