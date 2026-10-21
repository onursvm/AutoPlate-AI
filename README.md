# ⚡ AutoPlate AI - YOLOv11 & EasyOCR Plaka Tanıma ve Hedef Takip Sistemi

Yapay zeka tabanlı bu masaüstü uygulaması; **YOLOv11** ile araçların uzaktaki plakalarını yüksek hassasiyetle tespit eder, **EasyOCR** ile plaka üzerindeki alfanümerik metinleri okur ve belirlediğiniz **hedef plakalarla (aranan/şüpheli liste)** anlık eşleştirme yaparak sesli ve görsel alarm üretir.

---

## 🌟 Öne Çıkan Özellikler

1. **YOLOv11 Plaka Tespiti (`license-plate-finetune-v1n.pt`)**:
   - Araç plakaları üzerine özel eğitilmiş YOLOv11 mimarisi.
   - **Uzaktaki Araçlar İçin Yüksek Çözünürlük Desteği**: Standart 640px veya 1280px yüksek çözünürlüklü tarama modu.
   - Plaka sınırlarını pay bırakarak keser (karakterlerin kenardan kırpılmasını önler).

2. **Gelişmiş EasyOCR & Görüntü Ön İşleme (Enhancement)**:
   - Küçük ve uzaktaki plaka görüntüleri için otomatik **Bicubic boyut büyütme**.
   - **CLAHE (Contrast Limited Adaptive Histogram Equalization)** ile parlama ve gölgeleri dengeleme.
   - **Bilateral filtreleme** ve keskinleştirme ile harf kenarlarını belirginleştirme.
   - Türk plakalarındaki sol mavi bandı (`TR` / `R`) ve gereksiz karakterleri temizleyen akıllı normalizasyon.

3. **🎯 Hedef Plaka Arama ve Eşleştirme (Target Matcher)**:
   - Takip etmek istediğiniz plakayı (veya virgülle birden fazla plakayı) arayüze ekleyin.
   - Birebir tam eşleşme, alt dize eşleşmesi ve **Fuzzy (benzerlik toleransı)** desteği.
   - Hedef yakalandığında:
     - 🚨 **Görsel Alarm:** Canlı video üzerinde kırmızı fütüristik çerçeve ve üst bant uyarısı.
     - 🔊 **Sesli Alarm:** Çift tonlu dikkat çekici sesli uyarı.
     - 💾 **Otomatik Delil Kaydı:** Plakanın kırpılmış hali ve anlık tam kamera karesi `records/alerts/` klasörüne zaman damgasıyla otomatik kaydedilir.

4. **📹 Çoklu Giriş Kaynağı Desteği**:
   - **Video Dosyası**: `.mp4`, `.avi`, `.mkv`, `.mov` dosyalarını seçme, duraklatma ve döngüyle oynatma.
   - **Web Kamerası**: Bilgisayara bağlı dilediğiniz kamerayı (0, 1, 2...) canlı izleme.
   - **Güvenlik Kamerası (IP / RTSP Akışı)**: Güvenlik kamerasının RTSP (`rtsp://kullanici:sifre@ip:port/stream`) veya HTTP canlı akış adresini girerek anlık analiz yapma.

5. **📋 Canlı Tespit Tablosu & Excel/CSV Raporlama**:
   - Tespit edilen tüm plakaları anlık olarak listeler.
   - Çift tıklayarak plakanın büyütülmüş görselini ve tam kare fotoğrafını inceleme.
   - Tek tıkla tüm kayıtları Excel uyumlu UTF-8 CSV olarak dışa aktarma.

---

## 🚀 Kurulum ve Çalıştırma

### 1. Uygulamayı Başlatma
Proje dizininde sanal ortam (`.venv`) üzerinden uygulamayı başlatmak için:

```bash
.venv\Scripts\python.exe main.py
```
*(veya terminalde `python main.py`)*

### 2. Video ile Kullanım
1. Uygulamayı çalıştırın.
2. Sağdaki **"📹 Giriş Kaynağı"** sekmesinden **"Gözat..."** butonuna tıklayarak bilgisayarınızdaki herhangi bir araç/trafik videosunu (`.mp4`, `.avi`, `.mkv`, `.mov`) seçin.
3. İsterseniz **"🎯 Hedef Plaka"** sekmesinden aramak istediğiniz bir plakayı yazıp **"Ekle"**ye basın.
4. **"▶ Başlat"** butonuna basarak tespiti başlatın.

---

## 📁 Proje Dizin Yapısı

- `main.py`: Uygulamanın giriş noktası (Splash Screen ve High-DPI başlatma).
- `main_window.py`: PyQt6 tabanlı fütüristik kullanıcı arayüzü ve video HUD çizimleri.
- `detector.py`: YOLOv11 plaka tespit sınıfı, EasyOCR görüntü iyileştirme ve Türk plakası regex filtreleri.
- `video_worker.py`: Donmayı engelleyen QThread arka plan video, RTSP ve kamera işleme motoru.
- `plate_history.py`: Tespit edilen plakaların kaydedilmesi, alarmların yönetimi ve CSV dışa aktarımı.
- `styles.py`: Glassmorphism ve Dark Theme (Obsidian & Neon) QSS stil sayfası.
- `records/`: Tespit edilen normal plakaların (`detections/`) ve aranan hedef alarmlarının (`alerts/`) saklandığı klasör.
