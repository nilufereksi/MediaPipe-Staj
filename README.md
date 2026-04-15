# MediFace AI

> MediaPipe tabanlı yüz landmark tespiti, AU koordinatı kaydı ve mesh görselleştirmesi sunan, gizlilik odaklı gerçek zamanlı yüz analizi kontrol paneli.

<img width="1141" height="679" alt="resim" src="https://github.com/user-attachments/assets/d3004ec3-91ad-4d67-8498-58d217dc84f0" />

---

## 🚀 Özellikler

| Mod | Açıklama |
|-----|----------|
| **Data Logger** | Kamera veya videodan Action Unit (AU) koordinatlarını CSV olarak kaydeder |
| **Yüz Mesh** | 468 noktalı MediaPipe yüz mesh'ini canlı olarak görselleştirir |
| **Video Monitor** | Gerçek zamanlı çoklu yüz tespiti ve takibi |
| **Görüntü Logger** | Fotoğraf klasöründen toplu AU koordinatı çıkarımı |
| **Yüz Filtre** | BlazeFace ile veri seti ön-işleme ve yüz kırpma aracı |

## 🔒 Gizlilik

Kamera görüntüleri **asla diske yazılmaz.** Yalnızca sayısal landmark koordinatları (CSV) saklanır.

---

## 🛠️ Kurulum

```bash
# 1. Depoyu klonla
git clone https://github.com/kullanici-adin/mediface-ai.git
cd mediface-ai

# 2. Bağımlılıkları yükle
pip install -r requirements.txt

# 3. Model dosyalarını indir ve models/ klasörüne koy
#    face_landmarker.task  → https://developers.google.com/mediapipe/solutions/vision/face_landmarker
#    blaze_face_short_range.tflite (Yüz Filtre modu için)

# 4. Uygulamayı başlat
python main_gui.py
```

## 📦 Gereksinimler

- Python 3.9+
- `opencv-python >= 4.8.0`
- `mediapipe >= 0.10.0`
- `Pillow` (GUI kamera görüntüsü için)

---

## 📁 Proje Yapısı

```
mediface-ai/
├── main_gui.py          # Ana kontrol paneli (Tkinter GUI)
├── data_logger.py       # AU koordinatı kayıt motoru
├── mesh.py              # Yüz mesh çizim araçları
├── yuz_tespit_video.py  # Video izleme modülü
├── yuz_tespiti_image.py # Görüntü tabanlı tespit
├── yuz_filtre.py        # Veri seti ön-işleme
├── connections.py       # MediaPipe bağlantı haritası
├── models/              # Model dosyaları (git'e dahil değil)
├── data_logs/           # CSV çıktıları (git'e dahil değil)
└── datasets/            # İşlenmiş veri seti (git'e dahil değil)
```

---

## 📄 Lisans

MIT License — ayrıntılar için `LICENSE` dosyasına bakın.
