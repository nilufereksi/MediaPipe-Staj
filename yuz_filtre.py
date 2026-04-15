"""
yuz_filtre.py — MediFace AI Veri Seti Ön Filtreleyici
======================================================
Kaynak klasördeki görüntüleri tarar; yüz içerenleri ve içermeyenleri
ayrı klasörlere kopyalar.

Projedeki rolü:
  Dermnet / SCIN / JMIR gibi ham veri setlerini temizleyip
  data_logs/ klasörüne yönlendirilecek yüz görüntülerini hazırlamak.

  Ham veri seti → yuz_filtre.py → datasets/veri/yuz/<sinif>/
                                 → datasets/veri/yuz_disi/<sinif>/

  Sonraki adım: yuz_tespiti_image.py --folder datasets/veri/yuz/

Kullanım:
  python yuz_filtre.py                              # varsayılan: dermnet/
  python yuz_filtre.py --source baska_dataset/
  python yuz_filtre.py --source dermnet/ --conf 0.8 --min-size 80

Bağımlılıklar:
  pip install mediapipe opencv-python
  models/blaze_face_short_range.tflite  — aşağıdaki linkten indir:
  https://storage.googleapis.com/mediapipe-assets/blaze_face_short_range.tflite
"""

import argparse
import os
import shutil

import cv2
import mediapipe as mp
from mediapipe.tasks.python import vision


def main() -> None:
    parser = argparse.ArgumentParser(
        description="MediFace AI — Yüz / Yüz-dışı Filtreleyici"
    )
    parser.add_argument("--source",   default="dermnet",
                        help="Kaynak veri seti klasörü (varsayılan: dermnet/)")
    parser.add_argument("--output",   default="datasets/veri",
                        help="Çıktı kök klasörü (varsayılan: datasets/veri/)")
    parser.add_argument("--model",    default="models/blaze_face_short_range.tflite",
                        help="blaze_face_short_range.tflite yolu")
    parser.add_argument("--conf",     type=float, default=0.7,
                        help="Yüz güven eşiği 0–1 (varsayılan: 0.7)")
    parser.add_argument("--min-size", type=int,   default=50,
                        help="Min bbox boyutu px (varsayılan: 50)")
    args = parser.parse_args()

    # ── Çıktı klasörleri ──────────────────────────────────────────────────────
    out_yuz      = os.path.join(args.output, "yuz")
    out_yuz_disi = os.path.join(args.output, "yuz_disi")
    os.makedirs(out_yuz,      exist_ok=True)
    os.makedirs(out_yuz_disi, exist_ok=True)

    # ── MediaPipe BlazeFace kurulum ───────────────────────────────────────────
    if not os.path.exists(args.model):
        print(f"[HATA] Model bulunamadı: {args.model}")
        print("  İndir: https://storage.googleapis.com/mediapipe-assets/"
              "blaze_face_short_range.tflite")
        return

    BaseOptions      = mp.tasks.BaseOptions
    FaceDetector     = vision.FaceDetector
    FaceDetectorOptions = vision.FaceDetectorOptions
    VisionRunningMode   = vision.RunningMode

    options = FaceDetectorOptions(
        base_options=BaseOptions(model_asset_path=args.model),
        running_mode=VisionRunningMode.IMAGE,
        min_detection_confidence=args.conf,
    )

    detector = FaceDetector.create_from_options(options)

    # ── Klasör tarama ─────────────────────────────────────────────────────────
    sayac = {"yuz": 0, "yuz_disi": 0, "atlanan": 0}

    if not os.path.isdir(args.source):
        print(f"[HATA] Kaynak klasör bulunamadı: {args.source}")
        return

    siniflar = [s for s in os.listdir(args.source)
                if os.path.isdir(os.path.join(args.source, s))]

    if not siniflar:
        # Düz klasör (sınıf alt klasörü yok) — tek sınıf gibi işle
        siniflar = [""]

    for sinif in siniflar:
        sinif_yolu = os.path.join(args.source, sinif) if sinif else args.source

        cikis_yuz      = os.path.join(out_yuz,      sinif) if sinif else out_yuz
        cikis_yuz_disi = os.path.join(out_yuz_disi, sinif) if sinif else out_yuz_disi
        os.makedirs(cikis_yuz,      exist_ok=True)
        os.makedirs(cikis_yuz_disi, exist_ok=True)

        dosyalar = [
            d for d in os.listdir(sinif_yolu)
            if d.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp"))
        ]

        for dosya in dosyalar:
            tam_yol = os.path.join(sinif_yolu, dosya)
            img = cv2.imread(tam_yol)
            if img is None:
                sayac["atlanan"] += 1
                continue

            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

            try:
                sonuc = detector.detect(mp_image)
            except Exception as e:
                print(f"  [HATA] {dosya}: {e}")
                sayac["atlanan"] += 1
                continue

            yuz_var = False
            if sonuc.detections:
                for detection in sonuc.detections:
                    score = detection.categories[0].score
                    if score < args.conf:
                        continue
                    bbox = detection.bounding_box
                    if bbox.width < args.min_size or bbox.height < args.min_size:
                        continue
                    yuz_var = True
                    break

            if yuz_var:
                shutil.copy(tam_yol, os.path.join(cikis_yuz, dosya))
                sayac["yuz"] += 1
            else:
                shutil.copy(tam_yol, os.path.join(cikis_yuz_disi, dosya))
                sayac["yuz_disi"] += 1

    detector.close()

    print(f"\n[yuz_filtre] Tamamlandı.")
    print(f"  Yüz içeren : {sayac['yuz']}")
    print(f"  Yüz dışı   : {sayac['yuz_disi']}")
    print(f"  Atlanan    : {sayac['atlanan']} (okunamadı/hata)")
    print(f"  Yüzler     : {out_yuz}")
    print(f"  Bir sonraki adım:")
    print(f"    python yuz_tespiti_image.py --folder {out_yuz} --recursive")


if __name__ == "__main__":
    main()
