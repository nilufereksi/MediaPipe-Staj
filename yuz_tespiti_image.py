"""
yuz_tespiti_image.py — MediFace AI Statik Görüntü Loggeri
==========================================================
Tek bir fotoğraf ya da tüm bir klasördeki görüntüleri işler;
data_logger.py ile AYNI CSV formatında AU koordinatlarını kaydeder.

Bu dosya data_logger.py'nin VIDEO modunun kardeşidir:
  data_logger.py       → kamera / video  (VIDEO  mode)
  yuz_tespiti_image.py → fotoğraf / batch (IMAGE mode)

Her ikisi de aynı data_logs/ klasörüne, aynı sütun yapısıyla yazar.

Kullanım:
  python yuz_tespiti_image.py                   # tkinter ile tek fotoğraf
  python yuz_tespiti_image.py --folder datasets/veri/yuz/sinif_adi/
  python yuz_tespiti_image.py --folder datasets/veri/yuz/ --recursive
"""

import argparse
import csv
import os
import time
from datetime import datetime
from pathlib import Path

import cv2
import mediapipe as mp

# ─── Paylaşılan AU haritası (data_logger.py ile aynı tanım) ──────────────────
#
# İleride bu iki dosya büyüyünce au_landmarks.py adında ayrı bir utils modülüne
# taşıyabilirsin; şimdilik buraya kopyalanmış durumda.
AU_LANDMARKS: dict[str, list[int]] = {
    "AU4_sol_kas":   [65, 66, 107, 105, 70],
    "AU4_sag_kas":   [295, 285, 336, 334, 300],
    "AU6_sol_yanak": [46, 53, 52, 65, 159],
    "AU6_sag_yanak": [276, 283, 282, 295, 386],
    "AU7_sol_goz":   [159, 160, 161, 163, 144],
    "AU7_sag_goz":   [386, 387, 388, 390, 373],
    "AU9_burun":     [4, 5, 197, 195, 168],
    "AU43_sol":      [159, 145, 153, 154, 155],
    "AU43_sag":      [386, 374, 380, 381, 382],
    "AU12_sol_agiz": [61, 185, 40, 39, 37],
    "AU12_sag_agiz": [291, 409, 270, 269, 267],
    "AU17_cene":     [17, 200, 199, 175, 152],
    "AU20_dudak":    [61, 291, 306, 76, 77],
    "AU25_ac":       [13, 14, 78, 308, 82, 312],
    "AU26_cene_du":  [152, 148, 176, 377, 400],
    "REF_burun":     [168, 6, 197, 195],
}

AU_COLORS: dict[str, tuple[int, int, int]] = {
    "AU4":  (0, 165, 255),
    "AU6":  (255, 220, 0),
    "AU7":  (0, 230, 230),
    "AU9":  (0, 255, 150),
    "AU43": (100, 200, 255),
    "AU12": (255, 80, 200),
    "AU17": (200, 100, 255),
    "AU20": (255, 140, 180),
    "AU25": (255, 180, 100),
    "AU26": (180, 140, 255),
    "REF":  (140, 140, 140),
}

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _au_color(au_name: str) -> tuple[int, int, int]:
    for prefix, color in AU_COLORS.items():
        if au_name.startswith(prefix):
            return color
    return (180, 180, 180)


def build_csv_headers() -> list[str]:
    """data_logger.py ile aynı başlık yapısı + görüntü kaynağı sütunları."""
    headers = [
        "timestamp_ms",
        "session_id",
        "frame_id",       # görüntü sırası
        "window_id",      # görüntüde 0 (video gibi window yok)
        "face_count",
        "focus_face_idx",
        "bbox_area_px",
        "source_path",    # ek: hangi dosyadan geldi
        "source_type",    # "image" (video'da bu sütun olmaz, birleşince NaN olur — normal)
    ]
    for au_name, indices in AU_LANDMARKS.items():
        for idx in indices:
            headers += [
                f"{au_name}_p{idx}_x",
                f"{au_name}_p{idx}_y",
                f"{au_name}_p{idx}_z",
            ]
    return headers


def _focus_face(face_landmarks_list, frame_w: int, frame_h: int) -> tuple[int, float]:
    """En büyük bounding-box'a sahip yüzü seç (data_logger ile aynı mantık)."""
    best_idx, max_area = 0, 0.0
    for i, lm in enumerate(face_landmarks_list):
        xs = [p.x for p in lm]
        ys = [p.y for p in lm]
        area = (max(xs) - min(xs)) * frame_w * (max(ys) - min(ys)) * frame_h
        if area > max_area:
            max_area = area
            best_idx = i
    return best_idx, max_area


def process_image(
    image_path: str,
    landmarker,
    frame_id: int,
    session_id: str,
    timestamp_ms: int,
) -> list | None:
    """
    Tek bir görüntü dosyasını işler.
    Yüz bulunursa CSV satırını döndürür, bulunamazsa None.
    """
    frame = cv2.imread(image_path)
    if frame is None:
        print(f"  [ATLA] Okunamadı: {image_path}")
        return None

    h, w = frame.shape[:2]

    # MediaPipe IMAGE modu: cv2.imread BGR verir, RGB'ye çevir
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

    result = landmarker.detect(mp_image)

    if not result.face_landmarks:
        print(f"  [YOK]  Yüz bulunamadı: {os.path.basename(image_path)}")
        return None

    face_count = len(result.face_landmarks)
    focus_idx, bbox_area = _focus_face(result.face_landmarks, w, h)
    face_lm = result.face_landmarks[focus_idx]

    # Referans noktası (burun köprüsü) normalizasyon için
    ref_p = face_lm[AU_LANDMARKS["REF_burun"][0]]
    ref_x, ref_y = ref_p.x, ref_p.y

    row: list = [
        timestamp_ms,
        session_id,
        frame_id,
        0,              # window_id — statik görüntüde anlamsız, 0 yaz
        face_count,
        focus_idx,
        round(bbox_area, 2),
        image_path,
        "image",
    ]

    for au_name, indices in AU_LANDMARKS.items():
        for idx in indices:
            if idx < len(face_lm):
                p = face_lm[idx]
                row += [
                    round(p.x - ref_x, 6),
                    round(p.y - ref_y, 6),
                    round(p.z, 6),
                ]
            else:
                row += [0.0, 0.0, 0.0]

    print(f"  [OK]   {os.path.basename(image_path)}  "
          f"(yüz: {face_count}, alan: {bbox_area:.0f}px²)")
    return row


def draw_au_overlay(image_path: str, result, focus_idx: int) -> None:
    """İşlenen görüntüyü AU noktalarıyla ekranda gösterir."""
    frame = cv2.imread(image_path)
    if frame is None:
        return
    h, w = frame.shape[:2]
    face_lm = result.face_landmarks[focus_idx]

    for au_name, indices in AU_LANDMARKS.items():
        color = _au_color(au_name)
        for idx in indices:
            if idx < len(face_lm):
                p = face_lm[idx]
                cv2.circle(frame, (int(p.x * w), int(p.y * h)), 3, color, -1)

    cv2.putText(frame, "AU noktalari  [herhangi bir tus: devam]",
                (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)
    cv2.imshow("MediFace AI — Goruntu", frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def collect_image_paths(folder: str, recursive: bool) -> list[str]:
    """Klasördeki desteklenen görüntü yollarını toplar."""
    root = Path(folder)
    pattern = "**/*" if recursive else "*"
    paths = [
        str(p) for p in root.glob(pattern)
        if p.suffix.lower() in SUPPORTED_EXTENSIONS
    ]
    return sorted(paths)


# ─── Ana Fonksiyon ────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="MediFace AI — Statik Görüntü AU Loggeri"
    )
    parser.add_argument("--folder",    type=str, default=None,
                        help="Toplu işlenecek klasör yolu")
    parser.add_argument("--recursive", action="store_true",
                        help="Alt klasörleri de tara")
    parser.add_argument("--output",    type=str, default="data_logs",
                        help="CSV çıktı klasörü (data_logger ile aynı)")
    parser.add_argument("--model",     type=str, default="models/face_landmarker.task",
                        help="face_landmarker.task dosya yolu")
    parser.add_argument("--preview",   action="store_true",
                        help="Her görüntüyü AU overlay ile göster")
    args = parser.parse_args()

    # ── Görüntü listesini belirle ─────────────────────────────────────────────
    if args.folder:
        image_paths = collect_image_paths(args.folder, args.recursive)
        if not image_paths:
            print(f"[HATA] Klasörde desteklenen görüntü yok: {args.folder}")
            return
        print(f"[ImageLogger] {len(image_paths)} görüntü bulundu: {args.folder}")
    else:
        # Tkinter dosya seçici (tek görüntü modu — orijinal davranış)
        try:
            import tkinter as tk
            from tkinter import filedialog
            root_tk = tk.Tk()
            root_tk.withdraw()
            file_path = filedialog.askopenfilename(
                title="Bir fotoğraf seçin",
                filetypes=[("Resim dosyaları", "*.jpg *.jpeg *.png *.bmp *.webp")]
            )
            root_tk.destroy()
        except ImportError:
            print("[HATA] tkinter bulunamadı. --folder parametresiyle kullan.")
            return

        if not file_path:
            print("Dosya seçilmedi.")
            return
        image_paths = [file_path]

    # ── MediaPipe IMAGE modu kurulum ──────────────────────────────────────────
    BaseOptions           = mp.tasks.BaseOptions
    FaceLandmarker        = mp.tasks.vision.FaceLandmarker
    FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
    VisionRunningMode     = mp.tasks.vision.RunningMode

    options = FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=args.model),
        running_mode=VisionRunningMode.IMAGE,   # VIDEO değil IMAGE!
        num_faces=4,
        min_face_detection_confidence=0.5,
    )

    # ── CSV hazırlık ──────────────────────────────────────────────────────────
    os.makedirs(args.output, exist_ok=True)
    session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.join(args.output, f"au_image_log_{session_id}.csv")

    found, skipped = 0, 0

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(build_csv_headers())

        _last_ts = -1
        with FaceLandmarker.create_from_options(options) as landmarker:
            for frame_id, img_path in enumerate(image_paths):
                # Monoton timestamp — aynı ms'de birden fazla görüntü işlenebilir
                timestamp_ms = int(time.time() * 1000)
                if timestamp_ms <= _last_ts:
                    timestamp_ms = _last_ts + 1
                _last_ts = timestamp_ms

                row = process_image(img_path, landmarker, frame_id, session_id, timestamp_ms)

                if row is not None:
                    writer.writerow(row)
                    found += 1

                    if args.preview:
                        # Görüntüyü tekrar işle sadece overlay için
                        _frame = cv2.imread(img_path)
                        if _frame is not None:
                            h, w = _frame.shape[:2]
                            rgb = cv2.cvtColor(_frame, cv2.COLOR_BGR2RGB)
                            _mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                            _res = landmarker.detect(_mp_img)
                            if _res.face_landmarks:
                                from mediapipe.tasks.python import vision as _v
                                draw_au_overlay(img_path, _res, 0)
                else:
                    skipped += 1

    print(f"\n[ImageLogger] Tamamlandı.")
    print(f"  İşlenen  : {len(image_paths)}")
    print(f"  Kaydedilen: {found}")
    print(f"  Atlanan  : {skipped} (yüz yok / okunamadı)")
    print(f"  CSV      : {csv_path}")


if __name__ == "__main__":
    main()
