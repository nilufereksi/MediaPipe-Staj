"""
mesh.py — MediFace AI Yüz Mesh Görselleştiricisi
=================================================
Kameradan canlı olarak tam yüz mesh'ini çizer;
PSPI ve Affective AU gruplarını renkli katman olarak üstüne ekler.

Bu dosya geliştirme/debug aracıdır — CSV kaydetmez.
Hangi AU noktalarının doğru yerde olduğunu görsel olarak doğrulamak için kullan.

Bağımlılıklar:
  pip install mediapipe opencv-python
  connections.py  — aynı klasörde olmalı
  face_landmarker.task — models/ klasöründe veya aynı dizinde

Kullanım:
  python mesh.py
  python mesh.py --model models/face_landmarker.task
  python mesh.py --no-mesh     # sadece AU noktaları, mesh yok
"""

import argparse
import time

import cv2
import mediapipe as mp

from connections import FACEMESH_TESSELATION, FACEMESH_CONTOURS, FACEMESH_IRISES

# ─── AU Landmark Haritası (data_logger.py ile özdeş) ─────────────────────────
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

MESH_COLOR    = (195, 225, 245)   # natürel krem/bej — mesh bağlantıları
CONTOUR_COLOR = (180, 210, 235)   # yumuşak cilt alt tonu — yüz konturu
IRIS_COLOR    = (255, 180, 0)     # sarı — iris


def _au_color(au_name: str) -> tuple[int, int, int]:
    for prefix, color in AU_COLORS.items():
        if au_name.startswith(prefix):
            return color
    return (180, 180, 180)


def draw_mesh(frame, landmarks, w: int, h: int, show_mesh: bool) -> None:
    """Yüz mesh bağlantılarını çizer (opsiyonel)."""
    if not show_mesh:
        return
    pts = [(int(p.x * w), int(p.y * h)) for p in landmarks]

    for s, e in FACEMESH_TESSELATION:
        if s < len(pts) and e < len(pts):
            cv2.line(frame, pts[s], pts[e], MESH_COLOR, 1)

    for s, e in FACEMESH_CONTOURS:
        if s < len(pts) and e < len(pts):
            cv2.line(frame, pts[s], pts[e], CONTOUR_COLOR, 1)

    for s, e in FACEMESH_IRISES:
        if s < len(pts) and e < len(pts):
            cv2.line(frame, pts[s], pts[e], IRIS_COLOR, 1)


def draw_au_points(frame, landmarks, w: int, h: int) -> None:
    """AU noktalarını renk grubuna göre büyük dairelerle çizer."""
    for au_name, indices in AU_LANDMARKS.items():
        color = _au_color(au_name)
        for idx in indices:
            if idx < len(landmarks):
                p = landmarks[idx]
                cx, cy = int(p.x * w), int(p.y * h)
                cv2.circle(frame, (cx, cy), 4, color, -1)
                cv2.circle(frame, (cx, cy), 4, (0, 0, 0), 1)  # siyah kenar


def draw_legend(frame, h: int) -> None:
    """Sol alt köşeye AU renk açıklaması yazar."""
    legend = [
        ((0, 165, 255),   "AU4  Kas catma"),
        ((0, 230, 230),   "AU6/7 Goz"),
        ((0, 255, 150),   "AU9  Burun"),
        ((100, 200, 255), "AU43 Goz kapanma"),
        ((255, 80, 200),  "AU12-20 Agiz/Dudak"),
        ((180, 140, 255), "AU25/26 Cene"),
        ((140, 140, 140), "REF  Normalizasyon"),
    ]
    base_y = h - 20 - len(legend) * 20
    for j, (clr, lbl) in enumerate(legend):
        y = base_y + j * 20
        cv2.circle(frame, (14, y), 6, clr, -1)
        cv2.putText(frame, lbl, (26, y + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (210, 210, 210), 1)


def main() -> None:
    parser = argparse.ArgumentParser(description="MediFace AI — Yüz Mesh Görselleştirici")
    parser.add_argument("--model",   default="face_landmarker.task",
                        help="face_landmarker.task yolu")
    parser.add_argument("--no-mesh", action="store_true",
                        help="Mesh bağlantılarını gizle, sadece AU noktaları")
    parser.add_argument("--source",  default="0",
                        help="Kamera indeksi veya video dosyası")
    args = parser.parse_args()

    source = int(args.source) if args.source.isdigit() else args.source
    show_mesh = not args.no_mesh

    BaseOptions           = mp.tasks.BaseOptions
    FaceLandmarker        = mp.tasks.vision.FaceLandmarker
    FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
    VisionRunningMode     = mp.tasks.vision.RunningMode

    options = FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=args.model),
        running_mode=VisionRunningMode.VIDEO,
        num_faces=1,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"[HATA] Kaynak açılamadı: {source}")
        return

    _last_ts = -1

    print("[mesh.py] Başlatıldı. Çıkış: q veya ESC")

    with FaceLandmarker.create_from_options(options) as landmarker:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            h, w = frame.shape[:2]

            timestamp_ms = int(time.time() * 1000)
            if timestamp_ms <= _last_ts:
                timestamp_ms = _last_ts + 1
            _last_ts = timestamp_ms

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = landmarker.detect_for_video(mp_image, timestamp_ms)

            if result.face_landmarks:
                lm = result.face_landmarks[0]
                draw_mesh(frame, lm, w, h, show_mesh)
                draw_au_points(frame, lm, w, h)
                cv2.putText(frame, "YUZ TESPIT EDILDI", (w - 240, 28),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 230, 120), 1)
            else:
                cv2.putText(frame, "Yuz bulunamadi", (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 60, 220), 2)

            draw_legend(frame, h)
            cv2.imshow("MediFace AI — Mesh  [q: cikis]", frame)

            if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
