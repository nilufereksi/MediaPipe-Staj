"""
yuz_tespit_video.py — MediFace AI Canlı Yüz Tespit Monitörü
============================================================
Kamera veya video dosyasını gerçek zamanlı izler; yüz bulunup
bulunmadığını ve kaç yüz olduğunu ekranda gösterir.

Bu dosya veri KAYDETMEZ — yalnızca görsel izleme/test aracıdır.
Veri kaydetmek için: data_logger.py

Projedeki rolü:
  • Kamera açısı ve aydınlatmanın model için uygun olup olmadığını test et.
  • Kaç yüzün tespit edildiğini ve hangi yüze kilitlenileceğini gör.
  • data_logger.py'yi başlatmadan önce kurulumu doğrula.

Kullanım:
  python yuz_tespit_video.py
  python yuz_tespit_video.py --source video.mp4
  python yuz_tespit_video.py --faces 4    # çoklu yüz modu
"""

import argparse
import time

import cv2
import mediapipe as mp


def draw_face_boxes(frame, result, w: int, h: int) -> int:
    """
    Tespit edilen tüm yüzlerin etrafına kutu çizer.
    En büyük yüzü yeşil, diğerlerini sarı renkle işaretler.
    Yüz sayısını döndürür.
    """
    if not result.face_landmarks:
        return 0

    # En büyük yüzü bul
    best_idx, max_area = 0, 0.0
    for i, lm in enumerate(result.face_landmarks):
        xs = [p.x for p in lm]
        ys = [p.y for p in lm]
        area = (max(xs) - min(xs)) * (max(ys) - min(ys))
        if area > max_area:
            max_area = area
            best_idx = i

    for i, lm in enumerate(result.face_landmarks):
        xs = [int(p.x * w) for p in lm]
        ys = [int(p.y * h) for p in lm]
        x1, y1 = max(min(xs) - 10, 0), max(min(ys) - 10, 0)
        x2, y2 = min(max(xs) + 10, w), min(max(ys) + 10, h)

        if i == best_idx:
            color = (0, 230, 80)    # yeşil — seçili / odak yüz
            label = f"FOCUS #{i}"
            thickness = 2
        else:
            color = (0, 200, 240)   # sarı — diğer yüzler
            label = f"Yuz #{i}"
            thickness = 1

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
        cv2.putText(frame, label, (x1, y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

    return len(result.face_landmarks)


def draw_hud(frame, face_count: int, frame_id: int, w: int) -> None:
    """Sağ üst köşeye durum bilgisi yazar."""
    if face_count > 0:
        status     = f"YUZ: {face_count}"
        status_clr = (0, 230, 80)
        sub        = "data_logger.py ile kaydet"
        sub_clr    = (160, 160, 160)
    else:
        status     = "Yuz bulunamadi"
        status_clr = (0, 60, 220)
        sub        = "Kamera acisini duzelt"
        sub_clr    = (100, 100, 200)

    cv2.putText(frame, status, (12, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_clr, 2)
    cv2.putText(frame, sub, (12, 52),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, sub_clr, 1)
    cv2.putText(frame, f"frame: {frame_id}", (w - 130, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (130, 130, 130), 1)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="MediFace AI — Canlı Yüz Tespit Monitörü"
    )
    parser.add_argument("--source", default="0",
                        help="Kamera indeksi (0) veya video dosyası yolu")
    parser.add_argument("--model",  default="face_landmarker.task",
                        help="face_landmarker.task yolu")
    parser.add_argument("--faces",  type=int, default=2,
                        help="Max yüz sayısı (varsayılan: 2)")
    args = parser.parse_args()

    source = int(args.source) if args.source.isdigit() else args.source

    BaseOptions           = mp.tasks.BaseOptions
    FaceLandmarker        = mp.tasks.vision.FaceLandmarker
    FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
    VisionRunningMode     = mp.tasks.vision.RunningMode

    options = FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=args.model),
        running_mode=VisionRunningMode.VIDEO,
        num_faces=args.faces,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"[HATA] Kaynak açılamadı: {source}")
        return

    _last_ts = -1
    frame_id = 0
    print("[yuz_tespit_video] Başlatıldı. Çıkış: q veya ESC")

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

            face_count = draw_face_boxes(frame, result, w, h)
            draw_hud(frame, face_count, frame_id, w)

            cv2.imshow("MediFace AI — Yuz Tespiti  [q: cikis]", frame)

            if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                break

            frame_id += 1

    cap.release()
    cv2.destroyAllWindows()
    print(f"[yuz_tespit_video] Sonlandı. Toplam frame: {frame_id}")


if __name__ == "__main__":
    main()
