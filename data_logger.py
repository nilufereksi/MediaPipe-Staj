"""
MediFace AI — Aşama 1: Data Logger Modülü
==========================================
Görev:
  MediaPipe FaceLandmarker ile yüzü tespit et,
  PSPI ve duygu analizi için kritik Action Unit (AU) noktalarını
  filtrele, normalize et ve CSV'ye kaydet.

Kaydedilen AU Grupları
──────────────────────
  PSPI için (Lucey vd. 2011 & Hammal & Cohn 2012):
    AU4  Brow Lowerer      – Corrugator supercilii
    AU6  Cheek Raiser      – Orbicularis oculi (orbital)
    AU7  Lid Tightener     – Orbicularis oculi (palpebral)
    AU9  Nose Wrinkler     – Levator labii superioris alaeque nasi
    AU43 Eyes Closed       – Palpebral fissure yüksekliği

  PSPI Formülü: AU4 + max(AU6, AU7) + max(AU9, AU43)

  Duygu için ek (CK+ / FER2013 etiket yapısına uygun):
    AU12 Lip Corner Puller – Zygomaticus major   (mutluluk)
    AU17 Chin Raiser       – Mentalis            (üzüntü)
    AU20 Lip Stretcher     – Risorius            (korku)
    AU25 Lips Part         – Depressor labii
    AU26 Jaw Drop          – Relaxed masseter

  Referans (normalizasyon):
    REF_burun_koprusu      – Yüz ölçeği sabitleyici

Özellikler
──────────
  * Çoklu yüz → en büyük bounding-box'a kilitlenme (Focus)
  * 10 saniyelik Sliding Window ile periyodik segmentasyon
  * Zaman damgalı (ms), session-ID ve frame-ID ile tam loglama
  * data_logs/au_log_<session>.csv çıktısı

Bağımlılıklar:  mediapipe >= 0.10  |  opencv-python  |  Python 3.9+
Model dosyası:  face_landmarker.task  (aynı dizinde olmalı)

Kullanım:
  python data_logger.py                    # varsayılan webcam
  python data_logger.py --source video.mp4 # video dosyası
  python data_logger.py --window 15        # 15 s pencere
"""

import argparse
import csv
import os
import time
from datetime import datetime

import cv2
import mediapipe as mp

# ─── ACTION UNIT → LANDMARK İNDEKS HARİTASI ────────────────────────────────────
# MediaPipe Face Mesh 468 nokta koordinat sistemi üzerinden tanımlanmıştır.
# Kaynak: Lucey vd. (2011) FACS/PSPI + Hammal & Cohn (2012) AU haritaları
AU_LANDMARKS: dict[str, list[int]] = {
    # ── PSPI Çekirdeği ──────────────────────────────────────────────────────────
    "AU4_sol_kas":   [65, 66, 107, 105, 70],       # Sol kaş çatma (corrugator)
    "AU4_sag_kas":   [295, 285, 336, 334, 300],    # Sağ kaş çatma
    "AU6_sol_yanak": [46, 53, 52, 65, 159],        # Sol yanak yükseltme
    "AU6_sag_yanak": [276, 283, 282, 295, 386],    # Sağ yanak yükseltme
    "AU7_sol_goz":   [159, 160, 161, 163, 144],    # Sol göz kapağı germe
    "AU7_sag_goz":   [386, 387, 388, 390, 373],    # Sağ göz kapağı germe
    "AU9_burun":     [4, 5, 197, 195, 168],        # Burun kırışma
    "AU10_ust_dudak": [0, 11, 12, 37, 267, 269, 270, 314],  # Levator labii superioris
    "AU43_sol":      [159, 145, 153, 154, 155],    # Sol palpebral fissure
    "AU43_sag":      [386, 374, 380, 381, 382],    # Sağ palpebral fissure

    # ── Duygu / Sentiment ───────────────────────────────────────────────────────
    "AU12_sol_agiz": [61, 185, 40, 39, 37],        # Sol lip corner puller
    "AU12_sag_agiz": [291, 409, 270, 269, 267],    # Sağ lip corner puller
    "AU17_cene":     [17, 200, 199, 175, 152],     # Çene yükseltme (mentalis)
    "AU20_dudak":    [61, 291, 306, 76, 77],       # Dudak germe (risorius)
    "AU25_ac":       [13, 14, 78, 308, 82, 312],   # Dudak açılma
    "AU26_cene_du":  [152, 148, 176, 377, 400],    # Çene düşürme

    # ── Referans (Normalizasyon) ─────────────────────────────────────────────────
    "REF_burun":     [168, 6, 197, 195],            # Burun köprüsü — ölçek sabiti
}

# Renk paleti: AU gruplarına görsel ayrım
AU_COLORS: dict[str, tuple[int, int, int]] = {
    "AU4":  (0, 165, 255),   # Turuncu   — kaş
    "AU6":  (255, 220, 0),   # Sarı-cyan — yanak
    "AU7":  (0, 230, 230),   # Cyan      — göz kapağı
    "AU9":  (0, 255, 150),   # Yeşil     — burun
    "AU43": (100, 200, 255), # Açık mavi — göz kapanma
    "AU12": (255, 80, 200),  # Pembe     — ağız köşesi
    "AU17": (200, 100, 255), # Mor       — çene
    "AU20": (255, 140, 180), # Gül pembe — dudak
    "AU25": (255, 180, 100), # Şeftali   — dudak açma
    "AU26": (180, 140, 255), # Lavanta   — çene düş.
    "REF":  (140, 140, 140), # Gri       — referans
}


def _au_color(au_name: str) -> tuple[int, int, int]:
    """AU adından BGR rengi döndürür."""
    for prefix, color in AU_COLORS.items():
        if au_name.startswith(prefix):
            return color
    return (180, 180, 180)


def build_csv_headers() -> list[str]:
    """CSV sütun başlıklarını oluşturur."""
    headers = [
        "timestamp_ms",     # Mutlak zaman (ms)
        "session_id",       # Oturum kimliği
        "frame_id",         # Kare numarası
        "window_id",        # 10 s pencere numarası
        "face_count",       # Karede kaç yüz var
        "focus_face_idx",   # Seçilen yüz indeksi
        "bbox_area_px",     # Seçilen yüzün bounding-box alanı (piksel²)
    ]
    for au_name, indices in AU_LANDMARKS.items():
        for idx in indices:
            headers += [
                f"{au_name}_p{idx}_x",   # Normalize edilmiş x
                f"{au_name}_p{idx}_y",   # Normalize edilmiş y
                f"{au_name}_p{idx}_z",   # Derinlik (z)
            ]
    return headers


# ─── DATA LOGGER SINIFI ─────────────────────────────────────────────────────────

class MediFaceDataLogger:
    """
    MediaPipe AU noktalarını gerçek zamanlı olarak CSV'ye kaydeden modül.

    Temel davranışlar
    -----------------
    focus_face()     : Çoklu yüzde en büyük bbox'ı seçer.
    process_frame()  : Bir kareyi işler, CSV satırı yazar.
    draw_overlay()   : Görselleştirme katmanını kareye ekler.
    close()          : CSV dosyasını güvenle kapatır.
    """

    def __init__(
        self,
        output_dir: str = "data_logs",
        window_sec: int = 10,
        model_path: str = "models/face_landmarker.task",
        max_faces: int = 4,
    ):
        self.window_sec = window_sec
        self.output_dir = output_dir
        self.model_path = model_path
        os.makedirs(output_dir, exist_ok=True)

        # Oturum kimliği
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.frame_id = 0
        self.window_id = 0
        self._window_start_ms: int | None = None

        # CSV dosyası
        csv_path = os.path.join(output_dir, f"au_log_{self.session_id}.csv")
        self._csv_file = open(csv_path, "w", newline="", encoding="utf-8")
        self._writer = csv.writer(self._csv_file)
        self._writer.writerow(build_csv_headers())
        print(f"[DataLogger] Kayıt başladı → {csv_path}")

        # MediaPipe seçenekleri
        BaseOptions = mp.tasks.BaseOptions
        FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
        VisionRunningMode = mp.tasks.vision.RunningMode

        self.options = FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=self.model_path),
            running_mode=VisionRunningMode.VIDEO,
            num_faces=max_faces,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )

    # ── Yardımcı Metodlar ───────────────────────────────────────────────────────

    def _focus_face(
        self,
        face_landmarks_list: list,
        frame_w: int,
        frame_h: int,
    ) -> tuple[int, float]:
        """
        Birden fazla yüz tespit edildiğinde en büyük bounding-box alanına
        sahip yüzün indeksini ve piksel² cinsinden alanını döndürür.
        (Mentor notu: 'Focus ayarlayın, 2 kişi varsa tek kişiyle odaklanacak')
        """
        best_idx, max_area = 0, 0.0
        for i, lm in enumerate(face_landmarks_list):
            xs = [p.x for p in lm]
            ys = [p.y for p in lm]
            area = (max(xs) - min(xs)) * frame_w * (max(ys) - min(ys)) * frame_h
            if area > max_area:
                max_area = area
                best_idx = i
        return best_idx, max_area

    def _update_window(self, timestamp_ms: int) -> None:
        """Sliding window sayacını günceller."""
        if self._window_start_ms is None:
            self._window_start_ms = timestamp_ms
        elapsed_s = (timestamp_ms - self._window_start_ms) / 1000.0
        if elapsed_s >= self.window_sec:
            self.window_id += 1
            self._window_start_ms = timestamp_ms
            print(
                f"[DataLogger] Window {self.window_id} başladı "
                f"(frame {self.frame_id}, {timestamp_ms} ms)"
            )

    # ── Ana İşlem Metodu ─────────────────────────────────────────────────────────

    def process_frame(
        self,
        frame,           # numpy ndarray (BGR)
        timestamp_ms: int,
        landmarker,      # FaceLandmarker nesnesi
    ) -> tuple[int | None, object]:
        """
        Tek bir kareyi işler: landmark tespiti → AU filtresi → CSV yazma.

        Dönüş: (focus_face_idx, mediapipe_result)
        Yüz bulunamazsa (None, result) döner.
        """
        h, w = frame.shape[:2]
        self._update_window(timestamp_ms)

        # BGR → RGB ve MediaPipe imajı
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = landmarker.detect_for_video(mp_image, timestamp_ms)

        if not result.face_landmarks:
            self.frame_id += 1
            return None, result

        # Focus: en büyük yüzü seç
        face_count = len(result.face_landmarks)
        focus_idx, bbox_area = self._focus_face(result.face_landmarks, w, h)
        face_lm = result.face_landmarks[focus_idx]

        # ── CSV satırı ──────────────────────────────────────────────────────────
        row: list = [
            timestamp_ms,
            self.session_id,
            self.frame_id,
            self.window_id,
            face_count,
            focus_idx,
            round(bbox_area, 2),
        ]

        # Referans noktası (burun köprüsü) — normalizasyon için
        ref_indices = AU_LANDMARKS["REF_burun"]
        if ref_indices and ref_indices[0] < len(face_lm):
            ref_p = face_lm[ref_indices[0]]
            ref_x, ref_y = ref_p.x, ref_p.y
        else:
            ref_x, ref_y = 0.5, 0.5

        # Her AU grubunun noktaları → normalize edilmiş koordinatlar
        for au_name, indices in AU_LANDMARKS.items():
            for idx in indices:
                if idx < len(face_lm):
                    p = face_lm[idx]
                    # Burun köprüsüne göre normalize et (daha kararlı özellikler)
                    row += [
                        round(p.x - ref_x, 6),   # delta_x
                        round(p.y - ref_y, 6),   # delta_y
                        round(p.z, 6),            # z derinliği (zaten normalize)
                    ]
                else:
                    row += [0.0, 0.0, 0.0]

        self._writer.writerow(row)
        self.frame_id += 1
        return focus_idx, result

    # ── Görsel Katman ────────────────────────────────────────────────────────────

    def draw_overlay(self, frame, result, focus_idx: int):
        """
        Kamerayı görselleştirir:
        - Seçili yüzün AU noktaları renkli noktalar
        - Diğer yüzler soluk gri
        - Sağ üstte bilgi paneli
        - Sol altta AU renk açıklaması
        """
        if result is None or not result.face_landmarks:
            cv2.putText(
                frame, "Yuz bulunamadi", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 220), 2,
            )
            return frame

        h, w = frame.shape[:2]

        # Diğer yüzleri soluk çiz
        for i, lm in enumerate(result.face_landmarks):
            if i == focus_idx:
                continue
            for p in lm:
                cv2.circle(
                    frame,
                    (int(p.x * w), int(p.y * h)),
                    1, (90, 90, 90), -1,
                )

        # Seçili yüzün AU noktaları
        face_lm = result.face_landmarks[focus_idx]
        for au_name, indices in AU_LANDMARKS.items():
            color = _au_color(au_name)
            for idx in indices:
                if idx < len(face_lm):
                    p = face_lm[idx]
                    cx, cy = int(p.x * w), int(p.y * h)
                    cv2.circle(frame, (cx, cy), 3, color, -1)

        # ── Bilgi Paneli (sağ üst) ──────────────────────────────────────────────
        panel_x = w - 260
        texts = [
            (f"Session: {self.session_id}", (200, 200, 200)),
            (f"Frame: {self.frame_id}  |  Window: {self.window_id}", (200, 200, 200)),
            (f"Yuz: {len(result.face_landmarks)}  |  Focus: #{focus_idx}", (80, 220, 80)),
        ]
        for j, (txt, clr) in enumerate(texts):
            cv2.putText(
                frame, txt, (panel_x, 28 + j * 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, clr, 1,
            )

        # ── AU Renk Açıklaması (sol alt) ────────────────────────────────────────
        legend = [
            ((0, 165, 255), "AU4 Kas catma"),
            ((0, 230, 230), "AU6/7 Goz"),
            ((0, 255, 150), "AU9 Burun"),
            ((100, 200, 255), "AU43 Goz kapanma"),
            ((255, 80, 200), "AU12-26 Agiz"),
        ]
        base_y = h - 25 - len(legend) * 20
        for j, (clr, lbl) in enumerate(legend):
            y = base_y + j * 20
            cv2.circle(frame, (14, y), 5, clr, -1)
            cv2.putText(
                frame, lbl, (24, y + 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200, 200, 200), 1,
            )

        return frame

    # ── Kapatma ─────────────────────────────────────────────────────────────────

    def close(self) -> None:
        """CSV dosyasını ve kaynakları güvenle kapatır."""
        self._csv_file.flush()
        self._csv_file.close()
        print(
            f"[DataLogger] Kayıt tamamlandı. "
            f"Toplam kare: {self.frame_id}  |  Window: {self.window_id + 1}"
        )


# ─── MAIN ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="MediFace AI — AU Landmark Data Logger"
    )
    parser.add_argument(
        "--source", type=str, default="0",
        help="Kamera indeksi (0,1,...) veya video dosyası yolu",
    )
    parser.add_argument(
        "--window", type=int, default=10,
        help="Sliding window süresi (saniye, varsayılan: 10)",
    )
    parser.add_argument(
        "--model", type=str, default="models/face_landmarker.task",
        help="MediaPipe model dosyası yolu",
    )
    parser.add_argument(
        "--output", type=str, default="data_logs",
        help="CSV çıktı klasörü",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Kaynak: webcam indeksi mi yoksa video dosyası mı?
    source = int(args.source) if args.source.isdigit() else args.source

    logger = MediFaceDataLogger(
        output_dir=args.output,
        window_sec=args.window,
        model_path=args.model,
    )

    FaceLandmarker = mp.tasks.vision.FaceLandmarker

    with FaceLandmarker.create_from_options(logger.options) as landmarker:
        cap = cv2.VideoCapture(source)

        if not cap.isOpened():
            print(f"[HATA] Kaynak açılamadı: {source}")
            logger.close()
            return

        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        print(
            f"[DataLogger] Başlatıldı  |  Kaynak: {source}  "
            f"|  FPS≈{fps:.0f}  |  Window: {args.window}s\n"
            "  [q / ESC] çıkış"
        )

        _last_ts = -1   # ← Monoton timestamp takibi
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Monoton timestamp — MediaPipe VIDEO modu zorunluluğu
            timestamp_ms = int(time.time() * 1000)
            if timestamp_ms <= _last_ts:
                timestamp_ms = _last_ts + 1
            _last_ts = timestamp_ms

            focus_idx, result = logger.process_frame(frame, timestamp_ms, landmarker)

            if focus_idx is not None:
                frame = logger.draw_overlay(frame, result, focus_idx)
            else:
                cv2.putText(
                    frame, "Yuz bulunamadi", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 220), 2,
                )

            cv2.imshow("MediFace AI — Data Logger (q/ESC: cikis)", frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):   # q veya ESC
                break

    cap.release()
    cv2.destroyAllWindows()
    logger.close()


if __name__ == "__main__":
    main()
