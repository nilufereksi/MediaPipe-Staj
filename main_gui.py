"""
main_gui.py — MediFace AI v7
================================
Lux-inspired dashboard — clean light theme + gold/navy accents
Layout: slim sidebar | camera hero | settings+stats+console
Style: Bootswatch Lux aesthetic (light, sophisticated, refined)
"""

import os, sys, time, queue, threading, subprocess
from datetime import datetime

import cv2, mediapipe as mp
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

try:
    from PIL import Image, ImageTk
    PIL_OK = True
except ImportError:
    PIL_OK = False

BASE_DIR      = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MODEL = os.path.join(BASE_DIR, "models", "face_landmarker.task")
BLAZE_MODEL   = os.path.join(BASE_DIR, "models", "blaze_face_short_range.tflite")
DATA_LOGS_DIR = os.path.join(BASE_DIR, "data_logs")
sys.path.insert(0, BASE_DIR)

# ─── Dark Grey Palette ───────────────────────────────────────────────────────
# Backgrounds — all grey, no white
BG      = "#2b2b2b"   # Main background
PANEL   = "#323232"   # Panel / right column
CARD    = "#3a3a3a"   # Card surface
CARDUP  = "#404040"   # Elevated card / selected
FIELD   = "#2b2b2b"   # Input field background
BORD    = "#4a4a4a"   # Border
BORD2   = "#606060"   # Hover border

# Accents — Gold + deep grey
GOLD    = "#c9a84c"   # Primary gold accent
GOLD2   = "#e0c070"   # Lighter gold (hover)
GOLD3   = "#9a7830"   # Deep gold (pressed)
NAVY    = "#1e1e1e"   # Sidebar (deepest grey)
NAVY2   = "#252525"   # Sidebar hover
NAVY3   = "#333333"   # Sidebar divider / badge bg

# Status colors
C_GRN   = "#4caf6e"
C_RED   = "#e05555"
C_AMB   = "#e6a817"
C_BLU   = "#5b9bd5"
C_TEAL  = "#3ab8c8"

# Text
T1 = "#e8e8e8"   # Primary text
T2 = "#b0b0b0"   # Secondary text
T3 = "#787878"   # Muted text
T4 = "#505050"   # Disabled text
TW = "#ffffff"   # White
TG = "#ede0c0"   # Gold-tinted text

# Sidebar text
ST1 = "#e8e0d0"  # Sidebar primary
ST2 = "#a89880"  # Sidebar secondary
ST3 = "#5a5248"  # Sidebar muted

FF   = "Segoe UI"
MONO = "Consolas"

_fs  = lambda s, b=False: (FF, s, "bold") if b else (FF, s)
_fm  = lambda s: (MONO, s)

# ─────────────────────────────────────────────────────────────────────────────
# Kamera Motoru
# ─────────────────────────────────────────────────────────────────────────────
class CameraEngine:
    MODE_DATA_LOGGER   = "data_logger"
    MODE_MESH          = "mesh"
    MODE_VIDEO_MONITOR = "video_monitor"

    def __init__(self, q, log_cb, stats_cb):
        self.frame_queue = q
        self.log_cb      = log_cb
        self.stats_cb    = stats_cb
        self._stop       = threading.Event()
        self._thread     = None
        self.active_mode = None

    def start(self, mode, source, model_path, **kw):
        if self.is_running():
            self.log_cb("[!] Zaten calisiyor – once durdurun."); return
        self._stop.clear()
        self.active_mode = mode
        self._thread = threading.Thread(
            target=self._run, args=(mode, source, model_path),
            kwargs=kw, daemon=True)
        self._thread.start()

    def stop(self):  self._stop.set()
    def is_running(self):
        return self._thread is not None and self._thread.is_alive()

    def _run(self, mode, source, model_path,
             show_mesh=True, output_dir=DATA_LOGS_DIR,
             window_sec=10, max_faces=4):
        _DL=_dm=_da=_ml=_db=_dh=None
        try:
            if   mode==self.MODE_DATA_LOGGER:
                from data_logger import MediFaceDataLogger as _DL
            elif mode==self.MODE_MESH:
                from mesh import draw_mesh as _dm, draw_au_points as _da, draw_legend as _ml
            elif mode==self.MODE_VIDEO_MONITOR:
                from yuz_tespit_video import draw_face_boxes as _db, draw_hud as _dh
        except ImportError as e:
            self.log_cb(f"[HATA] {e}"); self.stats_cb(0,0,0,False); return

        try:
            opts = mp.tasks.vision.FaceLandmarkerOptions(
                base_options=mp.tasks.BaseOptions(model_asset_path=model_path),
                running_mode=mp.tasks.vision.RunningMode.VIDEO,
                num_faces=max_faces,
                min_face_detection_confidence=0.5,
                min_face_presence_confidence=0.5,
                min_tracking_confidence=0.5)
        except Exception as e:
            self.log_cb(f"[HATA] Model: {e}"); self.stats_cb(0,0,0,False); return

        logger=None
        if mode==self.MODE_DATA_LOGGER and _DL:
            try:
                logger=_DL(output_dir=output_dir,window_sec=window_sec,
                           model_path=model_path,max_faces=max_faces)
                self.log_cb(f"CSV → {output_dir}")
            except Exception as e:
                self.log_cb(f"[HATA] Logger: {e}"); self.stats_cb(0,0,0,False); return

        cap=cv2.VideoCapture(source)
        if not cap.isOpened():
            self.log_cb(f"[HATA] Kaynak acilamadi: {source}")
            if logger: logger.close()
            self.stats_cb(0,0,0,False); return

        fps0=cap.get(cv2.CAP_PROP_FPS) or 30
        self.log_cb(f"Kamera acildi  |  kaynak={source}  |  ~{fps0:.0f} fps")
        fid=0; _lts=-1; _buf=[]; is_rec=(mode==self.MODE_DATA_LOGGER)

        with mp.tasks.vision.FaceLandmarker.create_from_options(opts) as lm:
            while not self._stop.is_set():
                ret,frame=cap.read()
                if not ret: self.log_cb("Kaynak bitti."); break
                h,w=frame.shape[:2]
                ts=int(time.time()*1000)
                if ts<=_lts: ts=_lts+1
                _lts=ts; fc=0

                if mode==self.MODE_DATA_LOGGER and logger:
                    fi,res=logger.process_frame(frame,ts,lm)
                    if fi is not None:
                        frame=logger.draw_overlay(frame,res,fi)
                        fc=len(res.face_landmarks) if res and res.face_landmarks else 0
                    else:
                        cv2.putText(frame,"Yuz bulunamadi",(20,40),
                                    cv2.FONT_HERSHEY_SIMPLEX,0.7,(80,80,200),2)
                elif mode==self.MODE_MESH:
                    rgb=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
                    mi=mp.Image(image_format=mp.ImageFormat.SRGB,data=rgb)
                    res=lm.detect_for_video(mi,ts)
                    if res.face_landmarks:
                        ls=res.face_landmarks[0]; _dm(frame,ls,w,h,show_mesh)
                        _da(frame,ls,w,h); fc=len(res.face_landmarks)
                    else:
                        cv2.putText(frame,"Yuz bulunamadi",(20,40),
                                    cv2.FONT_HERSHEY_SIMPLEX,0.65,(80,80,200),2)
                    _ml(frame,h)
                elif mode==self.MODE_VIDEO_MONITOR:
                    rgb=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
                    mi=mp.Image(image_format=mp.ImageFormat.SRGB,data=rgb)
                    res=lm.detect_for_video(mi,ts)
                    fc=_db(frame,res,w,h); _dh(frame,fc,fid,w)

                now=time.time()
                _buf=[t for t in _buf+[now] if now-t<1.0]
                self.stats_cb(len(_buf),fid,fc,is_rec)
                try:    self.frame_queue.put_nowait(frame.copy())
                except queue.Full: pass
                fid+=1

        cap.release()
        if logger: logger.close(); self.log_cb(f"CSV kaydedildi. Kare: {fid}")
        self.log_cb(f"Durduruldu. Kare: {fid}")
        self.stats_cb(0,fid,0,False); self.active_mode=None


# ─── Subprocess yardımcıları ──────────────────────────────────────────────────
def _env():
    e=os.environ.copy(); e["PYTHONUTF8"]="1"; e["PYTHONIOENCODING"]="utf-8"; return e

def _pipe(args,label,log_cb):
    cmd=[sys.executable,"-X","utf8"]+args; log_cb(f"▶ {label} baslatiliyor…")
    try:
        p=subprocess.Popen(cmd,cwd=BASE_DIR,env=_env(),
                           stdout=subprocess.PIPE,stderr=subprocess.STDOUT,
                           text=True,encoding="utf-8",errors="replace")
        for line in iter(p.stdout.readline,""): log_cb(line.rstrip())
        p.wait(); log_cb(f"■ {label} bitti (kod {p.returncode})")
    except Exception as e: log_cb(f"[HATA] {label}: {e}")

def _bg(fn,*a): threading.Thread(target=fn,args=a,daemon=True).start()


# ─────────────────────────────────────────────────────────────────────────────
# Ana Uygulama
# ─────────────────────────────────────────────────────────────────────────────
MODES = [
    # (label, icon_text, desc, accent_color, engine_mode)
    ("Data Logger",    "DL", "Kameradan AU koordinati kaydet",  GOLD,   CameraEngine.MODE_DATA_LOGGER),
    ("Yuz Mesh",       "YM", "468 nokta yuz mesh goruntusu",    NAVY3,  CameraEngine.MODE_MESH),
    ("Video Monitor",  "VM", "Canli yuz tespit izleme",          C_TEAL, CameraEngine.MODE_VIDEO_MONITOR),
    ("Goruntu Logger", "GL", "Fotograf klasoru AU loggeri",      C_GRN,  None),
    ("Yuz Filtre",     "YF", "Veri seti on-isleme araci",        C_RED,  None),
]


class App:
    def __init__(self, root):
        self.root = root
        self.root.title("MediFace AI — Control Panel")
        self.root.configure(bg=BG)
        self.root.minsize(1140, 680)
        self.root.geometry("1340x780")

        if not PIL_OK:
            messagebox.showerror("Eksik", "pip install Pillow")

        self._q      = queue.Queue(maxsize=2)
        self._engine = CameraEngine(self._q, self._log, self._stats_cb)
        self._photo  = None
        self._blink  = False
        self._cur    = 0
        self._nav_items = []

        self._build()
        self._log("MediFace AI hazir.")
        self._log("Gizlilik: kamera goruntuleri asla diske yazilmaz.")
        self._check_models()
        self._poll()

    # ── Build ─────────────────────────────────────────────────────────────────
    def _build(self):
        self._setup_ttk()
        self._build_topbar()
        self._build_body()

    def _setup_ttk(self):
        s = ttk.Style(); s.theme_use("clam")
        s.configure(".", background=BG, foreground=T1,
                    font=_fs(9), borderwidth=0, relief="flat")
        s.configure("Vertical.TScrollbar",
                    background=BORD, troughcolor=CARD,
                    arrowcolor=T3, borderwidth=0, width=4,
                    gripcount=0)
        s.map("Vertical.TScrollbar",
              background=[("active", BORD2)])

    # ── Top bar ───────────────────────────────────────────────────────────────
    def _build_topbar(self):
        # Outer wrapper — navy dark bar
        bar = tk.Frame(self.root, bg=NAVY, height=56)
        bar.pack(fill="x"); bar.pack_propagate(False)

        # Brand (left)
        left = tk.Frame(bar, bg=NAVY)
        left.place(relx=0, rely=0.5, anchor="w", x=24)

        # Brand mark — gold square
        mark = tk.Frame(left, bg=GOLD, width=28, height=28)
        mark.pack(side="left", padx=(0, 12))
        mark.pack_propagate(False)
        tk.Label(mark, text="M", bg=GOLD, fg=NAVY,
                 font=(FF, 13, "bold")).place(relx=0.5, rely=0.5, anchor="center")

        tk.Label(left, text="MediFace", bg=NAVY, fg=TG,
                 font=(FF, 15, "bold")).pack(side="left")
        tk.Label(left, text=" AI", bg=NAVY, fg=GOLD,
                 font=(FF, 15, "bold")).pack(side="left")
        tk.Label(left, text="  Control Panel", bg=NAVY, fg=ST3,
                 font=(FF, 9)).pack(side="left", pady=(3,0))

        # Right — status pill + clock
        right = tk.Frame(bar, bg=NAVY)
        right.place(relx=1.0, rely=0.5, anchor="e", x=-24)

        self._clock = tk.Label(right, text="", bg=NAVY, fg=ST2,
                               font=_fs(9))
        self._clock.pack(side="right", padx=(16, 0))

        # Vertical separator
        tk.Frame(right, bg=NAVY3, width=1, height=22).pack(
            side="right", padx=12)

        # Status pill
        self._st_pill = tk.Frame(right, bg=NAVY2,
                                 highlightbackground=NAVY3,
                                 highlightthickness=1)
        self._st_pill.pack(side="right", ipady=4, ipadx=6)
        self._st_dot = tk.Label(self._st_pill, text="●", bg=NAVY2,
                                fg=C_GRN, font=_fs(7))
        self._st_dot.pack(side="left", padx=(8, 4))
        self._st_lbl = tk.Label(self._st_pill, text="Hazir", bg=NAVY2,
                                fg=ST2, font=_fs(8))
        self._st_lbl.pack(side="left", padx=(0, 8))

        # Gold accent line
        tk.Frame(self.root, bg=GOLD, height=2).pack(fill="x")
        self._tick_clock()

    def _tick_clock(self):
        self._clock.configure(
            text=datetime.now().strftime("%d %b %Y  %H:%M"))
        self.root.after(30000, self._tick_clock)

    # ── Body ──────────────────────────────────────────────────────────────────
    def _build_body(self):
        body = tk.Frame(self.root, bg=BG)
        body.pack(fill="both", expand=True)

        # Sidebar — navy (220 px)
        self._side = tk.Frame(body, bg=NAVY, width=220)
        self._side.pack(side="left", fill="y")
        self._side.pack_propagate(False)

        # Right panel — white (290 px)
        self._rpanel = tk.Frame(body, bg=PANEL,
                                highlightbackground=BORD,
                                highlightthickness=1)
        self._rpanel.pack(side="right", fill="y")
        # Fix width without pack_propagate for proper sizing
        self._rpanel.config(width=295)
        self._rpanel.pack_propagate(False)

        # Center — camera
        self._mid = tk.Frame(body, bg=BG)
        self._mid.pack(fill="both", expand=True)

        self._build_sidebar()
        self._build_camera()
        self._build_right_panel()

        # Select first mode
        self._select(0)

    # ── Sidebar ───────────────────────────────────────────────────────────────
    def _build_sidebar(self):
        # ── Bottom zone: buttons + privacy note ──────────────────────────────
        btn_zone = tk.Frame(self._side, bg=NAVY)
        btn_zone.pack(side="bottom", fill="x", padx=14, pady=14)

        # Privacy notice
        priv_f = tk.Frame(btn_zone, bg=NAVY2,
                          highlightbackground=NAVY3, highlightthickness=1)
        priv_f.pack(fill="x", pady=(0, 10), ipady=4, ipadx=6)
        tk.Label(priv_f, text="🔒", bg=NAVY2, fg=GOLD,
                 font=_fs(9)).pack(side="left", padx=(8,4))
        tk.Label(priv_f, text="Goruntu diske yazilmaz",
                 bg=NAVY2, fg=ST3, font=_fs(7),
                 wraplength=150, justify="left").pack(side="left", padx=(0,6))

        # START button
        self._btn_start = tk.Button(
            btn_zone, text="▶  BASLAT",
            bg=GOLD, fg=NAVY,
            font=(FF, 10, "bold"),
            relief="flat", cursor="hand2",
            pady=10, bd=0,
            activebackground=GOLD2,
            activeforeground=NAVY,
            command=self._on_start)
        self._btn_start.pack(fill="x", pady=(0, 6))

        # STOP button
        self._btn_stop = tk.Button(
            btn_zone, text="■  DURDUR",
            bg=NAVY2, fg=ST3,
            font=(FF, 10, "bold"),
            relief="flat", cursor="hand2",
            pady=10, bd=0,
            activebackground="#2a2a4e",
            activeforeground=ST2,
            state="disabled", command=self._on_stop)
        self._btn_stop.pack(fill="x")

        # Divider
        tk.Frame(self._side, bg=NAVY3, height=1).pack(
            side="bottom", fill="x")

        # ── Nav header ────────────────────────────────────────────────────────
        tk.Frame(self._side, bg=NAVY, height=20).pack(fill="x")
        hdr = tk.Frame(self._side, bg=NAVY)
        hdr.pack(fill="x", padx=18, pady=(0, 4))
        tk.Label(hdr, text="MODLAR", bg=NAVY, fg=GOLD,
                 font=(FF, 7, "bold")).pack(side="left")

        # Thin separator under header
        tk.Frame(self._side, bg=NAVY3, height=1).pack(
            fill="x", padx=14, pady=(0, 4))

        # ── Nav items ─────────────────────────────────────────────────────────
        for i, (label, icon, desc, color, _) in enumerate(MODES):
            item = tk.Frame(self._side, bg=NAVY, cursor="hand2", height=58)
            item.pack(fill="x"); item.pack_propagate(False)

            # Left accent bar
            ind = tk.Frame(item, bg=NAVY, width=4)
            ind.pack(side="left", fill="y")

            # Badge icon
            badge_f = tk.Frame(item, bg=NAVY)
            badge_f.pack(side="left", padx=(10, 10))
            badge = tk.Label(badge_f, text=icon, bg=NAVY3,
                             fg=ST1, font=(FF, 7, "bold"),
                             width=3, height=1, padx=2, pady=1)
            badge.pack(pady=17)

            # Text column
            inner = tk.Frame(item, bg=NAVY)
            inner.pack(side="left", fill="both", expand=True, padx=(0, 12))

            name_lbl = tk.Label(inner, text=label, bg=NAVY,
                                fg=ST2, font=(FF, 9, "bold"), anchor="w")
            name_lbl.pack(anchor="w", pady=(14, 0))

            desc_lbl = tk.Label(inner, text=desc, bg=NAVY,
                                fg=ST3, font=(FF, 7), anchor="w")
            desc_lbl.pack(anchor="w")

            all_w = [item, ind, badge_f, badge, inner, name_lbl, desc_lbl]

            def _cb(e=None, idx=i): self._select(idx)
            for w in all_w: w.bind("<Button-1>", _cb)

            def _he(e, ws=all_w):
                for w in ws:
                    try: w.configure(bg=NAVY2)
                    except Exception: pass
            def _hl(e, ws=all_w, idx=i):
                bg_ = NAVY2 if self._cur == idx else NAVY
                for w in ws:
                    try: w.configure(bg=bg_)
                    except Exception: pass
            for w in all_w:
                w.bind("<Enter>", _he)
                w.bind("<Leave>", _hl)

            self._nav_items.append({
                "all": all_w, "ind": ind, "name": name_lbl,
                "badge": badge, "desc": desc_lbl,
                "color": color, "item": item,
            })

    # ── Camera area ───────────────────────────────────────────────────────────
    def _build_camera(self):
        # Top info bar
        top = tk.Frame(self._mid, bg=PANEL,
                       highlightbackground=BORD, highlightthickness=1)
        top.pack(fill="x"); top.pack_propagate(False)
        top.configure(height=46)

        # Breadcrumb
        bc = tk.Frame(top, bg=PANEL)
        bc.place(relx=0, rely=0.5, anchor="w", x=18)
        tk.Label(bc, text="CANLI YAYIN", bg=PANEL, fg=T3,
                 font=(FF, 7, "bold")).pack(side="left")
        tk.Label(bc, text="  /", bg=PANEL, fg=BORD2,
                 font=_fs(8)).pack(side="left")
        self._badge = tk.Label(bc, text=" —", bg=PANEL, fg=GOLD,
                               font=(FF, 8, "bold"))
        self._badge.pack(side="left")

        # REC indicator
        self._rec_lbl = tk.Label(top, text="", bg=PANEL,
                                 fg=C_RED, font=(FF, 8, "bold"))
        self._rec_lbl.place(relx=1.0, rely=0.5, anchor="e", x=-18)

        # Camera display
        cam_border = tk.Frame(self._mid, bg=BORD)
        cam_border.pack(fill="both", expand=True, padx=1)

        self._cam_lbl = tk.Label(cam_border, bg="#1e1e1e", anchor="center")
        self._cam_lbl.pack(fill="both", expand=True, padx=1, pady=1)

        # Placeholder
        self._ph_frame = tk.Frame(cam_border, bg="#1e1e1e")
        self._ph_frame.place(relx=0.5, rely=0.5, anchor="center")

        # Camera icon placeholder
        cam_icon = tk.Frame(self._ph_frame, bg="#333333",
                            width=80, height=80)
        cam_icon.pack(); cam_icon.pack_propagate(False)
        tk.Label(cam_icon, text="📷", bg="#333333",
                 font=_fs(32)).place(relx=0.5, rely=0.5, anchor="center")

        self._ph_lbl = tk.Label(self._ph_frame,
            text="Soldan bir mod seçin, ardından  ▶ BASLAT  düğmesine basın",
            bg="#1e1e1e", fg=T3, font=(FF, 10))
        self._ph_lbl.pack(pady=(14, 0))

        # Bottom stat bar
        stat_bar = tk.Frame(self._mid, bg=PANEL,
                            highlightbackground=BORD, highlightthickness=1)
        stat_bar.pack(fill="x", side="bottom")
        stat_bar.configure(height=42)
        stat_bar.pack_propagate(False)

        for lbl, attr, clr in [("FPS", "_ov_fps", GOLD),
                                ("YÜZ", "_ov_face", C_GRN),
                                ("KARE", "_ov_kare", C_BLU)]:
            seg = tk.Frame(stat_bar, bg=PANEL)
            seg.pack(side="left", padx=20)
            tk.Label(seg, text=lbl, bg=PANEL, fg=T3,
                     font=(FF, 7, "bold")).pack(side="left", padx=(0, 6))
            v = tk.Label(seg, text="—", bg=PANEL, fg=clr,
                         font=(MONO, 11, "bold"))
            v.pack(side="left")
            setattr(self, attr, v)

        # Right — mode badge in stat bar
        self._stat_mode = tk.Label(stat_bar, text="", bg=PANEL,
                                   fg=T3, font=(FF, 7))
        self._stat_mode.pack(side="right", padx=18)

    # ── Right panel ───────────────────────────────────────────────────────────
    def _build_right_panel(self):
        p = self._rpanel

        # Section: Settings ───────────────────────────────────────────────────
        sett_hdr = tk.Frame(p, bg=PANEL)
        sett_hdr.pack(fill="x", padx=0, pady=0)
        sett_hdr.configure(height=44)
        sett_hdr.pack_propagate(False)

        # Gold left border accent for header
        tk.Frame(sett_hdr, bg=GOLD, width=3).pack(side="left", fill="y")
        hdr_inner = tk.Frame(sett_hdr, bg=PANEL)
        hdr_inner.pack(side="left", fill="both", expand=True, padx=14)
        self._sett_title = tk.Label(hdr_inner, text="AYARLAR",
                                    bg=PANEL, fg=GOLD,
                                    font=(FF, 8, "bold"), anchor="w")
        self._sett_title.pack(fill="x", pady=13)

        tk.Frame(p, bg=BORD, height=1).pack(fill="x")

        # Scrollable settings card
        sett_card = tk.Frame(p, bg=PANEL)
        sett_card.pack(fill="x")

        self._sc = tk.Canvas(sett_card, bg=PANEL, highlightthickness=0, height=215)
        vsb = ttk.Scrollbar(sett_card, orient="vertical", command=self._sc.yview)
        self._sc.configure(yscrollcommand=vsb.set)
        vsb.pack(side="right", fill="y")
        self._sc.pack(fill="x", expand=False)

        self._si = tk.Frame(self._sc, bg=PANEL)
        _wid = self._sc.create_window((0, 0), window=self._si, anchor="nw")
        self._sc.bind("<Configure>",
                      lambda e: self._sc.itemconfig(_wid, width=e.width))
        self._si.bind("<Configure>",
                      lambda e: self._sc.configure(
                          scrollregion=self._sc.bbox("all")))
        self._sc.bind_all("<MouseWheel>",
                          lambda e: self._sc.yview_scroll(
                              int(-1 * (e.delta / 120)), "units"))

        # Section: Live Stats ─────────────────────────────────────────────────
        tk.Frame(p, bg=BORD, height=1).pack(fill="x")

        stats_hdr = tk.Frame(p, bg=PANEL, height=44)
        stats_hdr.pack(fill="x"); stats_hdr.pack_propagate(False)
        tk.Frame(stats_hdr, bg=C_TEAL, width=3).pack(side="left", fill="y")
        tk.Label(stats_hdr, text="CANLI İSTATİSTİK", bg=PANEL,
                 fg=T2, font=(FF, 8, "bold"),
                 padx=14).pack(side="left", pady=13)

        # Stat cards grid
        grid = tk.Frame(p, bg=PANEL)
        grid.pack(fill="x", padx=12, pady=(0, 8))

        self._sv = {}
        stat_defs = [
            ("fps",   "FPS",  GOLD),
            ("faces", "YÜZ",  C_GRN),
            ("frame", "KARE", C_BLU),
        ]
        for col, (key, label, clr) in enumerate(stat_defs):
            card = tk.Frame(grid, bg=CARDUP,
                            highlightbackground=BORD,
                            highlightthickness=1)
            card.grid(row=0, column=col,
                      padx=(0, 6) if col < 2 else 0,
                      sticky="nsew")
            grid.columnconfigure(col, weight=1)

            # Top color bar
            tk.Frame(card, bg=clr, height=3).pack(fill="x")
            v = tk.Label(card, text="—", bg=CARDUP, fg=clr,
                         font=(MONO, 18, "bold"))
            v.pack(padx=10, pady=(8, 2))
            tk.Label(card, text=label, bg=CARDUP, fg=T3,
                     font=(FF, 7)).pack(padx=10, pady=(0, 8))
            self._sv[key] = v

        # Section: Console ────────────────────────────────────────────────────
        tk.Frame(p, bg=BORD, height=1).pack(fill="x")

        cons_hdr = tk.Frame(p, bg=PANEL, height=40)
        cons_hdr.pack(fill="x"); cons_hdr.pack_propagate(False)
        tk.Frame(cons_hdr, bg=NAVY3, width=3).pack(side="left", fill="y")
        tk.Label(cons_hdr, text="KONSOL", bg=PANEL, fg=T2,
                 font=(FF, 8, "bold"), padx=14).pack(side="left", pady=12)
        clr_btn = tk.Button(cons_hdr, text="Temizle",
                            bg=PANEL, fg=T3,
                            relief="flat", font=(FF, 7),
                            cursor="hand2", bd=0,
                            activebackground=CARDUP,
                            activeforeground=T2,
                            command=self._clear_log)
        clr_btn.pack(side="right", padx=14, pady=10)
        clr_btn.configure(padx=6, pady=2)

        tk.Frame(p, bg=BORD, height=1).pack(fill="x")

        # Console text area
        log_card = tk.Frame(p, bg="#222222")
        log_card.pack(fill="both", expand=True, padx=0, pady=0)

        vsb2 = ttk.Scrollbar(log_card, orient="vertical")
        vsb2.pack(side="right", fill="y")

        self._log_txt = tk.Text(
            log_card, bg="#222222", fg=T2,
            font=(MONO, 8), relief="flat", bd=0,
            wrap="word", state="disabled",
            yscrollcommand=vsb2.set, cursor="arrow",
            padx=12, pady=10,
            selectbackground="#44475a",
            insertbackground=GOLD)
        self._log_txt.pack(fill="both", expand=True)
        vsb2.config(command=self._log_txt.yview)

        for tag, fg in [("ts", T4), ("info", GOLD),
                        ("ok", C_GRN), ("warn", C_AMB),
                        ("err", C_RED), ("mut", T3)]:
            self._log_txt.tag_configure(tag, foreground=fg)

        # Model status footer
        tk.Frame(p, bg=BORD, height=1).pack(fill="x")
        self._model_lbl = tk.Label(p, text="", bg=PANEL,
                                   fg=T3, font=(FF, 7))
        self._model_lbl.pack(side="bottom", padx=14, pady=(5, 6), anchor="w")

    # ── Widget helpers ────────────────────────────────────────────────────────
    def _section(self, p, text):
        f = tk.Frame(p, bg=PANEL)
        f.pack(fill="x", padx=0, pady=(10, 2))
        # Thin gold line
        tk.Frame(f, bg=BORD, height=1).pack(fill="x")
        lbl_f = tk.Frame(f, bg=PANEL)
        lbl_f.pack(fill="x", padx=14, pady=(4, 0))
        tk.Label(lbl_f, text=text, bg=PANEL, fg=T3,
                 font=(FF, 7, "bold"), anchor="w").pack(fill="x")

    def _entry(self, p, label, default=""):
        outer = tk.Frame(p, bg=PANEL)
        outer.pack(fill="x", padx=14, pady=(2, 0))
        tk.Label(outer, text=label, bg=PANEL, fg=T2,
                 font=(FF, 7), anchor="w").pack(fill="x")
        var = tk.StringVar(value=default)
        entry_f = tk.Frame(outer,
                           highlightbackground=BORD,
                           highlightthickness=1,
                           bg=FIELD)
        entry_f.pack(fill="x", pady=(2, 4))
        e = tk.Entry(entry_f, textvariable=var,
                     bg=FIELD, fg=T1,
                     insertbackground=GOLD,
                     relief="flat", font=(FF, 8),
                     bd=6)
        e.pack(fill="x")
        e.bind("<FocusIn>",
               lambda ev, f=entry_f: f.configure(highlightbackground=GOLD))
        e.bind("<FocusOut>",
               lambda ev, f=entry_f: f.configure(highlightbackground=BORD))
        return var

    def _chk(self, p, label, default=False):
        var = tk.BooleanVar(value=default)
        f = tk.Frame(p, bg=PANEL)
        f.pack(fill="x", padx=14, pady=2)
        tk.Checkbutton(f, text=label, variable=var,
                       bg=PANEL, fg=T2,
                       selectcolor=FIELD,
                       activebackground=PANEL,
                       activeforeground=T1,
                       font=(FF, 8),
                       cursor="hand2").pack(anchor="w")
        return var

    def _pick_btn(self, p, label, cmd):
        f = tk.Frame(p, bg=PANEL)
        f.pack(fill="x", padx=14, pady=(0, 4))
        b = tk.Button(f, text=label,
                      bg=CARDUP, fg=T2,
                      relief="flat", font=(FF, 7),
                      cursor="hand2",
                      activebackground=BORD,
                      activeforeground=T1,
                      padx=10, pady=4, bd=0,
                      highlightbackground=BORD,
                      highlightthickness=1,
                      command=cmd)
        b.pack(anchor="w")

    def _info_box(self, p, text):
        f = tk.Frame(p, bg=CARDUP,
                     highlightbackground=BORD,
                     highlightthickness=1)
        f.pack(fill="x", padx=14, pady=(4, 10))
        tk.Frame(f, bg=GOLD, height=2).pack(fill="x")
        tk.Label(f, text=text, bg=CARDUP, fg=T3,
                 font=(FF, 7), justify="left",
                 padx=8, pady=6, wraplength=220).pack(fill="x")

    # ── Settings panels ───────────────────────────────────────────────────────
    def _build_sett(self, idx):
        for w in self._si.winfo_children(): w.destroy()
        [self._s0, self._s1, self._s2, self._s3, self._s4][idx](self._si)

    def _s0(self, p):
        self._section(p, "KAYNAK")
        self._dl_src = self._entry(p, "Kamera indeksi / video", "0")
        self._pick_btn(p, "📁  Video seç",
                       lambda: self._pf(self._dl_src, [("Video", "*.mp4 *.avi *.mov"), ("Tümü", "*.*")]))
        self._section(p, "MODEL")
        self._dl_mdl = self._entry(p, "face_landmarker.task", DEFAULT_MODEL)
        self._pick_btn(p, "📁  Model seç",
                       lambda: self._pf(self._dl_mdl, [("Task", "*.task"), ("Tümü", "*.*")]))
        self._section(p, "ÇIKTI")
        self._dl_out = self._entry(p, "CSV klasörü", DATA_LOGS_DIR)
        self._dl_win = self._entry(p, "Window (sn)", "10")

    def _s1(self, p):
        self._section(p, "KAYNAK")
        self._ms_src = self._entry(p, "Kamera indeksi / video", "0")
        self._pick_btn(p, "📁  Video seç",
                       lambda: self._pf(self._ms_src, [("Video", "*.mp4 *.avi *.mov"), ("Tümü", "*.*")]))
        self._section(p, "MODEL")
        self._ms_mdl = self._entry(p, "face_landmarker.task", DEFAULT_MODEL)
        self._section(p, "GÖRÜNÜM")
        self._ms_show = self._chk(p, "Mesh bağlantılarını göster", True)

    def _s2(self, p):
        self._section(p, "KAYNAK")
        self._vm_src = self._entry(p, "Kamera indeksi / video", "0")
        self._pick_btn(p, "📁  Video seç",
                       lambda: self._pf(self._vm_src, [("Video", "*.mp4 *.avi *.mov"), ("Tümü", "*.*")]))
        self._section(p, "MODEL")
        self._vm_mdl = self._entry(p, "face_landmarker.task", DEFAULT_MODEL)
        self._section(p, "ALGILAMA")
        self._vm_max = self._entry(p, "Max yüz", "4")

    def _s3(self, p):
        self._section(p, "KAYNAK (kamera değil)")
        self._il_fol = self._entry(p, "Klasör (boş = tek dosya)", "")
        self._pick_btn(p, "📂  Klasör seç", lambda: self._pd(self._il_fol))
        self._section(p, "MODEL")
        self._il_mdl = self._entry(p, "face_landmarker.task", DEFAULT_MODEL)
        self._section(p, "ÇIKTI")
        self._il_out = self._entry(p, "CSV klasörü", DATA_LOGS_DIR)
        self._il_rec = self._chk(p, "Recursive")
        self._il_pre = self._chk(p, "Önizleme")
        self._info_box(p, "Sadece sayısal AU koordinatı (CSV)\nkaydedilir. Görüntü diske yazılmaz.")

    def _s4(self, p):
        self._section(p, "KAYNAK (kamera değil)")
        self._yf_src = self._entry(p, "Kaynak klasör", "dermnet")
        self._pick_btn(p, "📂  Klasör seç", lambda: self._pd(self._yf_src))
        self._section(p, "MODEL (BlazeFace)")
        self._yf_mdl = self._entry(p, "blaze_face_short_range.tflite", BLAZE_MODEL)
        self._section(p, "ÇIKTI / PARAMETRE")
        self._yf_out = self._entry(p, "Çıktı kökü", "datasets/veri")
        self._yf_con = self._entry(p, "Güven eşiği", "0.7")
        self._yf_sz  = self._entry(p, "Min bbox (px)", "50")
        self._info_box(p, "Veri seti ön-işleme aracı.\nKamera kullanmaz.")

    # ── Mode selection ────────────────────────────────────────────────────────
    def _select(self, idx):
        self._cur = idx
        label, icon, desc, color, _ = MODES[idx]

        # Reset all nav items
        for info in self._nav_items:
            for w in info["all"]:
                try: w.configure(bg=NAVY)
                except Exception: pass
            info["ind"].configure(bg=NAVY)
            info["name"].configure(fg=ST2)
            info["badge"].configure(bg=NAVY3, fg=ST1)

        # Highlight selected
        p = self._nav_items[idx]
        for w in p["all"]:
            try: w.configure(bg=NAVY2)
            except Exception: pass
        p["ind"].configure(bg=color, width=4)
        p["name"].configure(fg=TG)
        p["badge"].configure(bg=color, fg=NAVY)

        # Update start button color
        self._btn_start.configure(bg=color, fg=NAVY,
                                  activebackground=_lighten(color, 20),
                                  activeforeground=NAVY)

        # Badge + settings
        self._badge.configure(text=f"  {label.upper()}", fg=color)
        self._sett_title.configure(text=f"{label.upper()} — AYARLARI", fg=color)
        self._stat_mode.configure(text=label.upper(), fg=color)
        self._build_sett(idx)

    # ── Start / Stop ──────────────────────────────────────────────────────────
    def _on_start(self):
        idx = self._cur
        label, icon, desc, color, eng_mode = MODES[idx]

        if eng_mode is None:
            if idx == 3: self._start_img()
            else:        self._start_filtre()
            return

        if self._engine.is_running():
            self._log("[!] Zaten calisiyor."); return

        if   idx == 0:
            src = self._dl_src.get().strip() or "0"
            mdl = self._dl_mdl.get().strip()
            kw  = dict(output_dir=self._dl_out.get().strip(),
                       window_sec=int(self._dl_win.get().strip() or "10"))
        elif idx == 1:
            src = self._ms_src.get().strip() or "0"
            mdl = self._ms_mdl.get().strip()
            kw  = dict(show_mesh=self._ms_show.get())
        elif idx == 2:
            src = self._vm_src.get().strip() or "0"
            mdl = self._vm_mdl.get().strip()
            kw  = dict(max_faces=int(self._vm_max.get().strip() or "4"))
        else: return

        source = int(src) if str(src).isdigit() else src
        self._engine.start(eng_mode, source, mdl, **kw)
        self._set_status(f"{label} çalışıyor", color)
        self._btn_start.configure(state="disabled", bg=BORD, fg=T3)
        self._btn_stop.configure(state="normal", bg=C_RED, fg=TW)
        self._ph_frame.place_forget()

    def _on_stop(self):
        self._engine.stop()
        self._set_status("Durduruluyor…", C_AMB)
        color = MODES[self._cur][3]
        self._btn_start.configure(state="normal", bg=color,
                                  fg=NAVY,
                                  activebackground=_lighten(color, 20))
        self._btn_stop.configure(state="disabled", bg=NAVY2, fg=ST3)

    # ── Frame poll ────────────────────────────────────────────────────────────
    def _poll(self):
        if PIL_OK:
            try:
                frame = self._q.get_nowait(); self._show(frame)
            except queue.Empty: pass

        if not self._engine.is_running() and self._btn_stop["state"] == "normal":
            self._ph_frame.place(relx=0.5, rely=0.5, anchor="center")
            color = MODES[self._cur][3]
            self._btn_start.configure(state="normal", bg=color,
                                      fg=NAVY,
                                      activebackground=_lighten(color, 20))
            self._btn_stop.configure(state="disabled", bg=NAVY2, fg=ST3)
            self._set_status("Hazır", C_GRN)

        self.root.after(30, self._poll)

    def _show(self, frame):
        if not PIL_OK: return
        try:
            lw = self._cam_lbl.winfo_width()
            lh = self._cam_lbl.winfo_height()
            fh, fw = frame.shape[:2]
            if lw < 4 or lh < 4: lw, lh = fw, fh
            s = min(lw / fw, lh / fh)
            nw, nh = max(1, int(fw * s)), max(1, int(fh * s))
            out = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_LINEAR)
            rgb = cv2.cvtColor(out, cv2.COLOR_BGR2RGB)
            photo = ImageTk.PhotoImage(image=Image.fromarray(rgb))
            self._cam_lbl.configure(image=photo, bg="#1e1e1e")
            self._photo = photo
        except Exception: pass

    # ── Stats ─────────────────────────────────────────────────────────────────
    def _stats_cb(self, fps, fid, faces, is_rec):
        self.root.after(0, self._upd_stats, fps, fid, faces, is_rec)

    def _upd_stats(self, fps, fid, faces, is_rec):
        self._sv["fps"].configure(text=str(fps))
        self._sv["frame"].configure(text=str(fid))
        self._sv["faces"].configure(text=str(faces),
                                    fg=C_GRN if faces > 0 else C_RED)
        self._ov_fps.configure(text=str(fps))
        self._ov_face.configure(text=str(faces),
                                fg=C_GRN if faces > 0 else T3)
        self._ov_kare.configure(text=str(fid))
        if is_rec:
            self._blink = not self._blink
            self._rec_lbl.configure(
                text="⏺  REC" if self._blink else "   REC",
                fg=C_RED if self._blink else "#cc000022")
        else:
            self._rec_lbl.configure(text="")

    # ── Log ──────────────────────────────────────────────────────────────────
    def _log(self, msg):
        self.root.after(0, self._log_ui, msg)

    def _log_ui(self, msg):
        ts  = datetime.now().strftime("%H:%M:%S")
        ml  = msg.lower()
        tag = ("err"  if any(w in ml for w in ["hata", "error", "traceback"]) else
               "ok"   if any(w in ml for w in ["■", "tamamlandi", "bitti", "sonlandi"]) else
               "info" if any(w in ml for w in ["▶", "basladi", "kamera acildi", "csv"]) else
               "warn" if any(w in ml for w in ["uyari", "warning", "!"]) else
               "mut")
        self._log_txt.configure(state="normal")
        self._log_txt.insert("end", f"[{ts}] ", "ts")
        self._log_txt.insert("end", msg + "\n", tag)
        self._log_txt.configure(state="disabled")
        self._log_txt.see("end")

    def _clear_log(self):
        self._log_txt.configure(state="normal")
        self._log_txt.delete("1.0", "end")
        self._log_txt.configure(state="disabled")

    # ── Subprocess ────────────────────────────────────────────────────────────
    def _start_img(self):
        folder = self._il_fol.get().strip()
        mdl    = self._il_mdl.get().strip()
        out    = self._il_out.get().strip()
        args   = [os.path.join(BASE_DIR, "yuz_tespiti_image.py"),
                  "--model", mdl, "--output", out]
        if folder: args += ["--folder", folder]
        if self._il_rec.get(): args.append("--recursive")
        if self._il_pre.get(): args.append("--preview")
        _bg(_pipe, args, "Goruntu Logger", self._log)

    def _start_filtre(self):
        src  = self._yf_src.get().strip() or "dermnet"
        out  = self._yf_out.get().strip() or "datasets/veri"
        mdl  = self._yf_mdl.get().strip()
        conf = self._yf_con.get().strip() or "0.7"
        sz   = self._yf_sz.get().strip() or "50"
        if not os.path.exists(mdl):
            messagebox.showwarning("Model Eksik",
                                   f"BlazeFace bulunamadı:\n{mdl}"); return
        args = [os.path.join(BASE_DIR, "yuz_filtre.py"),
                "--source", src, "--output", out,
                "--model", mdl, "--conf", conf, "--min-size", sz]
        _bg(_pipe, args, "Yuz Filtre", self._log)

    # ── Helpers ───────────────────────────────────────────────────────────────
    def _pf(self, var, ft):
        p = filedialog.askopenfilename(filetypes=ft)
        if p: var.set(p)

    def _pd(self, var):
        p = filedialog.askdirectory()
        if p: var.set(p)

    def _set_status(self, text, color=None):
        c = color or C_GRN
        self._st_dot.configure(fg=c)
        self._st_lbl.configure(text=text, fg=ST2)

    def _check_models(self):
        miss = [m for m, p in [("face_landmarker.task",  DEFAULT_MODEL),
                                ("blaze_face_short_range.tflite", BLAZE_MODEL)]
                if not os.path.exists(p)]
        if miss:
            self._model_lbl.configure(
                text=f"⚠  {', '.join(miss)}", fg=C_AMB)
            self._log(f"[!] Eksik model: {', '.join(miss)}")
        else:
            self._model_lbl.configure(text="✓  Tüm modeller hazır", fg=C_GRN)
            self._log("✓ Tum modeller yerinde.")


# ── Utility ───────────────────────────────────────────────────────────────────
def _lighten(hex_c, amt=20):
    try:
        h = hex_c.lstrip("#")
        if len(h) != 6: return hex_c
        r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
        return "#{:02x}{:02x}{:02x}".format(
            min(255, r + amt), min(255, g + amt), min(255, b + amt))
    except Exception: return hex_c


if __name__ == "__main__":
    root = tk.Tk()
    App(root)
    root.mainloop()
