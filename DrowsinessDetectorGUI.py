#!/usr/bin/env python
"""
GUI for Driver Drowsiness Detection
Allows user to select camera or upload video file for analysis
"""
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import cv2
import threading
import os
import sys

# Import detection modules
from scipy.spatial import distance as dist
from imutils.video import VideoStream
from imutils import face_utils
import imutils
import time
import dlib
import math
import numpy as np
from PIL import Image, ImageTk
from EAR import eye_aspect_ratio
from MAR import mouth_aspect_ratio
from HeadPose import getHeadTiltAndCoords


# Language translations
LANGUAGES = {
    'en': {
        'title': 'Driver Drowsiness Detection',
        'controls': 'Controls',
        'video_source': 'Video Source:',
        'camera': 'Camera:',
        'no_cameras': 'No cameras detected',
        'video_file': 'Video:',
        'select': 'Select',
        'browse': 'Browse',
        'no_source': 'No source selected',
        'start': 'START',
        'stop': 'STOP',
        'ready': 'Ready',
        'running': 'Detection running...',
        'stopped': 'Detection stopped',
        'video_feed': 'Video Feed',
        'no_video': 'No video',
        'instructions': 'Instructions:\n1. Select Camera or Video\n2. Click START to begin\n3. Click STOP to end',
        # Detection messages
        'face_found': 'face(s) found',
        'eyes_closed': 'EYES CLOSED!',
        'yawning': 'YAWNING!',
        'head_tilt': 'Head Tilt:',
        'mar': 'MAR:',
        # Video controls
        'video_controls': 'Video Controls',
        'play': 'Play',
        'pause': 'Pause',
        'forward': '+10s',
        'backward': '-10s',
        'export': 'Export MP4',
        # Export
        'export_title': 'Export Video',
        'export_success': 'Video exported successfully!',
        'export_error': 'Error exporting video',
        # Menu
        'language': 'Language',
        'english': 'English',
        'spanish': 'Spanish',
    },
    'es': {
        'title': 'Detección de Fatiga y Somnolencia',
        'controls': 'Controles',
        'video_source': 'Fuente de Video:',
        'camera': 'Cámara:',
        'no_cameras': 'No se detectaron cámaras',
        'video_file': 'Video:',
        'select': 'Seleccionar',
        'browse': 'Explorar',
        'no_source': 'Sin fuente seleccionada',
        'start': 'INICIAR',
        'stop': 'DETENER',
        'ready': 'Listo',
        'running': 'Detección en progreso...',
        'stopped': 'Detección detenida',
        'video_feed': 'Video en Vivo',
        'no_video': 'Sin video',
        'instructions': 'Instrucciones:\n1. Seleccione Cámara o Video\n2. Haga clic en INICIAR\n3. Haga clic en DETENER',
        # Detection messages
        'face_found': 'cara(s) encontrada(s)',
        'eyes_closed': 'OJOS CERRADOS',
        'yawning': 'BOSTEZO',
        'head_tilt': 'Inclinacion:',
        'mar': 'MAR:',
        # Video controls
        'video_controls': 'Controles de Video',
        'play': 'Reproducir',
        'pause': 'Pausar',
        'forward': '+10s',
        'backward': '-10s',
        'export': 'Exportar MP4',
        # Export
        'export_title': 'Exportar Video',
        'export_success': '¡Video exportado exitosamente!',
        'export_error': 'Error al exportar video',
        # Menu
        'language': 'Idioma',
        'english': 'Inglés',
        'spanish': 'Español',
    }
}


class DrowsinessDetectorGUI:
    def __init__(self, root):
        self.root = root
        self.root.title(LANGUAGES['es']['title'])
        self.root.geometry("1200x800")
        self.root.resizable(False, False)
        
        # Language - Default to Spanish
        self.current_lang = 'es'
        
        # Detection variables
        self.detection_thread = None
        self.is_running = False
        self.is_paused = False
        self.video_source = None  # Can be camera index (int) or video file path (str)
        self.source_type = None   # 'camera' or 'video'
        self.current_frame = None  # Current frame for display
        self.vs = None  # VideoCapture or VideoStream object
        
        # Video playback
        self.video_total_frames = 0
        self.video_fps = 0
        self.current_frame_pos = 0
        self.is_video_playing = False
        
        # Video export
        self.export_writer = None
        self.is_exporting = False
        
        # Available cameras (check on startup)
        self.available_cameras = self.detect_cameras()
        
        # Create GUI
        self.create_widgets()
        
        # Load dlib models (one time)
        self.load_models()
    
    def t(self, key):
        """Get translated string"""
        return LANGUAGES[self.current_lang][key]
    
    def detect_cameras(self):
        """Detect available cameras on the system"""
        cameras = []
        for i in range(5):  # Check first 5 indices
            try:
                cap = cv2.VideoCapture(i)
                if cap.isOpened():
                    cameras.append(i)
                    cap.release()
            except:
                pass
        return cameras
    
    def load_models(self):
        """Load dlib face detector and landmark predictor"""
        print("[INFO] Loading facial landmark predictor...")
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(
            './dlib_shape_predictor/shape_predictor_68_face_landmarks.dat')
        
        # 2D image points
        self.image_points = np.array([
            (359, 391),     # Nose tip 34
            (399, 561),     # Chin 9
            (337, 297),     # Left eye left corner 37
            (513, 301),     # Right eye right corner 46
            (345, 465),     # Left Mouth corner 49
            (453, 469)      # Right mouth corner 55
        ], dtype="double")
        
        (self.lStart, self.lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
        (self.rStart, self.rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
        
        # Thresholds
        self.EYE_AR_THRESH = 0.25
        self.MOUTH_AR_THRESH = 0.79
        self.EYE_AR_CONSEC_FRAMES = 3
        self.COUNTER = 0
        
        (self.mStart, self.mEnd) = (49, 68)
        
        print("[INFO] Models loaded successfully")
    
    def create_widgets(self):
        """Create GUI widgets"""
        # Store widget references for language update
        self.widgets = {}
        
        # Menu bar
        self.create_menu()
        
        # Title with logo
        title_frame = tk.Frame(self.root)
        title_frame.pack(pady=10)
        
        # Try to load logo
        try:
            logo_path = './img/isologo color.png'
            if os.path.exists(logo_path):
                logo_img = Image.open(logo_path)
                # Resize logo to fit (keep aspect ratio)
                logo_img.thumbnail((150, 60), Image.LANCZOS)
                self.logo_photo = ImageTk.PhotoImage(logo_img)
                logo_label = tk.Label(title_frame, image=self.logo_photo)
                logo_label.image = self.logo_photo
                logo_label.pack(side="left", padx=10)
        except Exception as e:
            print(f"[WARNING] Could not load logo: {e}")
        
        self.title_label = tk.Label(
            title_frame, 
            text=self.t('title'),
            font=("Arial", 20, "bold")
        )
        self.title_label.pack(side="left", padx=10)
        
        # Main container
        main_container = tk.Frame(self.root)
        main_container.pack(fill="both", expand=True, padx=10, pady=5)
        
        # Left panel - Controls
        self.left_panel = tk.LabelFrame(main_container, text=self.t('controls'), font=("Arial", 12))
        self.left_panel.pack(side="left", fill="both", padx=5, pady=5)
        
        # Source selection
        self.source_label = tk.Label(self.left_panel, text=self.t('video_source'), font=("Arial", 11, "bold"))
        self.source_label.pack(pady=(10, 5))
        
        # Camera selection
        camera_frame = tk.Frame(self.left_panel)
        camera_frame.pack(pady=5)
        
        tk.Label(camera_frame, text=self.t('camera')).pack(side="left", padx=5)
        
        if self.available_cameras:
            self.camera_var = tk.StringVar()
            self.camera_combo = ttk.Combobox(
                camera_frame, 
                textvariable=self.camera_var,
                values=[f"Camera {i}" for i in self.available_cameras],
                state="readonly",
                width=15
            )
            self.camera_combo.current(0)
            self.camera_combo.pack(side="left", padx=5)
            
            self.use_camera_btn = tk.Button(
                camera_frame,
                text=self.t('select'),
                command=self.use_camera,
                bg="#4CAF50",
                fg="white",
                width=10
            )
            self.use_camera_btn.pack(side="left", padx=5)
        else:
            tk.Label(camera_frame, text=self.t('no_cameras'), fg="red").pack(side="left", padx=5)
        
        # OR label
        or_label = tk.Label(self.left_panel, text="- OR -")
        or_label.pack(pady=5)
        
        # Video upload
        video_frame = tk.Frame(self.left_panel)
        video_frame.pack(pady=5)
        
        tk.Label(video_frame, text=self.t('video_file')).pack(side="left", padx=5)
        
        self.video_path_var = tk.StringVar()
        self.video_entry = tk.Entry(video_frame, textvariable=self.video_path_var, width=18)
        self.video_entry.pack(side="left", padx=5)
        
        self.browse_btn = tk.Button(
            video_frame,
            text=self.t('browse'),
            command=self.browse_video,
            width=8
        )
        self.browse_btn.pack(side="left", padx=2)
        
        self.use_video_btn = tk.Button(
            video_frame,
            text=self.t('select'),
            command=self.use_video,
            bg="#2196F3",
            fg="white",
            width=8
        )
        self.use_video_btn.pack(side="left", padx=2)
        
        # Selected source display
        self.selected_source_var = tk.StringVar(value=self.t('no_source'))
        selected_label = tk.Label(
            self.left_panel, 
            textvariable=self.selected_source_var,
            fg="blue",
            font=("Arial", 10)
        )
        selected_label.pack(pady=10)
        
        # Video controls (for video playback)
        self.video_controls_frame = tk.LabelFrame(self.left_panel, text=self.t('video_controls'), font=("Arial", 10))
        self.video_controls_frame.pack(pady=10, padx=10, fill="x")
        self.video_controls_frame.pack_forget()  # Hide initially
        
        # Progress bar
        self.progress_var = tk.DoubleVar()
        self.progress_scale = tk.Scale(
            self.video_controls_frame, 
            from_=0, to=100, 
            orient="horizontal",
            variable=self.progress_var,
            showvalue=False,
            length=200,
            command=self.on_seek
        )
        self.progress_scale.pack(pady=5)
        
        # Playback buttons
        btn_frame = tk.Frame(self.video_controls_frame)
        btn_frame.pack(pady=5)
        
        self.backward_btn = tk.Button(
            btn_frame,
            text=self.t('backward'),
            command=self.backward_10s,
            width=8
        )
        self.backward_btn.pack(side="left", padx=2)
        
        self.play_pause_btn = tk.Button(
            btn_frame,
            text=self.t('play'),
            command=self.toggle_play_pause,
            width=8
        )
        self.play_pause_btn.pack(side="left", padx=2)
        
        self.forward_btn = tk.Button(
            btn_frame,
            text=self.t('forward'),
            command=self.forward_10s,
            width=8
        )
        self.forward_btn.pack(side="left", padx=2)
        
        # Export button
        self.export_btn = tk.Button(
            self.video_controls_frame,
            text=self.t('export'),
            command=self.export_video,
            bg="#9C27B0",
            fg="white",
            width=15
        )
        self.export_btn.pack(pady=5)
        
        # Control buttons
        button_frame = tk.Frame(self.left_panel)
        button_frame.pack(pady=20)
        
        self.start_btn = tk.Button(
            button_frame,
            text=self.t('start'),
            command=self.start_detection,
            bg="#4CAF50",
            fg="white",
            font=("Arial", 14, "bold"),
            width=12,
            height=2
        )
        self.start_btn.pack(side="left", padx=5)
        
        self.stop_btn = tk.Button(
            button_frame,
            text=self.t('stop'),
            command=self.stop_detection,
            bg="#f44336",
            fg="white",
            font=("Arial", 14, "bold"),
            width=12,
            height=2,
            state="disabled"
        )
        self.stop_btn.pack(side="left", padx=5)
        
        # Status
        self.status_var = tk.StringVar(value=self.t('ready'))
        status_label = tk.Label(self.left_panel, textvariable=self.status_var, fg="blue", font=("Arial", 10))
        status_label.pack(pady=10)
        
        # Instructions
        self.instructions_var = tk.StringVar(value=self.t('instructions'))
        instructions = tk.Label(
            self.left_panel,
            textvariable=self.instructions_var,
            justify="left",
            font=("Arial", 9)
        )
        instructions.pack(pady=10, padx=10, anchor="w")
        
        # Right panel - Video display
        self.right_panel = tk.LabelFrame(main_container, text=self.t('video_feed'), font=("Arial", 12))
        self.right_panel.pack(side="right", fill="both", expand=True, padx=5, pady=5)
        
        # Video display label
        self.video_label = tk.Label(self.right_panel, text=self.t('no_video'), bg="black")
        self.video_label.pack(fill="both", expand=True, padx=5, pady=5)
    
    def create_menu(self):
        """Create menu bar"""
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)
        
        # Language menu
        lang_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label=self.t('language'), menu=lang_menu)
        lang_menu.add_radiobutton(label=self.t('english'), command=lambda: self.set_language('en'))
        lang_menu.add_radiobutton(label=self.t('spanish'), command=lambda: self.set_language('es'))
    
    def set_language(self, lang):
        """Change language without restart"""
        self.current_lang = lang
        
        # Update title
        self.root.title(self.t('title'))
        self.title_label.config(text=self.t('title'))
        
        # Rebuild menu
        self.root.config(menu='')  # Clear menu
        self.create_menu()
    
    def use_camera(self):
        """Set video source to selected camera"""
        if not self.available_cameras:
            messagebox.showerror("Error", self.t('no_cameras'))
            return
        
        camera_idx = self.available_cameras[self.camera_combo.current()]
        self.video_source = camera_idx
        self.source_type = 'camera'
        self.selected_source_var.set(f"Selected: Camera {camera_idx}")
        self.status_var.set(f"Camera {camera_idx} selected - Click {self.t('start')}")
        
        # Hide video controls for camera
        self.video_controls_frame.pack_forget()
    
    def browse_video(self):
        """Open file dialog to select video file"""
        file_types = [
            ("Video files", "*.mp4 *.avi *.mov *.mkv *.wmv"),
            ("MP4 files", "*.mp4"),
            ("AVI files", "*.avi"),
            ("MOV files", "*.mov"),
            ("All files", "*.*")
        ]
        
        filename = filedialog.askopenfilename(
            title=self.t('browse'),
            filetypes=file_types
        )
        
        if filename:
            self.video_path_var.set(filename)
    
    def use_video(self):
        """Set video source to uploaded file"""
        video_path = self.video_path_var.get()
        
        if not video_path:
            messagebox.showerror("Error", "Please select a video file first")
            return
        
        if not os.path.exists(video_path):
            messagebox.showerror("Error", "Video file not found")
            return
        
        self.video_source = video_path
        self.source_type = 'video'
        self.selected_source_var.set(f"Selected: {os.path.basename(video_path)}")
        
        # Get video properties
        cap = cv2.VideoCapture(video_path)
        self.video_total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.video_fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        
        # Show video controls
        self.progress_scale.config(to=self.video_total_frames)
        self.video_controls_frame.pack(pady=10, padx=10, fill="x")
        
        self.status_var.set(f"Video selected - Click {self.t('start')}")
    
    def toggle_play_pause(self):
        """Toggle play/pause"""
        if self.source_type != 'video':
            return
        
        self.is_paused = not self.is_paused
        
        if self.is_paused:
            self.play_pause_btn.config(text=self.t('play'))
        else:
            self.play_pause_btn.config(text=self.t('pause'))
    
    def forward_10s(self):
        """Skip forward 10 seconds"""
        if self.source_type != 'video' or not hasattr(self, 'vs') or self.vs is None:
            return
        
        # Skip 10 seconds worth of frames
        skip_frames = int(self.video_fps * 10)
        new_pos = min(self.current_frame_pos + skip_frames, self.video_total_frames - 1)
        
        self.vs.set(cv2.CAP_PROP_POS_FRAMES, new_pos)
        self.current_frame_pos = new_pos
        self.progress_var.set(new_pos)
    
    def backward_10s(self):
        """Skip backward 10 seconds"""
        if self.source_type != 'video' or not hasattr(self, 'vs') or self.vs is None:
            return
        
        # Skip 10 seconds worth of frames
        skip_frames = int(self.video_fps * 10)
        new_pos = max(self.current_frame_pos - skip_frames, 0)
        
        self.vs.set(cv2.CAP_PROP_POS_FRAMES, new_pos)
        self.current_frame_pos = new_pos
        self.progress_var.set(new_pos)
    
    def on_seek(self, value):
        """Handle seek"""
        if self.source_type != 'video' or not hasattr(self, 'vs') or self.vs is None:
            return
        
        frame_pos = int(float(value))
        self.vs.set(cv2.CAP_PROP_POS_FRAMES, frame_pos)
        self.current_frame_pos = frame_pos
    
    def export_video(self):
        """Export the analyzed video to MP4 - processes entire video"""
        if self.video_source is None or self.source_type != 'video':
            messagebox.showwarning(self.t('export_title'), "Please select a video first")
            return
        
        # Ask for save location
        filename = filedialog.asksaveasfilename(
            title=self.t('export_title'),
            defaultextension=".mp4",
            filetypes=[("MP4 files", "*.mp4")],
            initialfile="drowsiness_analysis.mp4"
        )
        
        if not filename:
            return
        
        # Start export in a separate thread
        self.is_exporting = True
        self.export_path = filename
        
        # Disable buttons during export
        self.export_btn.config(state="disabled")
        self.status_var.set("Exporting video... Please wait.")
        
        # Start export in separate thread
        export_thread = threading.Thread(target=self.run_full_export)
        export_thread.daemon = True
        export_thread.start()
    
    def run_full_export(self):
        """Run the full video export - processes entire video from frame 0"""
        # Initialize dlib
        detector = dlib.get_frontal_face_detector()
        predictor = dlib.shape_predictor('./dlib_shape_predictor/shape_predictor_68_face_landmarks.dat')
        
        # Open video
        vs = cv2.VideoCapture(self.video_source)
        total_frames = int(vs.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = vs.get(cv2.CAP_PROP_FPS)
        
        # Get first frame to get dimensions
        ret, frame = vs.read()
        if not ret:
            vs.release()
            self.root.after(0, lambda: messagebox.showerror(self.t('export_error'), "Could not read video"))
            return
        
        frame = cv2.resize(frame, (800, 600))
        h, w = frame.shape[:2]
        
        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(self.export_path, fourcc, fps, (w, h))
        
        # Reset to beginning
        vs.set(cv2.CAP_PROP_POS_FRAMES, 0)
        
        frame_height = 576
        frame_count = 0
        
        # Get translated messages
        face_text = self.t('face_found')
        eyes_text = self.t('eyes_closed')
        yawn_text = self.t('yawning')
        tilt_text = self.t('head_tilt')
        mar_text = self.t('mar')
        
        while True:
            ret, frame = vs.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Update status every 30 frames
            if frame_count % 30 == 0:
                progress = int((frame_count / total_frames) * 100)
                self.root.after(0, lambda p=progress: self.status_var.set(f"Exporting... {p}%"))
            
            # Process frame
            try:
                if len(frame.shape) == 2:
                    frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                elif len(frame.shape) == 3 and frame.shape[2] == 4:
                    frame = frame[:, :, :3]
                
                if frame.dtype != np.uint8:
                    frame = cv2.normalize(frame, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            except:
                continue
            
            frame = cv2.resize(frame, (800, 600))
            
            if frame.dtype != np.uint8:
                frame = frame.astype(np.uint8)
            
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype('uint8')
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype('uint8')
            
            if not rgb.flags['C_CONTIGUOUS']:
                rgb = np.ascontiguousarray(rgb)
            
            if gray is None or gray.size == 0:
                writer.write(frame)
                continue
            
            size = gray.shape
            
            # Detect faces
            rects = detector(rgb)
            
            if len(rects) > 0:
                text = f"{len(rects)} {face_text}"
                cv2.putText(frame, text, (10, 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            
            for rect in rects:
                (bX, bY, bW, bH) = face_utils.rect_to_bb(rect)
                cv2.rectangle(frame, (bX, bY), (bX + bW, bY + bH), (0, 255, 0), 2)
                
                shape = predictor(rgb, rect)
                shape = face_utils.shape_to_np(shape)
                
                # Eye detection
                leftEye = shape[self.lStart:self.lEnd]
                rightEye = shape[self.rStart:self.rEnd]
                leftEAR = eye_aspect_ratio(leftEye)
                rightEAR = eye_aspect_ratio(rightEye)
                ear = (leftEAR + rightEAR) / 2.0
                
                leftEyeHull = cv2.convexHull(leftEye)
                rightEyeHull = cv2.convexHull(rightEye)
                cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
                cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)
                
                if ear < self.EYE_AR_THRESH:
                    cv2.putText(frame, eyes_text, (300, 40),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                
                # Mouth detection
                mouth = shape[self.mStart:self.mEnd]
                mouthMAR = mouth_aspect_ratio(mouth)
                mar = mouthMAR
                mouthHull = cv2.convexHull(mouth)
                cv2.drawContours(frame, [mouthHull], -1, (0, 255, 0), 1)
                cv2.putText(frame, f"{mar_text}: {mar:.2f}", (500, 20), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                
                if mar > self.MOUTH_AR_THRESH:
                    cv2.putText(frame, yawn_text, (500, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                
                # Head pose
                for (i, (x, y)) in enumerate(shape):
                    if i == 33:
                        self.image_points[0] = np.array([x, y], dtype='double')
                    elif i == 8:
                        self.image_points[1] = np.array([x, y], dtype='double')
                    elif i == 36:
                        self.image_points[2] = np.array([x, y], dtype='double')
                    elif i == 45:
                        self.image_points[3] = np.array([x, y], dtype='double')
                    elif i == 48:
                        self.image_points[4] = np.array([x, y], dtype='double')
                    elif i == 54:
                        self.image_points[5] = np.array([x, y], dtype='double')
                
                (head_tilt_degree, start_point, end_point, 
                    end_point_alt) = getHeadTiltAndCoords(size, self.image_points, frame_height)
                
                if head_tilt_degree:
                    cv2.putText(frame, f"{tilt_text} {head_tilt_degree[0]:.1f}°", (10, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            
            # Write frame
            writer.write(frame)
        
        # Cleanup
        writer.release()
        vs.release()
        
        # Update UI
        self.root.after(0, lambda: self.status_var.set(self.t('ready')))
        self.root.after(0, lambda: self.export_btn.config(state="normal"))
        self.root.after(0, lambda: messagebox.showinfo(
            self.t('export_title'), 
            f"{self.t('export_success')}\nFile: {self.export_path}"
        ))
    
    def start_detection(self):
        """Start the drowsiness detection"""
        if self.video_source is None:
            messagebox.showerror("Error", "Please select a camera or video file first")
            return
        
        self.is_running = True
        self.is_paused = False
        self.start_btn.config(state="disabled")
        self.stop_btn.config(state="normal")
        self.status_var.set(self.t('running'))
        
        # Initialize export if video
        if self.source_type == 'video':
            self.current_frame_pos = 0
            # Prepare export writer
            output_path = self.video_path_var.get().rsplit('.', 1)[0] + '_analyzed.mp4'
            self.export_path = output_path
        
        # Start detection in separate thread
        self.detection_thread = threading.Thread(target=self.run_detection)
        self.detection_thread.daemon = True
        self.detection_thread.start()
    
    def stop_detection(self):
        """Stop the drowsiness detection"""
        self.is_running = False
        self.is_paused = False
        self.start_btn.config(state="normal")
        self.stop_btn.config(state="disabled")
        self.status_var.set(self.t('stopped'))
        
        # Release export writer
        if self.export_writer is not None:
            self.export_writer.release()
            self.export_writer = None
    
    def update_video_display(self, frame):
        """Update the video display label with the current frame"""
        if frame is None:
            return
        
        # Write to export video
        if self.is_exporting and self.export_writer is None and self.source_type == 'video':
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            h, w = frame.shape[:2]
            self.export_writer = cv2.VideoWriter(self.export_path, fourcc, 30, (w, h))
        
        if self.export_writer is not None:
            self.export_writer.write(frame)
        
        # Convert frame from BGR to RGB for display
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Convert to PIL Image
        pil_image = Image.fromarray(frame_rgb)
        
        # Resize to fit the display label (maintain aspect ratio)
        display_width = 700
        display_height = 550
        pil_image.thumbnail((display_width, display_height), Image.LANCZOS)
        
        # Convert to Tkinter compatible image
        tk_image = ImageTk.PhotoImage(pil_image)
        
        # Update label
        self.video_label.imagemostrar = tk_image  # Keep reference
        self.video_label.config(image=tk_image, text="")
        
        # Update progress bar for video
        if self.source_type == 'video':
            self.progress_var.set(self.current_frame_pos)
    
    def run_detection(self):
        """Run the drowsiness detection algorithm"""
        # Initialize dlib detector in THIS thread (important for thread safety)
        print("[INFO] Initializing dlib detector in worker thread...")
        detector = dlib.get_frontal_face_detector()
        predictor = dlib.shape_predictor('./dlib_shape_predictor/shape_predictor_68_face_landmarks.dat')
        print("[INFO] Detector initialized successfully")
        
        # Initialize video source
        if self.source_type == 'camera':
            self.vs = VideoStream(src=self.video_source).start()
            time.sleep(2.0)
        else:
            self.vs = cv2.VideoCapture(self.video_source)
            self.current_frame_pos = 0
        
        frame_height = 576
        self.COUNTER = 0
        
        print(f"[INFO] Starting detection with source type: {self.source_type}")
        
        while self.is_running:
            # Check for pause
            if self.is_paused and self.source_type == 'video':
                time.sleep(0.1)
                continue
            
            # Read frame
            if self.source_type == 'camera':
                frame = self.vs.read()
                if frame is None:
                    print("[ERROR] Failed to read frame from camera")
                    break
            else:
                ret, frame = self.vs.read()
                if not ret or frame is None:
                    print("[INFO] Video ended or failed to read, stopping...")
                    break
                
                self.current_frame_pos = int(self.vs.get(cv2.CAP_PROP_POS_FRAMES))
            
            # Validate frame
            if frame is None or frame.size == 0:
                continue
            
            # Ensure frame is in correct format
            try:
                if len(frame.shape) == 2:
                    frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                elif len(frame.shape) == 3 and frame.shape[2] == 4:
                    frame = frame[:, :, :3]
                
                if frame.dtype != np.uint8:
                    frame = cv2.normalize(frame, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            except Exception as e:
                print(f"[ERROR] Frame format conversion failed: {e}")
                continue
            
            # Resize
            frame = cv2.resize(frame, (800, 600))
            
            if frame.dtype != np.uint8:
                frame = frame.astype(np.uint8)
            
            # Convert to grayscale and RGB
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype('uint8')
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype('uint8')
            
            if not rgb.flags['C_CONTIGUOUS']:
                rgb = np.ascontiguousarray(rgb)
            
            if gray is None or gray.size == 0:
                continue
            
            size = gray.shape
            
            # Detect faces
            rects = detector(rgb)
            
            # Get translated messages
            face_text = self.t('face_found')
            eyes_text = self.t('eyes_closed')
            yawn_text = self.t('yawning')
            tilt_text = self.t('head_tilt')
            mar_text = self.t('mar')
            
            if len(rects) > 0:
                text = f"{len(rects)} {face_text}"
                cv2.putText(frame, text, (10, 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            
            # Process each detected face
            for rect in rects:
                (bX, bY, bW, bH) = face_utils.rect_to_bb(rect)
                cv2.rectangle(frame, (bX, bY), (bX + bW, bY + bH), (0, 255, 0), 2)
                
                shape = predictor(rgb, rect)
                shape = face_utils.shape_to_np(shape)
                
                # Eye detection
                leftEye = shape[self.lStart:self.lEnd]
                rightEye = shape[self.rStart:self.rEnd]
                leftEAR = eye_aspect_ratio(leftEye)
                rightEAR = eye_aspect_ratio(rightEye)
                ear = (leftEAR + rightEAR) / 2.0
                
                leftEyeHull = cv2.convexHull(leftEye)
                rightEyeHull = cv2.convexHull(rightEye)
                cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
                cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)
                
                if ear < self.EYE_AR_THRESH:
                    self.COUNTER += 1
                    if self.COUNTER >= self.EYE_AR_CONSEC_FRAMES:
                        cv2.putText(frame, eyes_text, (300, 40),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                else:
                    self.COUNTER = 0
                
                # Mouth detection
                mouth = shape[self.mStart:self.mEnd]
                mouthMAR = mouth_aspect_ratio(mouth)
                mar = mouthMAR
                mouthHull = cv2.convexHull(mouth)
                cv2.drawContours(frame, [mouthHull], -1, (0, 255, 0), 1)
                cv2.putText(frame, f"{mar_text}: {mar:.2f}", (500, 20), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                
                if mar > self.MOUTH_AR_THRESH:
                    cv2.putText(frame, yawn_text, (500, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                
                # Head pose estimation
                for (i, (x, y)) in enumerate(shape):
                    if i == 33:
                        self.image_points[0] = np.array([x, y], dtype='double')
                        cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)
                    elif i == 8:
                        self.image_points[1] = np.array([x, y], dtype='double')
                        cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)
                    elif i == 36:
                        self.image_points[2] = np.array([x, y], dtype='double')
                        cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)
                    elif i == 45:
                        self.image_points[3] = np.array([x, y], dtype='double')
                        cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)
                    elif i == 48:
                        self.image_points[4] = np.array([x, y], dtype='double')
                        cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)
                    elif i == 54:
                        self.image_points[5] = np.array([x, y], dtype='double')
                        cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)
                
                for p in self.image_points:
                    cv2.circle(frame, (int(p[0]), int(p[1])), 3, (0, 0, 255), -1)
                
                (head_tilt_degree, start_point, end_point, 
                    end_point_alt) = getHeadTiltAndCoords(size, self.image_points, frame_height)
                
                cv2.line(frame, start_point, end_point, (255, 0, 0), 2)
                cv2.line(frame, start_point, end_point_alt, (0, 0, 255), 2)
                
                if head_tilt_degree:
                    cv2.putText(frame, f"{tilt_text} {head_tilt_degree[0]:.1f}°", (10, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            
            # Update video display in GUI
            self.root.after(0, self.update_video_display, frame)
            
            # Small delay to prevent GUI freezing
            time.sleep(0.01)
        
        # Cleanup
        if self.source_type == 'camera' and self.vs:
            self.vs.stop()
        elif self.vs:
            self.vs.release()
        
        # Release export writer
        if self.export_writer is not None:
            self.export_writer.release()
            self.export_writer = None
            self.root.after(0, lambda: messagebox.showinfo(
                self.t('export_title'), 
                f"{self.t('export_success')}\nFile: {self.export_path}"
            ))
        
        # Reset UI
        self.root.after(0, self.stop_detection)
        self.root.after(0, lambda: self.status_var.set("Detection completed"))


def main():
    root = tk.Tk()
    app = DrowsinessDetectorGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
