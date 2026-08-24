RESQ.AI REAL COMPUTER-VISION PROTOTYPE

Run:
1. Install Python 3.10+.
2. Open Command Prompt in this folder.
3. Run: pip install opencv-python numpy
4. Run: python server.py
5. Open: http://127.0.0.1:8000
6. Allow camera permission.
7. Click START CAMERA.

REAL FEATURES:
- Browser webcam feed.
- OpenCV HOG people detector.
- Real bounding boxes and people count.
- Dynamic 0-100 risk scoring based on victims, selected hazard severity and isolation.
- Automatic rescue-team + medical-unit recommendation.
- Dispatch action.

The supplied deck's conceptual scoring formula is used. This is a prototype, not a production emergency system.
For the next upgrade, replace HOG with YOLOv8 custom-trained classes for people/fire/flood/collapse.
