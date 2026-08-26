import json
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse

import cv2
import numpy as np
from ultralytics import YOLO


# =========================================================
# RESQ.AI CONFIGURATION
# =========================================================

HOST = "127.0.0.1"
PORT = 8000

MODEL_PATH = "yolo11n.pt"

# YOLO settings
CONFIDENCE_THRESHOLD = 0.25
IOU_THRESHOLD = 0.45
IMAGE_SIZE = 640


# =========================================================
# LOAD YOLO MODEL
# =========================================================

print("[AI] Loading YOLO model...")

try:
    MODEL = YOLO(MODEL_PATH)
    print("[AI] YOLO model loaded successfully.")
except Exception as error:
    print("[AI ERROR] Could not load YOLO model.")
    print("[AI ERROR]", repr(error))
    raise


# =========================================================
# HAZARD CONFIGURATION
# =========================================================

HAZARD = {
    "fire": 40,
    "flood": 30,
    "collapse": 35,
    "landslide": 35,
    "normal": 5
}


HAZARD_LABEL = {
    "fire": "Active fire",
    "flood": "Rising flood water",
    "collapse": "Structural collapse",
    "landslide": "Landslide / debris",
    "normal": "No major hazard selected"
}


# =========================================================
# PEOPLE DETECTION USING YOLO
# =========================================================

def detect_people(image):

    height, width = image.shape[:2]

    # -----------------------------------------------------
    # Resize large frames for faster processing
    # -----------------------------------------------------

    scale = min(
        1.0,
        900.0 / max(width, 1)
    )

    if scale < 1.0:

        small = cv2.resize(
            image,
            (
                int(width * scale),
                int(height * scale)
            ),
            interpolation=cv2.INTER_AREA
        )

    else:

        small = image
        scale = 1.0


    # -----------------------------------------------------
    # YOLO PERSON DETECTION
    # COCO class 0 = person
    # -----------------------------------------------------

    try:

        results = MODEL.predict(
            source=small,
            conf=CONFIDENCE_THRESHOLD,
            iou=IOU_THRESHOLD,
            imgsz=IMAGE_SIZE,
            classes=[0],
            verbose=False
        )

    except Exception as error:

        print("[AI ERROR] YOLO inference failed:", repr(error))

        return []


    detections = []


    # -----------------------------------------------------
    # PROCESS YOLO RESULTS
    # -----------------------------------------------------

    for result in results:

        if result.boxes is None:
            continue

        boxes = result.boxes


        for i in range(len(boxes)):

            try:

                cls = int(
                    boxes.cls[i].item()
                )

                confidence = float(
                    boxes.conf[i].item()
                )

            except Exception:

                continue


            # Safety check:
            # Only person class
            if cls != 0:
                continue


            # Additional confidence filter
            if confidence < CONFIDENCE_THRESHOLD:
                continue


            # -------------------------------------------------
            # Bounding box
            # -------------------------------------------------

            x1, y1, x2, y2 = boxes.xyxy[i].tolist()


            # Convert coordinates back to original image
            x1 = int(x1 / scale)
            y1 = int(y1 / scale)
            x2 = int(x2 / scale)
            y2 = int(y2 / scale)


            # Keep coordinates inside image
            x1 = max(0, min(x1, width - 1))
            y1 = max(0, min(y1, height - 1))
            x2 = max(0, min(x2, width - 1))
            y2 = max(0, min(y2, height - 1))


            box_width = max(
                1,
                x2 - x1
            )

            box_height = max(
                1,
                y2 - y1
            )


            # Ignore extremely tiny detections
            if box_width < 15 or box_height < 15:
                continue


            detections.append({

                "x": x1,

                "y": y1,

                "w": box_width,

                "h": box_height,

                "confidence": round(
                    confidence,
                    2
                )

            })


    # -----------------------------------------------------
    # SORT BY CONFIDENCE
    # Highest confidence first
    # -----------------------------------------------------

    detections.sort(
        key=lambda item: item["confidence"],
        reverse=True
    )


    # -----------------------------------------------------
    # EXTRA DUPLICATE PROTECTION
    # YOLO already performs NMS, but this keeps the
    # returned list clean for the dashboard.
    # -----------------------------------------------------

    filtered = []


    def calculate_iou(box_a, box_b):

        ax1 = box_a["x"]
        ay1 = box_a["y"]
        ax2 = ax1 + box_a["w"]
        ay2 = ay1 + box_a["h"]

        bx1 = box_b["x"]
        by1 = box_b["y"]
        bx2 = bx1 + box_b["w"]
        by2 = by1 + box_b["h"]


        intersection_x1 = max(
            ax1,
            bx1
        )

        intersection_y1 = max(
            ay1,
            by1
        )

        intersection_x2 = min(
            ax2,
            bx2
        )

        intersection_y2 = min(
            ay2,
            by2
        )


        intersection_width = max(
            0,
            intersection_x2 - intersection_x1
        )

        intersection_height = max(
            0,
            intersection_y2 - intersection_y1
        )


        intersection_area = (
            intersection_width *
            intersection_height
        )


        area_a = (
            box_a["w"] *
            box_a["h"]
        )

        area_b = (
            box_b["w"] *
            box_b["h"]
        )


        union_area = (
            area_a +
            area_b -
            intersection_area
        )


        if union_area <= 0:
            return 0.0


        return (
            intersection_area /
            union_area
        )


    for detection in detections:

        duplicate = False


        for existing in filtered:

            overlap = calculate_iou(
                detection,
                existing
            )


            if overlap >= 0.60:

                duplicate = True
                break


        if not duplicate:

            filtered.append(
                detection
            )


    return filtered


# =========================================================
# RISK SCORING
# =========================================================

def calculate_risk(
    victims,
    hazard,
    isolation
):

    # -----------------------------------------------------
    # Hazard contribution
    # -----------------------------------------------------

    hazard_score = HAZARD.get(
        hazard,
        5
    )


    # -----------------------------------------------------
    # Victim contribution
    # Maximum 40 points
    # -----------------------------------------------------

    victim_score = min(
        40,
        victims * 10
    )


    # -----------------------------------------------------
    # Isolation contribution
    # Maximum 20 points
    # -----------------------------------------------------

    try:

        isolation = int(
            isolation
        )

    except (
        ValueError,
        TypeError
    ):

        isolation = 10


    isolation_score = max(
        0,
        min(
            20,
            isolation
        )
    )


    # -----------------------------------------------------
    # Final score
    # -----------------------------------------------------

    score = min(
        100,
        hazard_score +
        victim_score +
        isolation_score
    )


    # -----------------------------------------------------
    # Threat level
    # -----------------------------------------------------

    if score >= 80:

        level = "CRITICAL"

    elif score >= 55:

        level = "HIGH"

    elif score >= 30:

        level = "MEDIUM"

    else:

        level = "LOW"


    return score, level


# =========================================================
# SMART DISPATCH
# =========================================================

def get_recommendation(
    hazard,
    score
):

    # -----------------------------------------------------
    # Rescue team
    # -----------------------------------------------------

    teams = {

        "fire":
        "Fire & Rescue Unit",

        "collapse":
        "Heavy Rescue Unit",

        "flood":
        "High-Water Rescue Unit",

        "landslide":
        "Debris Rescue Unit",

        "normal":
        "Rapid Response Unit"

    }


    team = teams.get(
        hazard,
        "Rapid Response Unit"
    )


    # -----------------------------------------------------
    # Medical priority
    # -----------------------------------------------------

    if score >= 80:

        medical = (
            "Medical Unit 01 "
            "(Critical Care)"
        )

    elif score >= 55:

        medical = (
            "Medical Unit 02 "
            "(Advanced Life Support)"
        )

    else:

        medical = (
            "Medical Unit 03"
        )


    # -----------------------------------------------------
    # Route recommendation
    # -----------------------------------------------------

    route = (
        "Safest route selected: "
        "avoid blocked / hazard corridor"
    )


    return (
        team,
        medical,
        route
    )


# =========================================================
# MULTIPART FORM DATA PARSER
# =========================================================

def parse_multipart(
    body,
    content_type
):

    fields = {}


    boundary_marker = "boundary="


    if boundary_marker not in content_type:

        return fields


    boundary = content_type.split(
        boundary_marker,
        1
    )[1]


    boundary = boundary.strip()


    if boundary.startswith('"'):

        boundary = boundary.strip('"')


    boundary_bytes = (
        "--" + boundary
    ).encode()


    parts = body.split(
        boundary_bytes
    )


    for part in parts:

        if not part:
            continue


        # Remove multipart separators
        part = part.strip(
            b"\r\n-"
        )


        if not part:
            continue


        separator = (
            b"\r\n\r\n"
        )


        if separator not in part:
            continue


        header_data, data = part.split(
            separator,
            1
        )


        header_text = header_data.decode(
            "utf-8",
            errors="ignore"
        )


        # Remove final CRLF
        data = data.rstrip(
            b"\r\n"
        )


        name = None


        # -------------------------------------------------
        # Extract field name
        # -------------------------------------------------

        for line in header_text.split(
            "\r\n"
        ):

            if "Content-Disposition" in line:

                if 'name="' in line:

                    name = line.split(
                        'name="',
                        1
                    )[1].split(
                        '"',
                        1
                    )[0]


        if not name:
            continue


        # -------------------------------------------------
        # Image field
        # -------------------------------------------------

        if name == "image":

            fields["image"] = data


        # -------------------------------------------------
        # Normal form fields
        # -------------------------------------------------

        else:

            fields[name] = data.decode(
                "utf-8",
                errors="ignore"
            )


    return fields


# =========================================================
# HTTP SERVER
# =========================================================

class ResQHandler(
    BaseHTTPRequestHandler
):


    # -----------------------------------------------------
    # Disable noisy default logging
    # We still print AI results ourselves.
    # -----------------------------------------------------

    def log_message(
        self,
        format,
        *args
    ):

        print(
            "[HTTP]",
            format % args
        )


    # -----------------------------------------------------
    # JSON RESPONSE
    # -----------------------------------------------------

    def send_json(
        self,
        status,
        data
    ):

        response = json.dumps(
            data
        ).encode(
            "utf-8"
        )


        self.send_response(
            status
        )


        self.send_header(
            "Content-Type",
            "application/json"
        )


        self.send_header(
            "Access-Control-Allow-Origin",
            "*"
        )


        self.send_header(
            "Cache-Control",
            "no-store"
        )


        self.send_header(
            "Content-Length",
            str(len(response))
        )


        self.end_headers()


        self.wfile.write(
            response
        )


    # -----------------------------------------------------
    # GET
    # -----------------------------------------------------

    def do_GET(self):

        path = urlparse(
            self.path
        ).path


        # -------------------------------------------------
        # Homepage
        # -------------------------------------------------

        if path == "/":

            try:

                with open(
                    "index.html",
                    "rb"
                ) as file:

                    data = file.read()


                self.send_response(
                    200
                )


                self.send_header(
                    "Content-Type",
                    "text/html; charset=utf-8"
                )


                self.send_header(
                    "Cache-Control",
                    "no-store"
                )


                self.send_header(
                    "Content-Length",
                    str(len(data))
                )


                self.end_headers()


                self.wfile.write(
                    data
                )


            except Exception as error:

                self.send_json(
                    500,
                    {
                        "error": str(error)
                    }
                )


        # -------------------------------------------------
        # Health check
        # -------------------------------------------------

        elif path == "/health":

            self.send_json(
                200,
                {
                    "status": "online",
                    "engine": "YOLO",
                    "model": MODEL_PATH
                }
            )


        # -------------------------------------------------
        # 404
        # -------------------------------------------------

        else:

            self.send_json(
                404,
                {
                    "error": "Not found"
                }
            )


    # -----------------------------------------------------
    # POST /detect
    # -----------------------------------------------------

    def do_POST(self):

        path = urlparse(
            self.path
        ).path


        # -------------------------------------------------
        # Validate endpoint
        # -------------------------------------------------

        if path != "/detect":

            self.send_json(
                404,
                {
                    "error":
                    "Endpoint not found"
                }
            )

            return


        try:

            # ---------------------------------------------
            # Read request body
            # ---------------------------------------------

            content_length = int(
                self.headers.get(
                    "Content-Length",
                    "0"
                )
            )


            if content_length <= 0:

                self.send_json(
                    400,
                    {
                        "error":
                        "Empty request body"
                    }
                )

                return


            body = self.rfile.read(
                content_length
            )


            # ---------------------------------------------
            # Content type
            # ---------------------------------------------

            content_type = self.headers.get(
                "Content-Type",
                ""
            )


            if not content_type:

                self.send_json(
                    400,
                    {
                        "error":
                        "Content-Type missing"
                    }
                )

                return


            # ---------------------------------------------
            # Parse multipart form
            # ---------------------------------------------

            fields = parse_multipart(
                body,
                content_type
            )


            # ---------------------------------------------
            # Get image
            # ---------------------------------------------

            image_data = fields.get(
                "image"
            )


            if image_data is None:

                print(
                    "[ERROR] Image field missing"
                )


                self.send_json(
                    400,
                    {
                        "error":
                        "Image field missing"
                    }
                )

                return


            if len(image_data) == 0:

                self.send_json(
                    400,
                    {
                        "error":
                        "Image is empty"
                    }
                )

                return


            # ---------------------------------------------
            # Decode image
            # ---------------------------------------------

            image_array = np.frombuffer(
                image_data,
                dtype=np.uint8
            )


            image = cv2.imdecode(
                image_array,
                cv2.IMREAD_COLOR
            )


            if image is None:

                print(
                    "[ERROR] Image decode failed"
                )


                self.send_json(
                    400,
                    {
                        "error":
                        "Could not decode image"
                    }
                )

                return


            # ---------------------------------------------
            # Get hazard
            # ---------------------------------------------

            hazard = fields.get(
                "hazard",
                "collapse"
            )


            # Only accept known hazards
            if hazard not in HAZARD:

                hazard = "collapse"


            # ---------------------------------------------
            # Get isolation
            # ---------------------------------------------

            isolation = fields.get(
                "isolation",
                "10"
            )


            try:

                isolation = int(
                    isolation
                )

            except (
                ValueError,
                TypeError
            ):

                isolation = 10


            isolation = max(
                0,
                min(
                    20,
                    isolation
                )
            )


            # ---------------------------------------------
            # YOLO DETECTION
            # ---------------------------------------------

            people_boxes = detect_people(
                image
            )


            people_count = len(
                people_boxes
            )


            # ---------------------------------------------
            # Calculate average confidence
            # ---------------------------------------------

            if people_boxes:

                average_confidence = round(
                    sum(
                        box["confidence"]
                        for box in people_boxes
                    ) / len(people_boxes),
                    2
                )

            else:

                average_confidence = 0.0


            # ---------------------------------------------
            # Risk score
            # ---------------------------------------------

            score, level = calculate_risk(
                people_count,
                hazard,
                isolation
            )


            # ---------------------------------------------
            # Smart dispatch
            # ---------------------------------------------

            team, medical, route = (
                get_recommendation(
                    hazard,
                    score
                )
            )


            # ---------------------------------------------
            # Final response
            # ---------------------------------------------

            result = {

                "people":
                people_count,

                "boxes":
                people_boxes,

                "score":
                score,

                "level":
                level,

                "hazard":
                HAZARD_LABEL.get(
                    hazard,
                    hazard
                ),

                "team":
                team,

                "medical":
                medical,

                "route":
                route,

                "engine":
                "YOLO",

                "model":
                MODEL_PATH,

                "average_confidence":
                average_confidence

            }


            # ---------------------------------------------
            # Print AI result in CMD
            # ---------------------------------------------

            print(
                "[AI] "
                f"People={people_count} | "
                f"Confidence={average_confidence:.2f} | "
                f"Risk={score} | "
                f"Level={level} | "
                f"Hazard={hazard}"
            )


            # ---------------------------------------------
            # Send response
            # ---------------------------------------------

            self.send_json(
                200,
                result
            )


        except Exception as error:

            print(
                "[SERVER ERROR]",
                repr(error)
            )


            self.send_json(
                500,
                {
                    "error":
                    str(error)
                }
            )


# =========================================================
# START SERVER
# =========================================================

if __name__ == "__main__":

    print()
    print(
        "========================================"
    )

    print(
        "          RESQ.AI AI ENGINE"
    )

    print(
        "========================================"
    )

    print()

    print(
        f"Server  : http://{HOST}:{PORT}"
    )

    print(
        "Engine  : YOLO"
    )

    print(
        f"Model   : {MODEL_PATH}"
    )

    print(
        f"Confidence threshold : {CONFIDENCE_THRESHOLD}"
    )

    print(
        "Status  : ONLINE"
    )

    print()

    print(
        "Waiting for camera frames..."
    )

    print()


    server = HTTPServer(
        (
            HOST,
            PORT
        ),
        ResQHandler
    )


    try:

        server.serve_forever()


    except KeyboardInterrupt:

        print()

        print(
            "ResQ.AI server stopped."
        )

        server.server_close()
        