import json
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse
import cv2
import numpy as np


HOST = "127.0.0.1"
PORT = 8000


# =========================================================
# REAL OPENCV PERSON DETECTOR
# =========================================================

HOG = cv2.HOGDescriptor()

HOG.setSVMDetector(
    cv2.HOGDescriptor_getDefaultPeopleDetector()
)


# =========================================================
# HAZARD CONFIGURATION
# =========================================================

HAZARD = {
    "fire": 40,
    "flood": 35,
    "collapse": 40,
    "landslide": 35,
    "normal": 10
}


HAZARD_LABEL = {
    "fire": "Active fire",
    "flood": "Rising flood water",
    "collapse": "Structural collapse",
    "landslide": "Landslide / debris",
    "normal": "No major hazard selected"
}


# =========================================================
# PEOPLE DETECTION
# =========================================================

def detect_people(image):

    height, width = image.shape[:2]

    # Resize large frames for faster processing
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
            )
        )

    else:

        small = image
        scale = 1.0


    # REAL PERSON DETECTION
    boxes, weights = HOG.detectMultiScale(
        small,
        winStride=(8, 8),
        padding=(8, 8),
        scale=1.05
    )


    results = []


    for (x, y, w, h), weight in zip(
        boxes,
        weights
    ):

        confidence = float(weight)


        if confidence < 0.25:
            continue


        results.append({

            "x": int(x / scale),

            "y": int(y / scale),

            "w": int(w / scale),

            "h": int(h / scale),

            "confidence": round(
                confidence,
                2
            )

        })


    return results


# =========================================================
# RISK SCORING
# =========================================================

def calculate_risk(
    victims,
    hazard,
    isolation
):

    # Victim contribution
    victim_score = min(
        40,
        victims * 4
    )


    # Hazard contribution
    hazard_score = HAZARD.get(
        hazard,
        10
    )


    # Isolation contribution
    isolation_score = max(
        0,
        min(
            20,
            int(isolation)
        )
    )


    # Final score
    score = min(
        100,
        victim_score
        + hazard_score
        + isolation_score
    )


    # Threat level

    if score >= 75:

        level = "CRITICAL"

    elif score >= 40:

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

    if hazard == "flood":

        team = "High-Water Rescue Unit"


    elif hazard == "fire":

        team = "Fire & Rescue Unit"


    elif hazard == "collapse":

        team = "Heavy Rescue Unit"


    elif hazard == "landslide":

        team = "Debris Rescue Unit"


    else:

        team = "Rapid Response Unit"


    # Medical priority

    if score >= 60:

        medical = "Medical Unit 01"

    else:

        medical = "Medical Unit 03"


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
# PYTHON 3.13 COMPATIBLE
# =========================================================

def parse_multipart(
    body,
    content_type
):

    fields = {}


    # Get boundary
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


    # Split all multipart sections
    parts = body.split(
        boundary_bytes
    )


    for part in parts:

        if not part:
            continue


        # Remove CRLF
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


        # ---------------------------------
        # Extract field name
        # ---------------------------------

        name = None


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


        # ---------------------------------
        # File field
        # ---------------------------------

        if name == "image":

            fields["image"] = data


        # ---------------------------------
        # Normal text fields
        # ---------------------------------

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


    # -------------------------------------
    # JSON RESPONSE
    # -------------------------------------

    def send_json(
        self,
        status,
        data
    ):

        response = json.dumps(
            data
        ).encode("utf-8")


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
            "Content-Length",
            str(len(response))
        )


        self.end_headers()


        self.wfile.write(
            response
        )


    # -------------------------------------
    # GET
    # -------------------------------------

    def do_GET(self):

        path = urlparse(
            self.path
        ).path


        # Homepage
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


        # Health check
        elif path == "/health":

            self.send_json(
                200,
                {
                    "status": "online",
                    "engine": "OpenCV HOG"
                }
            )


        else:

            self.send_json(
                404,
                {
                    "error": "Not found"
                }
            )


    # -------------------------------------
    # POST /detect
    # -------------------------------------

    def do_POST(self):

        path = urlparse(
            self.path
        ).path


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

            # ---------------------------------
            # Read request body
            # ---------------------------------

            content_length = int(
                self.headers.get(
                    "Content-Length",
                    "0"
                )
            )


            body = self.rfile.read(
                content_length
            )


            # ---------------------------------
            # Content type
            # ---------------------------------

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


            # ---------------------------------
            # Parse FormData
            # ---------------------------------

            fields = parse_multipart(
                body,
                content_type
            )


            # ---------------------------------
            # Get image
            # ---------------------------------

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


            # ---------------------------------
            # Decode image
            # ---------------------------------

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


            # ---------------------------------
            # Get hazard
            # ---------------------------------

            hazard = fields.get(
                "hazard",
                "collapse"
            )


            # ---------------------------------
            # Get isolation
            # ---------------------------------

            isolation = fields.get(
                "isolation",
                "10"
            )


            try:

                isolation = int(
                    isolation
                )

            except:

                isolation = 10


            # ---------------------------------
            # REAL AI / CV DETECTION
            # ---------------------------------

            people_boxes = detect_people(
                image
            )


            people_count = len(
                people_boxes
            )


            # ---------------------------------
            # RISK SCORE
            # ---------------------------------

            score, level = calculate_risk(
                people_count,
                hazard,
                isolation
            )


            # ---------------------------------
            # SMART DISPATCH
            # ---------------------------------

            team, medical, route = (
                get_recommendation(
                    hazard,
                    score
                )
            )


            # ---------------------------------
            # Final response
            # ---------------------------------

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
                route
            }


            # Print AI result in CMD
            print(
                "[AI] "
                f"People={people_count} | "
                f"Risk={score} | "
                f"Level={level} | "
                f"Hazard={hazard}"
            )


            # Send response
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
        "Server  : http://127.0.0.1:8000"
    )

    print(
        "Engine  : OpenCV HOG"
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