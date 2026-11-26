"""Laser line-tracker configuration. Tune values here, not in the code."""

# ---- hardware ----
SERIAL_PORT = "/dev/ttyACM0"      # Arduino Mega 2560
SERIAL_BAUD = 115200
CAMERA_INDEX = 0                  # Arducam B0495 -> /dev/video0
FRAME_W, FRAME_H, FPS = 960, 600, 30

# ---- servos (STS3215: 4096 ticks/rev, 1 tick = 0.088 deg) ----
# joint order = firmware SERVO_IDS order: [base yaw, shoulder, elbow]
JOINT_MIN = [1024, 1024, 1024]    # !! set safe limits for your arm before running
JOINT_MAX = [3072, 3072, 3072]
SERVO_SPEED = 0                   # 0 = max speed (steps/s otherwise)
SERVO_ACC = 0                     # 0 = max acceleration (0..254)

# ---- vision ----
PROC_SCALE = 0.5                  # process at reduced resolution for speed
DARK_MAX = 90                     # gray level below which a pixel may be "black line"
MIN_LINE_AREA = 150               # px (at processed scale), ignore smaller blobs
LOCAL_RADIUS = 25                 # px (processed), neighbourhood for line direction (PCA)
LOOKAHEAD = 20                    # px (processed), how far ahead along the line to aim; 0 = hold
ROI_MARGIN = 0                    # px (processed) cropped from each border (vignetting/arm parts)

# ---- control (image-based visual servoing) ----
GAIN = 0.25                       # fraction of the pixel error corrected per frame
DAMPING = 0.1                     # damped least-squares lambda (px/tick); raise if motion jerks near singularities
JOINT_WEIGHTS = [1.0, 0.5, 1.0]   # lower weight = joint is used less
NULLSPACE_GAIN = 0.02             # pull redundant DOF back toward HOME
MAX_STEP = 30                     # max ticks per joint per frame
DEADBAND_PX = 2.0                 # full-res pixels
LOST_FRAMES_LIMIT = 15            # frames without line before declaring "lost"

# ---- calibration ----
CALIB_FILE = "calib.json"
CALIB_DELTA = 60                  # ticks to jog each joint during Jacobian calibration
CALIB_SETTLE_S = 0.7
