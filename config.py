# Camera and image configuration
NUMBER_OF_CAMERAS = 3
FRAME_WIDTH_PIXELS = 1280
FRAME_HEIGHT_PIXELS = 720

# Dartboard dimensions in millimeters
DARTBOARD_DIAMETER_MM = 451
MM_TO_PIXELS_SCALE = FRAME_HEIGHT_PIXELS / DARTBOARD_DIAMETER_MM

# Dartboard radii in millimeters
BULLSEYE_RADIUS_MM = 6.35
OUTER_BULLSEYE_RADIUS_MM = 15.9
TRIPLE_RING_INNER_RADIUS_MM = 99
TRIPLE_RING_OUTER_RADIUS_MM = 107
DOUBLE_RING_INNER_RADIUS_MM = 162
DOUBLE_RING_OUTER_RADIUS_MM = 170

# Dartboard radii in pixels
BULLSEYE_RADIUS_PIXELS = int(BULLSEYE_RADIUS_MM * MM_TO_PIXELS_SCALE)
OUTER_BULLSEYE_RADIUS_PIXELS = int(OUTER_BULLSEYE_RADIUS_MM * MM_TO_PIXELS_SCALE)
TRIPLE_RING_INNER_RADIUS_PIXELS = int(TRIPLE_RING_INNER_RADIUS_MM * MM_TO_PIXELS_SCALE)
TRIPLE_RING_OUTER_RADIUS_PIXELS = int(TRIPLE_RING_OUTER_RADIUS_MM * MM_TO_PIXELS_SCALE)
DOUBLE_RING_INNER_RADIUS_PIXELS = int(DOUBLE_RING_INNER_RADIUS_MM * MM_TO_PIXELS_SCALE)
DOUBLE_RING_OUTER_RADIUS_PIXELS = int(DOUBLE_RING_OUTER_RADIUS_MM * MM_TO_PIXELS_SCALE)

# Dartboard center position
DARTBOARD_CENTER_COORDS = (FRAME_WIDTH_PIXELS // 2, FRAME_HEIGHT_PIXELS // 2)
