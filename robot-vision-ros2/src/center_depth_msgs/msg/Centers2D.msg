# Parallel arrays (same length). header = RGB image used for inference.
# angle_rad: OBB rotation in image plane (Ultralytics xywhr[4]), i.e. in-plane vs. image horizontal
#   (camera +X projects along +u when using standard pinhole; see SDK depth–RGB alignment §3.1).
# angle_deg: degree form of angle_rad for direct use in UI/logs.
# cx_norm/cy_norm: center / frame size, same convention as yolo_obb_pose/obb/detect_obb.py angles.csv.
std_msgs/Header header
string[] obj_class
float32[] score
uint32[] u
uint32[] v
float32[] angle_rad
float32[] angle_deg
float32[] cx_norm
float32[] cy_norm
uint32 frame_width
uint32 frame_height
# Keypoints (pose mode only; all empty if task != pose)
# kpts_per_det: K keypoints per detection; total array length = len(u) * K
uint32 kpts_per_det
float32[] kpt_u        # pixel x, flattened row-major (det0_kpt0, det0_kpt1, ..., det1_kpt0, ...)
float32[] kpt_v        # pixel y, flattened row-major
float32[] kpt_conf     # visibility / confidence [0,1], flattened row-major
