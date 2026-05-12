# Parallel arrays. Invalid depth: depth_m=0 and x,y,z are NaN.
# (x,y,z) in meters, camera optical frame: z forward along ray, x right, y down (RGB camera_info K).
# angle_rad / angle_deg / cx_norm / cy_norm copied from Centers2D for the same detections.
std_msgs/Header header
string[] obj_class
float32[] score
uint32[] u
uint32[] v
float32[] depth_m
float32[] x
float32[] y
float32[] z
float32[] angle_rad
float32[] angle_deg
float32[] cx_norm
float32[] cy_norm
uint32 frame_width
uint32 frame_height
# Keypoints forwarded from Centers2D (same layout, same kpts_per_det)
uint32 kpts_per_det
float32[] kpt_u
float32[] kpt_v
float32[] kpt_conf
