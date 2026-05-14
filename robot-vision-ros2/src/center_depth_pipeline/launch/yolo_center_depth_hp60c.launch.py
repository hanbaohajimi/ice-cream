"""
HP60C 专用包装 launch。
直接包含 yolo_center_depth.launch.py 并固定 HP60C 话题前缀。

所有参数默认值从仓库根 config.yaml 读取，修改 config.yaml 即可生效，
命令行传参优先级高于 config.yaml。

模型路径也可通过环境变量覆盖：
  ROBOT_WEIGHTS_PATH    - YOLO 权重文件路径
  ROBOT_POSE_CLASSES    - pose_classes.yaml 路径
"""

import os
from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration


def _read_config() -> dict:
    """从仓库根 config.yaml 读取全部配置，返回原始 dict。"""
    try:
        import yaml
        here = Path(__file__).resolve().parent
        for _ in range(6):
            candidate = here / "config.yaml"
            if candidate.exists():
                with open(candidate, encoding="utf-8") as f:
                    return yaml.safe_load(f) or {}
            here = here.parent
    except Exception:
        pass
    return {}


def generate_launch_description() -> LaunchDescription:
    pkg_share  = get_package_share_directory("center_depth_pipeline")
    main_launch = os.path.join(pkg_share, "launch", "yolo_center_depth.launch.py")

    cfg = _read_config()
    cam    = cfg.get("camera",    {})
    model  = cfg.get("model",     {})
    det    = cfg.get("detection", {})
    depth  = cfg.get("depth",     {})

    # model.weights_path / pose_classes_path 支持环境变量覆盖
    _weights = model.get("weights_path", "")
    _classes = model.get("pose_classes_path", "")
    default_weights      = os.environ.get("ROBOT_WEIGHTS_PATH", _weights)
    default_pose_classes = os.environ.get("ROBOT_POSE_CLASSES", _classes)

    # ── declare all forwarded arguments ───────────────────────────────────────
    weights_path         = LaunchConfiguration("weights_path")
    pose_classes_path    = LaunchConfiguration("pose_classes_path")
    sample_radius        = LaunchConfiguration("sample_radius")
    sync_slop_sec        = LaunchConfiguration("sync_slop_sec")
    conf_threshold       = LaunchConfiguration("conf_threshold")
    device               = LaunchConfiguration("device")
    ema_alpha            = LaunchConfiguration("ema_alpha")
    use_half             = LaunchConfiguration("use_half")
    imgsz                = LaunchConfiguration("imgsz")
    min_stable_frames    = LaunchConfiguration("min_stable_frames")
    min_depth_m          = LaunchConfiguration("min_depth_m")
    max_depth_m          = LaunchConfiguration("max_depth_m")
    use_compressed       = LaunchConfiguration("use_compressed")
    use_compressed_depth = LaunchConfiguration("use_compressed_depth")
    max_det              = LaunchConfiguration("max_det")
    log_detections          = LaunchConfiguration("log_detections")
    log_empty_throttle_sec  = LaunchConfiguration("log_empty_throttle_sec")

    _HP60C_RGB   = cam.get("rgb_topic",          "/ascamera_hp60c/camera_publisher/rgb0/image/compressed")
    _HP60C_DEPTH = cam.get("depth_topic",         "/ascamera_hp60c/camera_publisher/depth0/image_raw/compressedDepth")
    _HP60C_INFO  = cam.get("camera_info_topic",   "/ascamera_hp60c/camera_publisher/rgb0/camera_info")

    return LaunchDescription([
        DeclareLaunchArgument(
            "weights_path",
            default_value=default_weights,
            description="YOLO .pt weights；默认从 config.yaml model.weights_path 读取",
        ),
        DeclareLaunchArgument(
            "pose_classes_path",
            default_value=default_pose_classes,
            description="关键点角色语义定义文件 pose_classes.yaml",
        ),
        DeclareLaunchArgument("sample_radius",        default_value=str(depth.get("sample_radius",        8))),
        DeclareLaunchArgument("sync_slop_sec",        default_value=str(depth.get("sync_slop_sec",        0.12))),
        DeclareLaunchArgument("conf_threshold",       default_value=str(det.get("conf_threshold",         0.25))),
        DeclareLaunchArgument("device",               default_value=str(model.get("device",               "cuda:0"))),
        DeclareLaunchArgument("ema_alpha",            default_value=str(det.get("ema_alpha",              0.4))),
        DeclareLaunchArgument("use_half",             default_value=str(model.get("use_half",             True))),
        DeclareLaunchArgument("imgsz",                default_value=str(model.get("imgsz",                640))),
        DeclareLaunchArgument("min_stable_frames",    default_value=str(det.get("min_stable_frames",      2))),
        DeclareLaunchArgument("min_depth_m",          default_value=str(depth.get("min_depth_m",          0.05))),
        DeclareLaunchArgument("max_depth_m",          default_value=str(depth.get("max_depth_m",          4.0))),
        DeclareLaunchArgument(
            "use_compressed", default_value=str(cam.get("use_compressed", True)).lower(),
            description="true=CompressedImage，false=原始 Image（用于 RGB）"),
        DeclareLaunchArgument(
            "use_compressed_depth", default_value=str(cam.get("use_compressed_depth", True)).lower(),
            description="true=compressedDepth，false=原始 Image（用于深度）"),
        DeclareLaunchArgument("max_det",                  default_value=str(det.get("max_det",                   50))),
        DeclareLaunchArgument("log_detections",           default_value=str(det.get("log_detections",            True)).lower()),
        DeclareLaunchArgument("log_empty_throttle_sec",   default_value=str(det.get("log_empty_throttle_sec",    2.0))),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(main_launch),
            launch_arguments=[
                ("rgb_topic",            _HP60C_RGB),
                ("depth_topic",          _HP60C_DEPTH),
                ("camera_info_topic",    _HP60C_INFO),
                ("use_compressed",       use_compressed),
                ("use_compressed_depth", use_compressed_depth),
                ("weights_path",         weights_path),
                ("pose_classes_path",    pose_classes_path),
                ("sample_radius",        sample_radius),
                ("sync_slop_sec",        sync_slop_sec),
                ("conf_threshold",       conf_threshold),
                ("device",               device),
                ("ema_alpha",            ema_alpha),
                ("use_half",             use_half),
                ("imgsz",                imgsz),
                ("min_stable_frames",    min_stable_frames),
                ("min_depth_m",          min_depth_m),
                ("max_depth_m",          max_depth_m),
                ("max_det",              max_det),
                ("log_detections",          log_detections),
                ("log_empty_throttle_sec",  log_empty_throttle_sec),
            ],
        ),
    ])
