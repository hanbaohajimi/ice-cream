"""
HP60C 专用包装 launch。
直接包含 yolo_center_depth.launch.py 并固定 HP60C 话题前缀。

压缩话题（默认开启，节省 Wi-Fi 带宽）：
  RGB:   /ascamera_hp60c/camera_publisher/rgb0/image/compressed
  Depth: /ascamera_hp60c/camera_publisher/depth0/image_raw/compressedDepth
  K:     /ascamera_hp60c/camera_publisher/rgb0/camera_info  （不压缩）

若相机侧未开启 image_transport 压缩插件，请传入：
  use_compressed:=false use_compressed_depth:=false

模型路径默认从仓库根 config.yaml 读取（model.weights_path / model.pose_classes_path）。
也可通过环境变量覆盖：
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


def _read_config_paths():
    """从仓库根 config.yaml 读取 model.weights_path 和 model.pose_classes_path。"""
    try:
        import yaml
        # launch 文件在 src/center_depth_pipeline/launch/，仓库根在上级 3 层
        here = Path(__file__).resolve().parent
        for _ in range(6):
            candidate = here / "config.yaml"
            if candidate.exists():
                with open(candidate, encoding="utf-8") as f:
                    cfg = yaml.safe_load(f) or {}
                model = cfg.get("model", {})
                weights = model.get("weights_path", "")
                classes = model.get("pose_classes_path", "")
                # 相对路径以 config.yaml 所在目录为基准
                if weights and not Path(weights).is_absolute():
                    weights = str(here / weights)
                if classes and not Path(classes).is_absolute():
                    classes = str(here / classes)
                return weights, classes
            here = here.parent
    except Exception:
        pass
    return "", ""


def generate_launch_description() -> LaunchDescription:
    pkg_share  = get_package_share_directory("center_depth_pipeline")
    main_launch = os.path.join(pkg_share, "launch", "yolo_center_depth.launch.py")

    _cfg_weights, _cfg_classes = _read_config_paths()
    default_weights = os.environ.get("ROBOT_WEIGHTS_PATH", _cfg_weights)
    default_pose_classes = os.environ.get("ROBOT_POSE_CLASSES", _cfg_classes)

    # ── declare all forwarded arguments ───────────────────────────────────────
    weights_path         = LaunchConfiguration("weights_path")
    pose_classes_path    = LaunchConfiguration("pose_classes_path")
    enable_viz           = LaunchConfiguration("enable_viz")
    sample_radius        = LaunchConfiguration("sample_radius")
    sync_slop_sec        = LaunchConfiguration("sync_slop_sec")
    conf_threshold       = LaunchConfiguration("conf_threshold")
    device               = LaunchConfiguration("device")
    ema_alpha            = LaunchConfiguration("ema_alpha")
    use_half             = LaunchConfiguration("use_half")
    min_stable_frames    = LaunchConfiguration("min_stable_frames")
    min_depth_m          = LaunchConfiguration("min_depth_m")
    max_depth_m          = LaunchConfiguration("max_depth_m")
    use_compressed       = LaunchConfiguration("use_compressed")
    use_compressed_depth = LaunchConfiguration("use_compressed_depth")
    max_det              = LaunchConfiguration("max_det")
    log_detections          = LaunchConfiguration("log_detections")
    log_empty_throttle_sec  = LaunchConfiguration("log_empty_throttle_sec")
    viz_display_scale    = LaunchConfiguration("viz_display_scale")
    viz_rgb_stride       = LaunchConfiguration("viz_rgb_stride")
    viz_opencv_threads   = LaunchConfiguration("viz_opencv_threads")
    viz_publish_image    = LaunchConfiguration("viz_publish_image")
    viz_show_window      = LaunchConfiguration("viz_show_window")
    _HP60C_RGB   = "/ascamera_hp60c/camera_publisher/rgb0/image/compressed"
    _HP60C_DEPTH = "/ascamera_hp60c/camera_publisher/depth0/image_raw/compressedDepth"
    _HP60C_INFO  = "/ascamera_hp60c/camera_publisher/rgb0/camera_info"

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
        DeclareLaunchArgument("enable_viz",           default_value="true"),
        DeclareLaunchArgument("sample_radius",        default_value="8"),
        DeclareLaunchArgument("sync_slop_sec",        default_value="0.12"),
        DeclareLaunchArgument("conf_threshold",       default_value="0.25"),
        DeclareLaunchArgument("device",               default_value="cuda:0"),
        DeclareLaunchArgument("ema_alpha",            default_value="0.4"),
        DeclareLaunchArgument("use_half",             default_value="True"),
        DeclareLaunchArgument("min_stable_frames",    default_value="2"),
        DeclareLaunchArgument("min_depth_m",          default_value="0.05"),
        DeclareLaunchArgument("max_depth_m",          default_value="4.0"),
        DeclareLaunchArgument(
            "use_compressed", default_value="true",
            description="true=CompressedImage，false=原始 Image（用于 RGB）"),
        DeclareLaunchArgument(
            "use_compressed_depth", default_value="true",
            description="true=compressedDepth，false=原始 Image（用于深度）"),
        DeclareLaunchArgument("max_det", default_value="50"),
        DeclareLaunchArgument("log_detections", default_value="true"),
        DeclareLaunchArgument("log_empty_throttle_sec", default_value="2.0"),
        DeclareLaunchArgument("viz_display_scale", default_value="1.0"),
        DeclareLaunchArgument("viz_rgb_stride", default_value="1"),
        DeclareLaunchArgument("viz_opencv_threads", default_value="2"),
        DeclareLaunchArgument("viz_publish_image", default_value="true"),
        DeclareLaunchArgument("viz_show_window", default_value="true"),
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
                ("enable_viz",           enable_viz),
                ("sample_radius",        sample_radius),
                ("sync_slop_sec",        sync_slop_sec),
                ("conf_threshold",       conf_threshold),
                ("device",               device),
                ("ema_alpha",            ema_alpha),
                ("use_half",             use_half),
                ("min_stable_frames",    min_stable_frames),
                ("min_depth_m",          min_depth_m),
                ("max_depth_m",          max_depth_m),
                ("max_det",              max_det),
                ("log_detections",          log_detections),
                ("log_empty_throttle_sec",  log_empty_throttle_sec),
                ("viz_display_scale",    viz_display_scale),
                ("viz_rgb_stride",       viz_rgb_stride),
                ("viz_opencv_threads",   viz_opencv_threads),
                ("viz_publish_image",    viz_publish_image),
                ("viz_show_window",      viz_show_window),
            ],
        ),
    ])
