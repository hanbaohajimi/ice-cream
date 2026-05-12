from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description() -> LaunchDescription:
    # ── 话题名称 ────────────────────────────────────────────────────────────────
    # 默认使用压缩话题（image_transport），节省 Wi-Fi 带宽。
    # 传入 use_compressed:=false / use_compressed_depth:=false 可回退到
    # 原始 sensor_msgs/Image 话题。
    rgb_topic         = LaunchConfiguration("rgb_topic")
    depth_topic       = LaunchConfiguration("depth_topic")
    camera_info_topic = LaunchConfiguration("camera_info_topic")
    weights_path      = LaunchConfiguration("weights_path")
    pose_classes_path = LaunchConfiguration("pose_classes_path")
    enable_viz        = LaunchConfiguration("enable_viz")
    sample_radius     = LaunchConfiguration("sample_radius")
    sync_slop_sec     = LaunchConfiguration("sync_slop_sec")
    conf_threshold    = LaunchConfiguration("conf_threshold")
    device            = LaunchConfiguration("device")
    ema_alpha         = LaunchConfiguration("ema_alpha")
    use_half          = LaunchConfiguration("use_half")
    min_stable_frames = LaunchConfiguration("min_stable_frames")
    min_depth_m       = LaunchConfiguration("min_depth_m")
    max_depth_m       = LaunchConfiguration("max_depth_m")
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
    return LaunchDescription([
        # ── 话题参数 ──────────────────────────────────────────────────────────
        DeclareLaunchArgument(
            "rgb_topic",
            default_value="/ascamera_hp60c/camera_publisher/rgb0/image/compressed",
            description="RGB 压缩话题（image_transport）；use_compressed=false 时使用原始 Image",
        ),
        DeclareLaunchArgument(
            "depth_topic",
            default_value="/ascamera_hp60c/camera_publisher/depth0/image_raw/compressedDepth",
            description="深度 compressedDepth 话题；use_compressed_depth=false 时使用原始 Image",
        ),
        DeclareLaunchArgument(
            "camera_info_topic",
            default_value="/ascamera_hp60c/camera_publisher/rgb0/camera_info",
            description="RGB camera_info（不压缩，所有模式通用）",
        ),
        DeclareLaunchArgument(
            "use_compressed",
            default_value="true",
            description="RGB 订阅 CompressedImage（true）或原始 Image（false）",
        ),
        DeclareLaunchArgument(
            "use_compressed_depth",
            default_value="true",
            description="深度订阅 compressedDepth（true）或原始 Image（false）",
        ),
        # ── YOLO / 检测参数 ────────────────────────────────────────────────────
        DeclareLaunchArgument(
            "weights_path", default_value="",
            description="YOLO .pt 权重文件路径",
        ),
        DeclareLaunchArgument(
            "pose_classes_path", default_value="",
            description="可选的 pose_classes.yaml 路径，用于语义关键点角色映射",
        ),
        DeclareLaunchArgument(
            "enable_viz", default_value="true",
            description="是否启动叠加可视化节点",
        ),
        DeclareLaunchArgument(
            "conf_threshold", default_value="0.25",
            description="YOLO 检测置信度阈值",
        ),
        DeclareLaunchArgument(
            "device", default_value="cuda:0",
            description="推理设备：cuda:0 | cpu",
        ),
        DeclareLaunchArgument(
            "ema_alpha", default_value="0.4",
            description="角度 EMA 平滑系数（1.0=关闭，0.2=强平滑）",
        ),
        DeclareLaunchArgument(
            "use_half", default_value="True",
            description="在 CUDA 上使用 FP16 推理",
        ),
        DeclareLaunchArgument(
            "min_stable_frames", default_value="2",
            description="发布检测前所需的最少连续帧数",
        ),
        # ── 深度参数 ──────────────────────────────────────────────────────────
        DeclareLaunchArgument(
            "sample_radius", default_value="2",
            description="深度中位采样半径（像素）",
        ),
        DeclareLaunchArgument(
            "sync_slop_sec", default_value="0.08",
            description="ApproximateTimeSynchronizer 时间窗口（秒）",
        ),
        DeclareLaunchArgument(
            "min_depth_m", default_value="0.05",
            description="有效深度最小值（米）",
        ),
        DeclareLaunchArgument(
            "max_depth_m", default_value="4.0",
            description="有效深度最大值（米）",
        ),
        DeclareLaunchArgument(
            "max_det", default_value="50",
            description="NMS 后 YOLO 最大检测数（限制最坏情况下的 NMS 开销）",
        ),
        DeclareLaunchArgument(
            "log_detections", default_value="true",
            description="向终端输出检测 INFO 日志（设为 false 可节省 CPU）",
        ),
        DeclareLaunchArgument(
            "log_empty_throttle_sec", default_value="2.0",
            description="'Detection: None' 日志的最小间隔秒数（场景为空时）",
        ),
        DeclareLaunchArgument(
            "viz_display_scale", default_value="1.0",
            description="可视化缩放比例（<1 降低 imshow/发布开销，如 0.5）",
        ),
        DeclareLaunchArgument(
            "viz_rgb_stride", default_value="1",
            description="可视化每 N 帧解码一次（2 = 可视化 CPU 减半；检测不受影响）",
        ),
        DeclareLaunchArgument(
            "viz_opencv_threads", default_value="2",
            description="cv2.setNumThreads — 避免与 PyTorch 过度争抢资源",
        ),
        DeclareLaunchArgument(
            "viz_publish_image", default_value="true",
            description="发布 /object_overlay/image（较大的 bgr8）；设为 false 可节省 DDS CPU",
        ),
        DeclareLaunchArgument(
            "viz_show_window", default_value="true",
            description="显示 OpenCV 叠加窗口；GUI 模式下传 false 并使用 /object_overlay/image",
        ),
        # ── 节点 ──────────────────────────────────────────────────────────────
        Node(
            package="center_depth_pipeline",
            executable="detection_node",
            name="yolo_center_detection_node",
            parameters=[{
                "rgb_topic":                rgb_topic,
                "use_compressed":           use_compressed,
                "centers_topic":            "/object_centers_2d",
                "weights_path":             weights_path,
                "pose_classes_path":        pose_classes_path,
                "conf_threshold":           conf_threshold,
                "device":                   device,
                "ema_alpha":                ema_alpha,
                "use_half":                 use_half,
                "min_stable_frames":        ParameterValue(min_stable_frames, value_type=int),
                "max_det":                  ParameterValue(max_det, value_type=int),
                "log_detections":           ParameterValue(log_detections, value_type=bool),
                "log_empty_throttle_sec":   ParameterValue(log_empty_throttle_sec, value_type=float),
            }],
            output="screen",
        ),
        Node(
            package="center_depth_pipeline",
            executable="depth_node",
            name="center_depth_lookup_node",
            parameters=[{
                "depth_topic":              depth_topic,
                "use_compressed_depth":     use_compressed_depth,
                "camera_info_topic":        camera_info_topic,
                "centers_topic":            "/object_centers_2d",
                "centers_3d_topic":         "/object_centers_3d",
                "sample_radius":            sample_radius,
                "sync_slop_sec":            sync_slop_sec,
                "depth_aligned_to_rgb":     True,
                "min_depth_m":              min_depth_m,
                "max_depth_m":              max_depth_m,
            }],
            output="screen",
        ),
        Node(
            package="center_depth_pipeline",
            executable="visualization_node",
            name="center_depth_overlay_node",
            condition=IfCondition(enable_viz),
            parameters=[{
                "rgb_topic":            rgb_topic,
                "use_compressed":       use_compressed,
                "centers_3d_topic":     "/object_centers_3d",
                "viz_image_topic":      "/object_overlay/image",
                "show_window":          ParameterValue(viz_show_window, value_type=bool),
                "publish_image":        ParameterValue(viz_publish_image, value_type=bool),
                "display_scale":        ParameterValue(viz_display_scale, value_type=float),
                "rgb_process_stride":   ParameterValue(viz_rgb_stride, value_type=int),
                "opencv_num_threads":   ParameterValue(viz_opencv_threads, value_type=int),
            }],
            output="screen",
        ),
    ])
