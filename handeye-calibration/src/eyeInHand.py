# coding=utf-8
"""
眼在手上
用采集到的图片信息和机械臂位姿信息计算相机坐标系相对于机械臂末端坐标系的旋转矩阵和平移向量

配置：优先读取同目录 config.yaml 中 calibration 节，CLI 参数可覆盖。
"""

import os.path
import cv2
import numpy as np
import csv
from pathlib import Path
np.set_printoptions(precision=8,suppress=True)


def _load_config():
    """向上查找 config.yaml，返回 (解析结果, config所在目录)；找不到则返回 (空字典, 脚本目录)。"""
    import yaml
    here = Path(__file__).resolve().parent
    for _ in range(4):
        candidate = here / "config.yaml"
        if candidate.exists():
            with open(candidate, encoding="utf-8") as f:
                return yaml.safe_load(f) or {}, here
        here = here.parent
    return {}, Path(__file__).resolve().parent


def euler_angles_to_rotation_matrix(rx, ry, rz):
    # 计算旋转矩阵
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(rx), -np.sin(rx)],
                   [0, np.sin(rx), np.cos(rx)]])
    Ry = np.array([[np.cos(ry), 0, np.sin(ry)],
                   [0, 1, 0],
                   [-np.sin(ry), 0, np.cos(ry)]])
    Rz = np.array([[np.cos(rz), -np.sin(rz), 0],
                   [np.sin(rz), np.cos(rz), 0],
                   [0, 0, 1]])
    R = Rx@Ry@Rz      #xyz

    return R


def pose_to_homogeneous_matrix(pose):
    x, y, z, rx, ry, rz = pose
    R = euler_angles_to_rotation_matrix(rx, ry, rz)
    t = np.array([x, y, z]).reshape(3, 1)
    H = np.eye(4)
    H[:3, :3] = R
    H[:3, 3] = t[:, 0]
    return H

def save_matrices_to_csv(matrices, file_name):
    rows, cols = matrices[0].shape
    num_matrices = len(matrices)
    combined_matrix = np.zeros((rows, cols * num_matrices))

    for i, matrix in enumerate(matrices):
        combined_matrix[:, i * cols: (i + 1) * cols] = matrix

    with open(file_name, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        for row in combined_matrix:
            csv_writer.writerow(row)

def poses_save_csv(filepath, csv_out):
    # 打开文本文件
    with open(filepath, "r", encoding="utf-8") as f:
        # 读取文件中的所有行
        lines = f.readlines()

    # 遍历每一行数据
    lines = [float(i) for line in lines for i in line.split(',')]

    matrices = []
    for i in range(0,len(lines),6):
        matrices.append(pose_to_homogeneous_matrix(lines[i:i+6]))
    # 将齐次变换矩阵列表存储到 CSV 文件中
    save_matrices_to_csv(matrices, csv_out)


def _rotation_matrix_to_axis_angle(R):
    """返回旋转向量（弧度），用于角度差计算。"""
    rvec, _ = cv2.Rodrigues(R)
    return rvec.flatten()


def evaluate_calibration(ret, mtx, dist, rvecs, tvecs, obj_points, img_points,
                          methods_results, image_size, T_ee2base_list=None):
    """
    计算所有标定评估指标，返回 dict。

    methods_results: list of (name, R, t)
    """
    lines = []
    passed = True

    # ── 1. 相机内参重投影误差 ──────────────────────────────────────────
    reproj_errors = []
    for objp, imgp, rvec, tvec in zip(obj_points, img_points, rvecs, tvecs):
        projected, _ = cv2.projectPoints(objp, rvec, tvec, mtx, dist)
        err = np.linalg.norm(imgp.reshape(-1, 2) - projected.reshape(-1, 2), axis=1)
        reproj_errors.extend(err.tolist())
    reproj_rms = float(np.sqrt(np.mean(np.array(reproj_errors) ** 2)))
    reproj_ok = reproj_rms < 0.5
    passed = passed and reproj_ok
    lines.append(f"[相机内参] 重投影 RMS = {reproj_rms:.4f} px  阈值<0.5px  {'PASS' if reproj_ok else 'FAIL'}")

    # ── 2. 多方法手眼结果一致性 ────────────────────────────────────────
    lines.append("")
    lines.append("[手眼标定多方法结果]")
    for name, R, t in methods_results:
        det = float(np.linalg.det(R))
        ortho_err = float(np.linalg.norm(R @ R.T - np.eye(3)))
        t_norm_mm = float(np.linalg.norm(t)) * 1000
        lines.append(f"  {name}:")
        lines.append(f"    旋转矩阵:\n{R}")
        lines.append(f"    平移向量: {t.flatten()} m  |t|={t_norm_mm:.1f} mm")
        lines.append(f"    det(R)={det:.6f}  正交误差={ortho_err:.2e}")
        det_ok = abs(det - 1.0) < 1e-4
        ortho_ok = ortho_err < 1e-4
        lines.append(f"    数学合法性: det {'PASS' if det_ok else 'FAIL'}  正交 {'PASS' if ortho_ok else 'FAIL'}")
        passed = passed and det_ok and ortho_ok

    # 方法间平移/旋转差异
    lines.append("")
    lines.append("[多方法一致性（两两对比）]")
    for i in range(len(methods_results)):
        for j in range(i + 1, len(methods_results)):
            n1, R1, t1 = methods_results[i]
            n2, R2, t2 = methods_results[j]
            t_diff_mm = float(np.linalg.norm(t1.flatten() - t2.flatten())) * 1000
            dR = R1 @ R2.T
            rvec_diff = _rotation_matrix_to_axis_angle(dR)
            rot_diff_deg = float(np.degrees(np.linalg.norm(rvec_diff)))
            t_ok = t_diff_mm < 5.0
            r_ok = rot_diff_deg < 1.0
            passed = passed and t_ok and r_ok
            lines.append(
                f"  {n1} vs {n2}:  平移差={t_diff_mm:.2f} mm {'PASS' if t_ok else 'FAIL'}(<5mm)"
                f"  旋转差={rot_diff_deg:.3f}° {'PASS' if r_ok else 'FAIL'}(<1°)"
            )

    # ── 3. 固定靶一致性（需要 capture_meta.jsonl）──────────────────────
    lines.append("")
    lines.append("[固定靶一致性]")
    if T_ee2base_list is not None:
        for name, R, t in methods_results:
            # 平移一致性
            rms_mm = _target_rms(R, t, T_ee2base_list, rvecs, tvecs) * 1000
            t_ok = rms_mm < 10.0
            # 旋转一致性：各帧标定板在基座系的朝向与均值旋转的差角 RMS
            N = min(len(T_ee2base_list), len(rvecs))
            rot_vecs = []
            for i in range(N):
                R_c, _ = cv2.Rodrigues(rvecs[i])
                R_base = T_ee2base_list[i][:3, :3] @ R @ R_c
                rot_vecs.append(R_base)
            # 均值旋转：用第一帧为参考，计算各帧与参考的差角
            R_ref = rot_vecs[0]
            rot_diffs_deg = []
            for R_b in rot_vecs:
                dR = R_ref @ R_b.T
                rvec_d = _rotation_matrix_to_axis_angle(dR)
                rot_diffs_deg.append(np.degrees(np.linalg.norm(rvec_d)))
            rot_rms_deg = float(np.sqrt(np.mean(np.array(rot_diffs_deg) ** 2)))
            r_ok = rot_rms_deg < 1.0
            passed = passed and t_ok and r_ok
            lines.append(
                f"  {name}: target_rms={rms_mm:.2f}mm {'PASS' if t_ok else 'FAIL'}(<10mm)"
                f"  rot_rms={rot_rms_deg:.3f}° {'PASS' if r_ok else 'FAIL'}(<1°)"
            )
    else:
        lines.append("  无 capture_meta.jsonl，跳过此项")

    # ── 4. 总结 ────────────────────────────────────────────────────────
    lines.append("")
    lines.append("=" * 56)
    lines.append(f"综合评定: {'[PASS] 可以部署' if passed else '[FAIL] 建议重新采集数据'}")
    lines.append("=" * 56)

    return "\n".join(lines), passed


def _load_meta(images_path):
    """读取 capture_meta.jsonl，按 index 排序返回 T_ee2base 列表（numpy 4×4）。"""
    import json
    meta_path = Path(images_path) / "capture_meta.jsonl"
    if not meta_path.exists():
        return None
    records = []
    with open(meta_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    records.sort(key=lambda r: r["index"])
    return [np.array(r["T_ee2base"]) for r in records]


def _target_rms(R_he, t_he, T_ee2base_list, rvecs, tvecs):
    """用固定靶一致性评估手眼矩阵质量，返回 RMS (米)。
    标定板原点在相机系的位置由 tvecs[i] 给出，经 T_cam2ee 再经 T_ee2base 变换到基座系，
    所有帧应收敛到同一点，散布越小越好。
    """
    N = min(len(T_ee2base_list), len(rvecs))
    T_cam2ee = np.eye(4)
    T_cam2ee[:3, :3] = R_he
    T_cam2ee[:3, 3] = t_he.flatten()
    pts = []
    for i in range(N):
        # 标定板原点在相机系的坐标（tvecs 单位：m，与 T_ee2base 一致）
        p_cam = np.array([*tvecs[i].flatten(), 1.0])
        p_base = T_ee2base_list[i] @ T_cam2ee @ p_cam
        pts.append(p_base[:3])
    pts = np.array(pts)
    mean_pt = pts.mean(axis=0)
    rms = float(np.sqrt(np.mean(np.sum((pts - mean_pt) ** 2, axis=1))))
    return rms



def compute_T(images_path, csv_path, corner_point_long, corner_point_short,
              corner_point_size, subpix_max_iter=30, subpix_epsilon=0.001,
              max_images=30):
    print("标定板的中长度对应的角点的个数", corner_point_long)
    print("标定板的中宽度对应的角点的个数", corner_point_short)
    print("标定板一格的长度", corner_point_size)

    # 设置寻找亚像素角点的参数
    criteria = (cv2.TERM_CRITERIA_MAX_ITER | cv2.TERM_CRITERIA_EPS,
                subpix_max_iter, subpix_epsilon)
    # 获取标定板角点的位置
    objp = np.zeros((corner_point_long * corner_point_short, 3), np.float32)
    objp[:, :2] = np.mgrid[0:corner_point_long, 0:corner_point_short].T.reshape(-1, 2)     
    # 将世界坐标系建在标定板上，所有点的Z坐标全部为0，所以只需要赋值x和y
    objp = corner_point_size*objp

    obj_points = []     # 存储3D点
    img_points = []     # 存储2D点


    for i in range(0, max_images):   #标定好的图片在images_path路径下，从0.jpg到x.jpg
        image = f"{images_path}/{i}.jpg"        #ubuntu下
        if os.path.exists(image):
            img = cv2.imread(image)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            size = gray.shape[::-1]
            ret, corners = cv2.findChessboardCorners(gray, (corner_point_long, corner_point_short), None)
            if ret:
                obj_points.append(objp)
                corners2 = cv2.cornerSubPix(gray, corners, (5, 5), (-1, -1), criteria)  # 在原角点的基础上寻找亚像素角点
                
                # 绘制检测到的角点
                cv2.drawChessboardCorners(img, (corner_point_long,corner_point_short), corners, ret)
                # 显示图片和提示信息
                cv2.imshow(f'Image {i}', img)
                cv2.putText(img, "Press ESC for next image", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)
                cv2.imshow(f'Image {i}', img)
                
                # 等待按键事件
                key = cv2.waitKey(0)
                if [corners2]:
                    img_points.append(corners2)
                else:
                    img_points.append(corners)
        cv2.destroyAllWindows()
    N = len(img_points)
    # 标定,得到图案在相机坐标系下的位姿
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, size, None, None)
    # print("ret:", ret)
    print("内参矩阵:\n", mtx) # 内参数矩阵
    print("畸变系数:\n", dist)  # 畸变系数   distortion cofficients = (k_1,k_2,p_1,p_2,k_3)
    print("-----------------------------------------------------")


    # 机器人末端在基坐标系下的位姿
    tool_pose = np.loadtxt(csv_path, delimiter=',')
    R_tool = []
    t_tool = []
    for i in range(int(N)):
        R_tool.append(tool_pose[0:3,4*i:4*i+3])
        t_tool.append(tool_pose[0:3,4*i+3])

    # 调用 cv2.calibrateHandEye 进行手眼标定 (方法: TSAI)
    method_tsai = cv2.CALIB_HAND_EYE_TSAI
    R_cam2gripper_tsai, t_cam2gripper_tsai = cv2.calibrateHandEye(
        R_tool, t_tool, 
        rvecs, tvecs, 
        method=method_tsai
    )
    # 输出 TSAI 方法的结果
    print("TSAI 方法计算的旋转矩阵:")
    print(R_cam2gripper_tsai)
    print("TSAI 方法计算的平移向量:")
    print(t_cam2gripper_tsai)

    # 调用 cv2.calibrateHandEye 进行手眼标定 (方法: PARK)
    method_park = cv2.CALIB_HAND_EYE_PARK
    R_cam2gripper_park, t_cam2gripper_park = cv2.calibrateHandEye(
        R_tool, t_tool, 
        rvecs, tvecs,  
        method=method_park
    )
    # 输出 PARK 方法的结果
    print("PARK 方法计算的旋转矩阵:")
    print(R_cam2gripper_park)
    print("PARK 方法计算的平移向量:")
    print(t_cam2gripper_park)

    # 调用 cv2.calibrateHandEye 进行手眼标定 (方法: HORAUD)
    method_HORAUD = cv2.CALIB_HAND_EYE_HORAUD
    R_cam2gripper_HORAUD, t_cam2gripper_HORAUD = cv2.calibrateHandEye(
        R_tool, t_tool, 
        rvecs, tvecs,  
        method=method_HORAUD
    )
    # 输出 HORAUD 方法的结果
    print("HORAUD 方法计算的旋转矩阵:")
    print(R_cam2gripper_HORAUD)
    print("HORAUD 方法计算的平移向量:")
    print(t_cam2gripper_HORAUD)    


    methods_results = [
        ("TSAI",   R_cam2gripper_tsai,   t_cam2gripper_tsai),
        ("PARK",   R_cam2gripper_park,   t_cam2gripper_park),
        ("HORAUD", R_cam2gripper_HORAUD, t_cam2gripper_HORAUD),
    ]
    # 选最优：用固定靶一致性（target_rms）；无 meta 数据时回退到平移一致性
    T_ee2base_list = _load_meta(images_path)
    if T_ee2base_list is not None:
        scores = [(name, _target_rms(R, t, T_ee2base_list, rvecs, tvecs)) for name, R, t in methods_results]
        for name, rms in scores:
            print(f"  {name} target_rms = {rms*1000:.2f} mm")
        best_name, _ = min(scores, key=lambda x: x[1])
    else:
        # 回退：平移向量两两差总和最小
        best_name = min(
            methods_results,
            key=lambda x: sum(
                np.linalg.norm(x[2].flatten() - y[2].flatten())
                for y in methods_results if y[0] != x[0]
            )
        )[0]
    best_R, best_t = next((R, t) for name, R, t in methods_results if name == best_name)
    print(f"\n最优方法: {best_name}")
    return best_R, best_t, \
           ret, mtx, dist, rvecs, tvecs, obj_points, img_points, size, methods_results, best_name



if __name__ == '__main__':
    cfg_raw, cfg_dir = _load_config()
    cfg = cfg_raw.get("calibration", {})

    images_path  = str(cfg_dir / cfg.get("images_dir", "images"))
    poses_file   = str(cfg_dir / cfg.get("poses_file", "images/poses.txt"))
    csv_out      = str(cfg_dir / cfg.get("robot_tool_pose_csv", "robotToolPose.csv"))
    corner_long  = cfg.get("board_corners_long", 11)
    corner_short = cfg.get("board_corners_short", 8)
    square_size  = cfg.get("board_square_size_m", 0.015)
    subpix_iter  = cfg.get("subpix_max_iter", 30)
    subpix_eps   = cfg.get("subpix_epsilon", 0.001)
    max_images   = cfg.get("max_images", 30)

    print("手眼标定采集的标定版图片所在路径", images_path)
    print("采集标定板图片时对应的机械臂末端的位姿", poses_file)
    poses_save_csv(poses_file, csv_out)
    (rotation_matrix, translation_vector,
     ret, mtx, dist, rvecs, tvecs, obj_points, img_points, img_size,
     methods_results, best_name) = compute_T(
        images_path, csv_out, corner_long, corner_short, square_size,
        subpix_max_iter=subpix_iter, subpix_epsilon=subpix_eps,
        max_images=max_images,
    )
    print(f'最优方法: {best_name}')
    print('rotation_matrix:')
    print(rotation_matrix)
    print('translation_vector:')
    print(translation_vector)

    # ── 自动写入 robot-vision-ros2/config.yaml T_cam2ee ──────────────
    import re
    ros2_config = cfg_dir.parent / "robot-vision-ros2" / "config.yaml"
    if ros2_config.exists():
        R, t = rotation_matrix, translation_vector.flatten()
        rows = [
            f"    - [{R[0,0]: .8f}, {R[0,1]: .8f}, {R[0,2]: .8f}, {t[0]: .8f}]",
            f"    - [{R[1,0]: .8f}, {R[1,1]: .8f}, {R[1,2]: .8f}, {t[1]: .8f}]",
            f"    - [{R[2,0]: .8f}, {R[2,1]: .8f}, {R[2,2]: .8f}, {t[2]: .8f}]",
            f"    - [ 0.0,          0.0,          0.0,          1.0         ]",
        ]
        new_block = "  T_cam2ee:\n" + "\n".join(rows) + "\n"
        text = ros2_config.read_text(encoding="utf-8")
        text = re.sub(
            r"  T_cam2ee:\n(?:    -[^\n]*\n){4}",
            new_block,
            text,
        )
        ros2_config.write_text(text, encoding="utf-8")
        print(f"\nT_cam2ee 已写入: {ros2_config}  (方法: {best_name})")
    else:
        print(f"\n警告: 未找到 {ros2_config}，跳过自动写入")

    # ── 写评估报告 ──────────────────────────────────────────────────────
    import datetime
    T_ee2base_list = _load_meta(images_path)
    report_str, passed = evaluate_calibration(
        ret, mtx, dist, rvecs, tvecs, obj_points, img_points,
        methods_results, img_size, T_ee2base_list,
    )
    header = (
        f"手眼标定评估报告\n"
        f"生成时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        f"数据集路径: {images_path}\n"
        + "=" * 56 + "\n"
    )
    full_report = header + report_str
    report_path = Path(images_path) / "calibration_eval.txt"
    report_path.write_text(full_report, encoding="utf-8")
    print(f"\n评估报告已写入: {report_path}")
    print(report_str)
