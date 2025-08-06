import glob
import os
import cv2
from cv2 import aruco
import numpy as np
from scipy.spatial.transform import Rotation as R
import argparse


def detect_aruco_and_estimate_pose(frame, target_id=6, marker_size_m=0.018):
    aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
    parameters = aruco.DetectorParameters()
    data = np.load("/workspace/robocasa/robocasa/collect_demo/calib/intrinsics.npz", allow_pickle=True)
    K, D = data["K"], data["D"]
    detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)
    corners, ids, _ = detector.detectMarkers(frame)
    if ids is None:
        print("Not detect any ArUco")
        return None, None
    ids = ids.flatten()
    for i, id_val in enumerate(ids):
        if id_val != target_id:
            continue
        pts2D = corners[i].reshape(-1, 2).astype(np.float32)
        s = float(marker_size_m)
        pts3D = np.array([
            [-s/2,  s/2, 0], 
            [ s/2,  s/2, 0], 
            [ s/2, -s/2, 0],
            [-s/2, -s/2, 0], 
        ], dtype=np.float32)
        flag = getattr(cv2, "SOLVEPNP_IPPE_SQUARE", cv2.SOLVEPNP_ITERATIVE)
        ok, rvec, tvec = cv2.solvePnP(pts3D, pts2D, K, D, flags=flag)
        if not ok:
            ok, rvec, tvec = cv2.solvePnP(pts3D, pts2D, K, D, flags=cv2.SOLVEPNP_ITERATIVE)
        if not ok:
            print("PnP failed for marker", target_id)
            return None, None
        # if tvec[2] <= 0:
        #     print("Warning: tvec.z <= 0; check corner order / size.")
        cv2.aruco.drawDetectedMarkers(frame, [corners[i]])
        cv2.drawFrameAxes(frame, K, D, rvec, tvec, 0.02, 6)
        cv2.imshow("Result", resize_for_show(frame))
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return rvec, tvec
    print(f"Not detect ID {target_id}")
    return None, None

def resize_for_show(img, max_width=800):
    h, w = img.shape[:2]
    if w > max_width:
        scale = max_width / w
        new_size = (int(w * scale), int(h * scale))
        return cv2.resize(img, new_size, interpolation=cv2.INTER_AREA)
    return img

def obj_to_cam(rvec, tvec):
    R_obj2cam, _ = cv2.Rodrigues(rvec)
    T_obj2cam = np.eye(4)
    T_obj2cam[:3, :3] = R_obj2cam
    T_obj2cam[:3,  3] = tvec.flatten()
    return T_obj2cam

def cam_to_eef():
    # GoPro pose relative to parent
    p_cam = np.array([-0.00275, -0.0315, 0.086])
    q_cam_wxyz = np.array([0.0, 0.0, 0.707107, 0.707107])
    R_cam = R.from_quat([q_cam_wxyz[1], q_cam_wxyz[2], q_cam_wxyz[3], q_cam_wxyz[0]]).as_matrix()
    T_parent_cam = np.eye(4)
    T_parent_cam[:3,:3] = R_cam
    T_parent_cam[:3,3] = p_cam
    # EEF pose relative to parent
    p_eef = np.array([0.0, -0.18, 0.0])
    q_eef_wxyz = np.array([0.0, 0.707107, -0.707107, 0.0])
    R_eef = R.from_quat([q_eef_wxyz[1], q_eef_wxyz[2], q_eef_wxyz[3], q_eef_wxyz[0]]).as_matrix()
    T_parent_eef = np.eye(4)
    T_parent_eef[:3,:3] = R_eef
    T_parent_eef[:3,3] = p_eef
    # Compute T_cam2eef
    T_cam2eef = np.linalg.inv(T_parent_cam) @ T_parent_eef
    return T_cam2eef


if __name__ == "__main__":
    # Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path",
        type=str,
        default="/workspace/robocasa/robocasa/collect_demo",
    )
    parser.add_argument(
        "--obj",
        type=str,
        default="pnp_cheese",
        choices=["pnp_cheese", "pnp_cupcake"]
    )
    args = parser.parse_args()
    img_list = sorted(glob.glob(args.path + '/' + args.obj + '/*.JPG'))
    obj_id = {'pnp_cheese': 15, 'pnp_cupcake': 16}
    pose_num = 1
    for i in range(len(img_list)):
        print(img_list[i])
        img = cv2.imread(img_list[i])
        # object pose under gopro coordinate
        rvec, tvec = detect_aruco_and_estimate_pose(img, target_id=obj_id[args.obj])       
        if rvec is None or tvec is None:
            print("[skip] marker not found")
            continue
        # obtain predefined transform
        T_obj2cam = obj_to_cam(rvec, tvec)
        T_cam2eef = cam_to_eef()
        # OpenCV to Mujoco   
        S = np.eye(4); S[:3,:3] = np.diag([1, -1, -1])
        T_obj2eef = T_cam2eef @ (S @ T_obj2cam)
        # save grasping pose
        np.save(args.path + '/' + args.obj + '/' + 'T_obj2eef_{}.npy'.format(str(pose_num).zfill(2)), T_obj2eef)
        pose_num += 1