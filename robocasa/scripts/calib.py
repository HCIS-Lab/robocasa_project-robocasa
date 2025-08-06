import os 
import glob
import numpy as np
import cv2


def main():
    dir_path = "/workspace/robocasa/robocasa/collect_demo/calib"
    img_list = sorted(glob.glob(dir_path + '/*.JPG'))

    # charuco setting
    squares_x  = 5
    squares_y  = 7
    square_len = 0.040
    marker_len = 0.028
    dict_id    = cv2.aruco.DICT_5X5_1000
    min_corners_keep = 12
    dictionary = cv2.aruco.getPredefinedDictionary(dict_id)
    board = cv2.aruco.CharucoBoard((squares_x, squares_y), square_len, marker_len, dictionary)
    detector = cv2.aruco.CharucoDetector(board)
    
    img_size = None
    all_obj, all_img = [], []
    for i, fn in enumerate(img_list):
        img = cv2.imread(fn, cv2.IMREAD_COLOR)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if img_size is None:
            img_size = gray.shape[::-1]
        ch_corners, ch_ids, mk_corners, mk_ids = detector.detectBoard(gray)
        n = 0 if ch_ids is None else int(np.asarray(ch_ids).size)
        if n < min_corners_keep:
            print(f"[skip] {os.path.basename(fn)} Lack of corners: {n}")
            continue
        ch_xy = np.asarray(ch_corners, dtype=np.float32).reshape(-1, 2)
        ch_id = np.asarray(ch_ids, dtype=np.int32).reshape(-1, 1)
        obj_pts, img_pts = board.matchImagePoints(ch_xy, ch_id)
        if obj_pts is None or img_pts is None or len(img_pts) < min_corners_keep:
            print(f"[skip] {os.path.basename(fn)} Matching image points fails")
            continue
        obj_pts = np.asarray(obj_pts, dtype=np.float32).reshape(-1, 3)
        img_pts = np.asarray(img_pts, dtype=np.float32).reshape(-1, 2)
        all_obj.append(obj_pts)
        all_img.append(img_pts)

    flags_pin = (cv2.CALIB_RATIONAL_MODEL | cv2.CALIB_TILTED_MODEL | cv2.CALIB_THIN_PRISM_MODEL)
    rms_pin, K_pin, D_pin, rvecs, tvecs = cv2.calibrateCamera(
        all_obj, all_img, img_size, None, None, flags=flags_pin
    )
    np.savez(dir_path + "/intrinsics.npz", K=K_pin, D=D_pin, rms=rms_pin, size=img_size)
    print(K_pin)


if __name__ == '__main__':
    main()