"""
A script to collect a batch of human demonstrations that can be used
to generate a learning curriculum (see `demo_learning_curriculum.py`).

The demonstrations can be played back using the `playback_demonstrations_from_pkl.py`
script.
"""

import argparse
from copy import deepcopy
import datetime
import json
import os
import time
from glob import glob

import h5py
import imageio
import mujoco
import numpy as np
import robosuite

# from robosuite import load_controller_config
from robosuite.controllers import load_composite_controller_config
from robosuite.wrappers import DataCollectionWrapper, VisualizationWrapper
from termcolor import colored

import robocasa
import robocasa.macros as macros
from robocasa.models.fixtures import FixtureType
from robocasa.utils.robomimic.robomimic_dataset_utils import convert_to_robomimic_format

from robosuite.utils.transform_utils import quat2axisangle, pose2mat, convert_quat, mat2quat, quat2mat, get_orientation_error
from mplib.pymp import Pose
import mimicgen.utils.pose_utils as PoseUtils
import robosuite.utils.transform_utils as T
from mplib import Planner
import xml.etree.ElementTree as ET
from scipy.spatial.transform import Rotation as R
from robosuite.utils.control_utils import orientation_error
from robosuite.devices.keyboard import Keyboard


def get_site_pose(env):
    ctrl = env.robots[0].composite_controller.get_controller("right")
    sid = env.sim.model.site_name2id(ctrl.ref_name)
    p = env.sim.data.site_xpos[sid].copy()
    R = env.sim.data.site_xmat[sid].reshape(3,3).copy()
    q_xyzw = T.mat2quat(R)
    return p, R, q_xyzw

def quat_slerp_xyzw(q0, q1, s):
    q0 = q0 / np.linalg.norm(q0)
    q1 = q1 / np.linalg.norm(q1)
    dot = float(np.dot(q0, q1))
    if dot < 0.0:
        q1 = -q1
        dot = -dot
    if dot > 0.9995:
        q = q0 + s*(q1 - q0)
        return q / np.linalg.norm(q)
    theta0 = np.arccos(np.clip(dot, -1.0, 1.0))
    sin0 = np.sin(theta0)
    s0 = np.sin((1.0 - s) * theta0) / sin0
    s1 = np.sin(s * theta0) / sin0
    return s0*q0 + s1*q1

def plan_line_cartesian(env, p_goal_w, q_goal_xyzw=None, step_m=None):
    p0, R0, q0_xyzw = get_site_pose(env)
    if q_goal_xyzw is None:
        q_goal_xyzw = q0_xyzw
    ctrl = env.robots[0].composite_controller.get_controller("right")
    max_dpos = np.array(ctrl.output_max[:3])
    if step_m is None:
        step_m = 0.2 * float(np.min(max_dpos))
    dist = float(np.linalg.norm(p_goal_w - p0))
    n = max(2, int(np.ceil(dist / step_m)))
    waypoints = []
    for i in range(1, n+1):
        s = i / n
        p = (1.0 - s) * p0 + s * p_goal_w
        q_xyzw = quat_slerp_xyzw(q0_xyzw, q_goal_xyzw, s)
        waypoint = np.concatenate([p, q_xyzw])
        waypoints.append(waypoint)
    return waypoints

def target_pose_to_action(env, target_pose, relative=True):
    target_pos = target_pose[:3]
    target_q   = target_pose[3:]
    ctrl = env.robots[0].composite_controller.get_controller("right")
    sid  = env.sim.model.site_name2id(ctrl.ref_name)
    curr_pos = env.sim.data.site_xpos[sid].copy()
    curr_R   = env.sim.data.site_xmat[sid].reshape(3, 3).copy()
    dpos_world = target_pos - curr_pos
    R_err_w  = T.quat2mat(target_q) @ curr_R.T
    da_world = T.quat2axisangle(T.mat2quat(R_err_w))
    kind = getattr(ctrl, "name", type(ctrl).__name__).upper()
    if "IK" in kind:
        dp = dpos_world
        dr = curr_R.T @ da_world 
    else:
        ref = getattr(ctrl, "input_ref_frame", "base")
        if ref == "world":
            dp, dr = dpos_world, da_world
        else:  # "base"
            base_pos, base_R = env.robots[0].composite_controller.get_controller_base_pose("right")
            dp = base_R.T @ dpos_world
            dr = base_R.T @ da_world
    pos_max = np.array(ctrl.output_max[:3])
    rot_max = np.array(ctrl.output_max[3:6])
    return np.concatenate([np.clip(dp/pos_max, -1, 1), np.clip(dr/rot_max, -1, 1)])

def move_to_pose(env, p_goal, q_goal_xyzw=None, gripper=1.0, hold_time=0, step_m=None, print_debug=True, stage=None):
    # control end-effector
    if stage=='grasp':
        pos_var = 0.03  # up to Â±3cm variation
        random_offset = np.random.uniform(-pos_var, pos_var, size=3)
        p_goal = p_goal + random_offset
        if q_goal_xyzw is not None:
            # small random axis-angle perturbation
            rot_var = np.deg2rad(5)  # e.g., up to Â±5 deg
            random_axis = np.random.randn(3)
            random_axis = random_axis / np.linalg.norm(random_axis)
            random_angle = np.random.uniform(-rot_var, rot_var)
            from scipy.spatial.transform import Rotation as R
            random_rot = R.from_rotvec(random_axis * random_angle)
            q_goal_xyzw = (random_rot * R.from_quat(q_goal_xyzw)).as_quat()
    if p_goal[2]>=1.3:  # avoid collision to cabinent
        p_goal[2] = 1.3
    if p_goal[2] <= 0.98: # avoid collision to table
        p_goal[2] = 0.98
    cart_trajectory = plan_line_cartesian(env, p_goal, q_goal_xyzw, step_m)
    for i, target_pose in enumerate(cart_trajectory):
        norm_delta = target_pose_to_action(env, target_pose)
        action_dict = {
            "right": norm_delta,
            "right_gripper": np.array([gripper], dtype=np.float32),
            "base": np.zeros(3, dtype=np.float32),
            "torso": np.zeros(1, dtype=np.float32),
        }
        action = env.robots[0].create_action_vector(action_dict)
        env.step(action)
        env.render()
        if print_debug:
            p, _, q_xyzw = get_site_pose(env)
    if hold_time > 0:
        time.sleep(hold_time)

def move_gripper(env, target_pose, gripper, repeat=30):
    #control gripper
    for _ in range(repeat):
        action_dict = {
            "right": target_pose_to_action(env, target_pose),
            "right_gripper": np.array([gripper], dtype=np.float32),
            "base": np.zeros(3, dtype=np.float32),
            "torso": np.zeros(1, dtype=np.float32),
        }
        action = env.robots[0].create_action_vector(action_dict)
        env.step(action)
        env.render()

def gather_demonstrations_as_hdf5(directory, out_dir, env_info, excluded_episodes=None):
    """
    Gathers the demonstrations saved in @directory into a
    single hdf5 file.
    The strucure of the hdf5 file is as follows.
    data (group)
        date (attribute) - date of collection
        time (attribute) - time of collection
        repository_version (attribute) - repository version used during collection
        env (attribute) - environment name on which demos were collected
        demo1 (group) - every demonstration has a group
            model_file (attribute) - model xml string for demonstration
            states (dataset) - flattened mujoco states
            actions (dataset) - actions applied during demonstration
        demo2 (group)
        ...
    Args:
        directory (str): Path to the directory containing raw demonstrations.
        out_dir (str): Path to where to store the hdf5 file.
        env_info (str): JSON-encoded string containing environment information,
            including controller and robot info
    """

    hdf5_path = os.path.join(out_dir, "demo.hdf5")
    print("Saving hdf5 to", hdf5_path)
    f = h5py.File(hdf5_path, "w")

    # store some metadata in the attributes of one group
    grp = f.create_group("data")

    num_eps = 0
    env_name = None  # will get populated at some point

    for ep_directory in os.listdir(directory):
        # print("Processing {} ...".format(ep_directory))
        if (excluded_episodes is not None) and (ep_directory in excluded_episodes):
            # print("\tExcluding this episode!")
            continue

        state_paths = os.path.join(directory, ep_directory, "state_*.npz")
        states = []
        actions = []
        actions_abs = []
        # success = False

        for state_file in sorted(glob(state_paths)):
            dic = np.load(state_file, allow_pickle=True)
            env_name = str(dic["env"])

            states.extend(dic["states"])
            for ai in dic["action_infos"]:
                actions.append(ai["actions"])
                if "actions_abs" in ai:
                    actions_abs.append(ai["actions_abs"])
            # success = success or dic["successful"]

        if len(states) == 0:
            continue

        # # Add only the successful demonstration to dataset
        # if success:

        # print("Demonstration is successful and has been saved")
        # Delete the last state. This is because when the DataCollector wrapper
        # recorded the states and actions, the states were recorded AFTER playing that action,
        # so we end up with an extra state at the end.
        del states[-1]
        assert len(states) == len(actions)

        num_eps += 1
        ep_data_grp = grp.create_group("demo_{}".format(num_eps))

        # store model xml as an attribute
        xml_path = os.path.join(directory, ep_directory, "model.xml")
        with open(xml_path, "r") as f:
            xml_str = f.read()
        ep_data_grp.attrs["model_file"] = xml_str

        # store ep meta as an attribute
        ep_meta_path = os.path.join(directory, ep_directory, "ep_meta.json")
        if os.path.exists(ep_meta_path):
            with open(ep_meta_path, "r") as f:
                ep_meta = f.read()
            ep_data_grp.attrs["ep_meta"] = ep_meta

        # write datasets for states and actions
        ep_data_grp.create_dataset("states", data=np.array(states))
        ep_data_grp.create_dataset("actions", data=np.array(actions))
        if len(actions_abs) > 0:
            print(np.array(actions_abs).shape)
            ep_data_grp.create_dataset("actions_abs", data=np.array(actions_abs))

        # else:
        #     pass
        #     # print("Demonstration is unsuccessful and has NOT been saved")

    print("{} successful demos so far".format(num_eps))

    if num_eps == 0:
        f.close()
        return

    # write dataset attributes (metadata)
    now = datetime.datetime.now()
    grp.attrs["date"] = "{}-{}-{}".format(now.month, now.day, now.year)
    grp.attrs["time"] = "{}:{}:{}".format(now.hour, now.minute, now.second)
    grp.attrs["robocasa_version"] = robocasa.__version__
    grp.attrs["robosuite_version"] = robosuite.__version__
    grp.attrs["mujoco_version"] = mujoco.__version__
    grp.attrs["env"] = env_name
    grp.attrs["env_info"] = env_info

    f.close()

    return hdf5_path


if __name__ == "__main__":
    # Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--directory",
        type=str,
        default=os.path.join(robocasa.models.assets_root, "demonstrations_private"),
    )
    parser.add_argument("--environment", type=str, default="Kitchen")
    parser.add_argument(
        "--robots",
        nargs="+",
        type=str,
        default="PandaOmron",
        help="Which robot(s) to use in the env",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="single-arm-opposed",
        help="Specified environment configuration if necessary",
    )
    parser.add_argument(
        "--arm",
        type=str,
        default="right",
        help="Which arm to control (eg bimanual) 'right' or 'left'",
    )
    parser.add_argument(
        "--obj_groups",
        type=str,
        nargs="+",
        # default="brown_cuboid",
        default=None,
        help="In kitchen environments, either the name of a group to sample object from or path to an .xml file",
    )
    parser.add_argument(
        "--camera",
        type=str,
        default=None,
        help="Which camera to use for collecting demos",
    )
    parser.add_argument(
        "--controller",
        type=str,
        default=None,
        help="Choice of controller. Can be, eg. 'NONE' or 'WHOLE_BODY_IK', etc. Or path to controller json file",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="keyboard",
        choices=["keyboard", "keyboardmobile", "spacemouse", "dummy"],
    )
    parser.add_argument(
        "--pos-sensitivity",
        type=float,
        default=4.0,
        help="How much to scale position user inputs",
    )
    parser.add_argument(
        "--rot-sensitivity",
        type=float,
        default=4.0,
        help="How much to scale rotation user inputs",
    )
    parser.add_argument("--debug", action="store_true")
    parser.add_argument(
        "--renderer", type=str, default="mjviewer", choices=["mjviewer", "mujoco"]
    )
    parser.add_argument(
        "--max_fr", default=30, type=int, help="If specified, limit the frame rate"
    )
    parser.add_argument("--layout", type=int, nargs="+", default=0)
    parser.add_argument(
        "--style", type=int, nargs="+", default=[0, 1, 2, 3, 4, 5, 6, 7, 8, 11]
    )
    parser.add_argument("--generative_textures", action="store_true")
    parser.add_argument("--num_episodes", type=int, default=20)
    args = parser.parse_args()

    # Get controller config
    # controller_config = load_controller_config(default_controller=args.controller)
    controller_config = load_composite_controller_config(
        controller=args.controller,
        robot=args.robots if isinstance(args.robots, str) else args.robots[0],
    )
    controller_config["body_parts"]["right"]["kp"] = 400
    controller_config["body_parts"]["right"]["damping_ratio"] =  0.95 

    if controller_config["type"] == "WHOLE_BODY_MINK_IK":
        # mink-speicific import. requires installing mink
        from robosuite.examples.third_party_controller.mink_controller import (
            WholeBodyMinkIK,
        )

    env_name = args.environment

    # Create argument configuration
    config = {
        "env_name": env_name,
        "robots": args.robots,
        "controller_configs": controller_config,
    }

    if args.generative_textures is True:
        config["generative_textures"] = "100p"

    # Check if we're using a multi-armed environment and use env_configuration argument if so
    if "TwoArm" in env_name:
        config["env_configuration"] = args.config

    # Mirror actions if using a kitchen environment
    if env_name in ["Lift"]:  # add other non-kitchen tasks here
        if args.obj_groups is not None:
            print(
                "Specifying 'obj_groups' in non-kitchen environment does not have an effect."
            )
        mirror_actions = False
        if args.camera is None:
            args.camera = "agentview"
        # special logic: "free" camera corresponds to Null camera
        elif args.camera == "free":
            args.camera = None
    else:
        mirror_actions = True
        config["layout_ids"] = args.layout
        config["style_ids"] = args.style
        ### update config for kitchen envs ###
        if args.obj_groups is not None:
            config.update({"obj_groups": args.obj_groups})
        if args.camera is None:
            args.camera = "robot0_frontview"
        # special logic: "free" camera corresponds to Null camera
        elif args.camera == "free":
            args.camera = None

        config["translucent_robot"] = True

        # by default use obj instance split A
        config["obj_instance_split"] = "A"

    # Create environment
    env = robosuite.make(
        **config,
        has_renderer=True,
        has_offscreen_renderer=False,
        render_camera=args.camera,
        ignore_done=True,
        use_camera_obs=False,
        control_freq=20,
        renderer=args.renderer,
        #allowed_instance=args.allowed_instance,
    )

    # Wrap this with visualization wrapper
    env = VisualizationWrapper(env)

    # Grab reference to controller config and convert it to json-encoded string
    env_info = json.dumps(config)

    t_now = time.time()
    time_str = datetime.datetime.fromtimestamp(t_now).strftime("%Y-%m-%d-%H-%M-%S")

    if not args.debug:
        # wrap the environment with data collection wrapper
        tmp_directory = "/tmp/{}".format(time_str)
        env = DataCollectionWrapper(env, tmp_directory)

    # make a new timestamped directory
    new_dir = os.path.join(args.directory, time_str)
    os.makedirs(new_dir)

    transform_dir = "/workspace/robocasa/robocasa/collect_demo/{}".format(args.obj_groups[0])
    T_file_list = sorted(glob(os.path.join(transform_dir, "T_obj2eef_*.npy")))
    excluded_eps = []

    for T_files in T_file_list:
        T_obj2eef  = np.load(T_files)
        for ep in range(args.num_episodes):
            print(f"\n=== Auto demo {ep+1} ===")
            success = False
            task_completion_hold_count = -1 

            # auto-generate data
            obs = env.reset()
            model = env.sim.model
            data  = env.sim.data
            robot = env.robots[0]
            ctrl = env.robots[0].composite_controller.get_controller("right")
            ctrl.set_goal_update_mode("desired")
            site_id = env.sim.model.site_name2id(ctrl.ref_name)
            site_z_id = env.sim.model.site_name2id("gripper0_right_ee_z")
            obj_id = env.sim.model.site_name2id("obj_default_site")
            sink_id = env.sim.model.site_name2id("sink_main_group_water")

            # offset
            p_grip = env.sim.data.site_xpos[site_id].copy()
            R_grip = env.sim.data.site_xmat[site_id].reshape(3,3).copy()
            p_ee_z = env.sim.data.site_xpos[site_z_id].copy()
            t_world = p_ee_z - p_grip
            t_local = R_grip.T @ t_world
            T_off_local = np.eye(4)
            T_off_local[:3,3] = t_local

            # pre-steps
            env.render()
            zero_action = np.zeros(env.action_dim)
            for _ in range(1):
                # do a dummy step thru base env to initalize things, but don't record the step
                if isinstance(env, DataCollectionWrapper):
                    env.env.step(zero_action)
                else:
                    env.step(zero_action)
            time.sleep(2)

            # STEP 1: Move above object
            obj_pos = env.sim.data.site_xpos[obj_id]
            p_goal = obj_pos + np.array([-0.04, 0.05, 0.15])
            _, _, q_now_xyzw = get_site_pose(env)
            move_to_pose(env, p_goal, q_now_xyzw, gripper=1.0, hold_time=1)

            # STEP 2: Move to grasp pose (calculate pose as before)
            obj_mat = np.eye(4)
            obj_mat[:3,:3] = env.sim.data.site_xmat[obj_id].reshape(3,3)
            obj_mat[:3, 3] = env.sim.data.site_xpos[obj_id]
            g_nominal = obj_mat @ T_obj2eef
            p_nominal = g_nominal[:3,3]
            obj_R = obj_mat[:3,:3]
            ez = obj_R[:,2]
            z_des = -ez
            x_ref = obj_R[:,0] if abs(z_des.dot(obj_R[:,0]))<0.95 else obj_R[:,1]
            y_des = np.cross(z_des, x_ref);  y_des /= np.linalg.norm(y_des)
            x_des = np.cross(y_des, z_des)
            R_top = np.column_stack([x_des, y_des, z_des])
            def Rz(deg):
                theta = np.deg2rad(deg)
                return np.array([[np.cos(theta), -np.sin(theta), 0],
                                [np.sin(theta),  np.cos(theta), 0],
                                [0,           0,        1]])
            R_goal = R_top @ Rz(-90)
            g_goal = np.eye(4)
            g_goal[:3,:3] = R_goal
            g_goal[:3, 3] = p_nominal
            g_goal_tip = g_goal @ T_off_local
            p_goal, q_goal_xyzw = T.mat2pose(g_goal_tip)
            # Apply extra adjustment
            pl = env.sim.data.site_xpos[model.site_name2id("gripper0_right_ee_x")]
            pr = env.sim.data.site_xpos[model.site_name2id("gripper0_right_ee_y")]
            mid_xy = (pl[:2] + pr[:2]) / 2
            p_goal[:2] = mid_xy
            obj_z = env.sim.data.site_xpos[obj_id][2]
            # =============================================================    
            # magic number
            p_goal[1] -= 0.07 
            p_goal[2] = obj_z - 0.01 # -0.01 for cheese and for -0.003 cupcake
            # =============================================================
            move_to_pose(env, p_goal, q_goal_xyzw, gripper=1.0, hold_time=0, stage='grasp')

            # STEP 3: Close gripper
            move_gripper(env, np.concatenate([p_goal, q_goal_xyzw]), gripper=1.0, repeat=30)   # hold open
            move_gripper(env, np.concatenate([p_goal, q_goal_xyzw]), gripper=-1.0, repeat=50)  # close

            # STEP 4: Lift up
            p, _, q_now_xyzw = get_site_pose(env)
            p_goal = p + np.array([0.0, 0.0, 0.8])
            move_to_pose(env, p_goal, q_now_xyzw, gripper=-1.0, hold_time=1)

            # STEP 5: Move to sink
            p, _, q_now_xyzw = get_site_pose(env)
            sink_pos = env.sim.data.site_xpos[sink_id].copy()
            p_goal = np.array([sink_pos[0]+0.1, sink_pos[1], p[2]-0.1])
            move_to_pose(env, p_goal, q_now_xyzw, gripper=-1.0, hold_time=1)

            # STEP 6: Open gripper (release)
            move_gripper(env, np.concatenate([p_goal, q_now_xyzw]), gripper=-1.0, repeat=30)
            move_gripper(env, np.concatenate([p_goal, q_now_xyzw]), gripper=1.0, repeat=50)
            time.sleep(1)

            if hasattr(env, "_check_success") and callable(env._check_success):
                success = env._check_success()
            else:
                print("Warning: env._check_success() is not implemented!")

            ep_directory = getattr(env, "ep_directory", None)
            if success:
                if ep_directory is not None:
                    print(f"Saved successful demo: {ep_directory}")
                continue  # Only keep the first successful attempt per episode
            else:
                if ep_directory is not None:
                    print(f"Excluding failed demo: {ep_directory}")
                    excluded_eps.append(ep_directory.split("/")[-1])

            # ends
            env.close()

            if not args.debug:
                hdf5_path = gather_demonstrations_as_hdf5(
                    tmp_directory, new_dir, env_info, excluded_episodes=excluded_eps
                )
                if hdf5_path is not None:
                    convert_to_robomimic_format(hdf5_path)
                    print(f"Robomimic dataset written to: {hdf5_path}")
                else:
                    print("No successful demos, skipping conversion.")
