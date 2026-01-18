import numpy as np
from gym import utils
from mjrl.envs import mujoco_env
from mujoco_py import MjViewer
from mj_envs.utils.quatmath import *
import os

ADD_BONUS_REWARDS = True
USE_SPARSE_REWARDS = True

class HammerEnvV0(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self, num_distraction_nails=0):
        self.target_obj_sid = -1
        self.S_grasp_sid = -1
        self.obj_bid = -1
        self.tool_sid = -1
        self.goal_sid = -1
        self.num_distraction_nails = num_distraction_nails

        curr_dir = os.path.dirname(os.path.abspath(__file__))
        mujoco_env.MujocoEnv.__init__(self, curr_dir+'/assets/DAPG_hammer.xml', 5)
        utils.EzPickle.__init__(self)

        # change actuator sensitivity
        self.sim.model.actuator_gainprm[self.sim.model.actuator_name2id('A_WRJ1'):self.sim.model.actuator_name2id('A_WRJ0')+1,:3] = np.array([10, 0, 0])
        self.sim.model.actuator_gainprm[self.sim.model.actuator_name2id('A_FFJ3'):self.sim.model.actuator_name2id('A_THJ0')+1,:3] = np.array([1, 0, 0])
        self.sim.model.actuator_biasprm[self.sim.model.actuator_name2id('A_WRJ1'):self.sim.model.actuator_name2id('A_WRJ0')+1,:3] = np.array([0, -10, 0])
        self.sim.model.actuator_biasprm[self.sim.model.actuator_name2id('A_FFJ3'):self.sim.model.actuator_name2id('A_THJ0')+1,:3] = np.array([0, -1, 0])

        self.target_obj_sid = self.sim.model.site_name2id('S_target')
        self.S_grasp_sid = self.sim.model.site_name2id('S_grasp')
        self.obj_bid = self.sim.model.body_name2id('Object')
        self.tool_sid = self.sim.model.site_name2id('tool')
        self.goal_sid = self.sim.model.site_name2id('nail_goal')

        # Setup distraction nails based on num_distraction_nails
        self._setup_distraction_nails()

        self.act_mid = np.mean(self.model.actuator_ctrlrange, axis=1)
        self.act_rng = 0.5 * (self.model.actuator_ctrlrange[:, 1] - self.model.actuator_ctrlrange[:, 0])
        self.action_space.high = np.ones_like(self.model.actuator_ctrlrange[:,1])
        self.action_space.low  = -1.0 * np.ones_like(self.model.actuator_ctrlrange[:,0])

        # Setup distraction nails based on num_distraction_nails
        self._setup_distraction_nails()

    def _setup_distraction_nails(self):
        """Setup distraction nails based on num_distraction_nails parameter"""
        if self.num_distraction_nails > 0:
            # Define positions for distraction nails (relative to nail board center)
            # XY offsets; z will be computed relative to board_pos + z_offset
            # increase offsets to spread nails further away from the target
            positions = [
                [0.25, 0.25],   # Position for nail_1 (xy)
                [-0.25, 0.25],  # Position for nail_2
                [0.25, -0.25],  # Position for nail_3
                [-0.25, -0.25], # Position for nail_4
                [0.15, 0.15],   # Position for nail_5 (closer positions)
                [-0.15, 0.15],  # Position for nail_6
                [0.15, -0.15],  # Position for nail_7
                [-0.15, -0.15], # Position for nail_8
            ]

            # Get nail board position as reference (use sim.model to affect current sim state)
            try:
                board_bid = self.sim.model.body_name2id('nail_board')
                board_pos = self.sim.model.body_pos[board_bid].copy()
            except Exception:
                board_pos = self.model.body_pos[self.model.body_name2id('nail_board')].copy()

            # Debug logging to help diagnose why nails may remain underground.
            try:
                with open('/tmp/nail_setup.log', 'a') as flog:
                    flog.write('=== _setup_distraction_nails called ===\\n')
                    flog.write('num_distraction_nails=%d\\n' % self.num_distraction_nails)
                    flog.write('board_pos=%s\\n' % (repr(board_pos),))
            except Exception:
                pass

            # Activate the requested number of nails (move via sim.model.body_pos)
            # choose a small positive z offset so nails sit slightly above the board surface
            z_offset = 0.02
            # compute quaternion that aligns local +Z to board normal
            def rotate_vec_by_quat(q, v):
                # q: [w, x, y, z], v: (3,)
                qw, qx, qy, qz = q
                q_vec = np.array([qx, qy, qz])
                t = 2.0 * np.cross(q_vec, v)
                return v + qw * t + np.cross(q_vec, t)

            def axis_angle_to_quat(axis, angle):
                axis = axis / (np.linalg.norm(axis) + 1e-12)
                qw = np.cos(angle / 2.0)
                qxyz = axis * np.sin(angle / 2.0)
                return np.array([qw, qxyz[0], qxyz[1], qxyz[2]])

            try:
                board_bid = self.sim.model.body_name2id('nail_board')
                board_quat = self.sim.model.body_quat[board_bid].copy()
            except Exception:
                board_quat = None
            for i in range(min(self.num_distraction_nails, 8)):
                try:
                    body_name = f'nail_{i+1}'
                    body_id = self.sim.model.body_name2id(body_name)
                    xy = positions[i]
                    target_pos = board_pos + np.array([xy[0], xy[1], z_offset])
                    # Debug before/after
                    try:
                        with open('/tmp/nail_setup.log', 'a') as flog:
                            flog.write('moving %s (id=%d) to %s\\n' % (body_name, int(body_id), repr(target_pos)))
                    except Exception:
                        pass
                    # Move to surface position relative to board
                    self.sim.model.body_pos[body_id] = target_pos
                    # read back current position
                    cur = None
                    try:
                        cur = self.sim.model.body_pos[body_id].copy()
                    except Exception:
                        cur = None
                    # set body orientation to match nail_board orientation (so nails are perpendicular to board)
                    try:
                        if board_quat is not None:
                            self.sim.model.body_quat[body_id] = board_quat.copy()
                    except Exception:
                        pass
                    # log the assigned position
                    try:
                        with open('/tmp/nail_setup.log', 'a') as flog:
                            flog.write(' after assign, body_pos=%s\\n' % (repr(cur),))
                    except Exception:
                        pass
                except Exception:
                    # Body not found or other error, skip
                    try:
                        with open('/tmp/nail_setup.log', 'a') as flog:
                            flog.write(' failed to move nail index %d\\n' % (i+1,))
                    except Exception:
                        pass
                    pass

            # Ensure simulator state updated
            try:
                self.sim.forward()
            except Exception:
                pass

    def get_distraction_nails(self):
        """Return list of active distraction nail indices (1..8) whose z > 0 (above ground)"""
        active = []
        for i in range(1, 9):
            try:
                bid = self.sim.model.body_name2id(f'nail_{i}')
                z = float(self.sim.model.body_pos[bid][2])
                if z > 0.0:
                    active.append(i)
            except Exception:
                pass
        return active

    def step(self, a):
        a = np.clip(a, -1.0, 1.0)
        try:
            a = self.act_mid + a * self.act_rng  # mean center and scale
        except:
            a = a  # only for the initialization phase
        self.do_simulation(a, self.frame_skip)
        ob = self.get_obs()
        obj_pos = self.data.body_xpos[self.obj_bid].ravel()
        palm_pos = self.data.site_xpos[self.S_grasp_sid].ravel()
        tool_pos = self.data.site_xpos[self.tool_sid].ravel()
        target_pos = self.data.site_xpos[self.target_obj_sid].ravel()
        goal_pos = self.data.site_xpos[self.goal_sid].ravel()
        
        # get to hammer
        reward = - 0.1 * np.linalg.norm(palm_pos - obj_pos)
        # take hammer head to nail
        reward -= np.linalg.norm((tool_pos - target_pos))
        # make nail go inside
        reward -= 10 * np.linalg.norm(target_pos - goal_pos)
        # velocity penalty
        reward -= 1e-2 * np.linalg.norm(self.data.qvel.ravel())

        if ADD_BONUS_REWARDS:
            # bonus for lifting up the hammer
            if obj_pos[2] > 0.04 and tool_pos[2] > 0.04:
                reward += 2

            # bonus for hammering the nail
            if (np.linalg.norm(target_pos - goal_pos) < 0.020):
                reward += 25
            if (np.linalg.norm(target_pos - goal_pos) < 0.010):
                reward += 75

        if USE_SPARSE_REWARDS:
            reward = -10 * np.linalg.norm(target_pos - goal_pos)
            if (np.linalg.norm(target_pos - goal_pos) < 0.020):
                reward += 25
            if (np.linalg.norm(target_pos - goal_pos) < 0.010):
                reward += 75            

        goal_achieved = True if np.linalg.norm(target_pos - goal_pos) < 0.010 else False

        return ob, reward, False, dict(goal_achieved=goal_achieved)

    def get_obs(self):
        # qpos for hand
        # xpos for obj
        # xpos for target
        qp = self.data.qpos.ravel()
        qv = np.clip(self.data.qvel.ravel(), -1.0, 1.0)
        obj_pos = self.data.body_xpos[self.obj_bid].ravel()
        obj_rot = quat2euler(self.data.body_xquat[self.obj_bid].ravel()).ravel()
        palm_pos = self.data.site_xpos[self.S_grasp_sid].ravel()
        target_pos = self.data.site_xpos[self.target_obj_sid].ravel()
        nail_impact = np.clip(self.sim.data.sensordata[self.sim.model.sensor_name2id('S_nail')], -1.0, 1.0)
        return np.concatenate([qp[:-6], qv[-6:], palm_pos, obj_pos, obj_rot, target_pos, np.array([nail_impact])])

    def reset_model(self):
        self.sim.reset()
        target_bid = self.model.body_name2id('nail_board')
        self.model.body_pos[target_bid,2] = self.np_random.uniform(low=0.1, high=0.25)
        # Ensure sim.model reflects updated model positions
        try:
            self.sim.forward()
        except Exception:
            pass

        # After forward, apply distraction nails placement so they use current board_pos
        try:
            self._setup_distraction_nails()
        except Exception:
            pass
        try:
            # Ensure sim state updated before forcing orientations
            self.sim.forward()
        except Exception:
            pass

        # Force align nail bodies' orientations to the nail_board orientation
        try:
            board_bid = self.sim.model.body_name2id('nail_board')
            board_quat = self.sim.model.body_quat[board_bid].copy()
            # Align only distraction nails (nail_1..nail_8). Do NOT overwrite nail_0.
            for i in range(1, 9):
                try:
                    bid = self.sim.model.body_name2id(f'nail_{i}')
                    self.sim.model.body_quat[bid] = board_quat.copy()
                except Exception:
                    pass
        except Exception:
            pass

        try:
            self.sim.forward()
        except Exception:
            pass
        return self.get_obs()

    def get_env_state(self):
        """
        Get state of hand as well as objects and targets in the scene
        """
        qpos = self.data.qpos.ravel().copy()
        qvel = self.data.qvel.ravel().copy()
        board_pos = self.model.body_pos[self.model.body_name2id('nail_board')].copy()
        target_pos = self.data.site_xpos[self.target_obj_sid].ravel().copy()
        return dict(qpos=qpos, qvel=qvel, board_pos=board_pos, target_pos=target_pos)

    def set_env_state(self, state_dict):
        """
        Set the state which includes hand as well as objects and targets in the scene
        """
        qp = state_dict['qpos']
        qv = state_dict['qvel']
        board_pos = state_dict['board_pos']
        self.set_state(qp, qv)
        self.model.body_pos[self.model.body_name2id('nail_board')] = board_pos
        self.sim.forward()

    def mj_viewer_setup(self):
        self.viewer = MjViewer(self.sim)
        self.viewer.cam.azimuth = 45
        self.viewer.cam.distance = 2.0
        self.sim.forward()

    def evaluate_success(self, paths):
        num_success = 0
        num_paths = len(paths)
        # success if nail insude board for 25 steps
        for path in paths:
            if np.sum(path['env_infos']['goal_achieved']) > 25:
                num_success += 1
        success_percentage = num_success*100.0/num_paths
        return success_percentage
