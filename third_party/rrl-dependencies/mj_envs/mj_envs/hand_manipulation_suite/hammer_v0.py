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

        # Nail management
        self.nail_bodies = []  # List of nail body ids
        self.nail_joints = []  # List of nail joint ids
        self.nail_sites = []   # List of nail site ids (for distraction nails)
        self.active_nails = [] # Currently active distraction nails

        curr_dir = os.path.dirname(os.path.abspath(__file__))
        mujoco_env.MujocoEnv.__init__(self, curr_dir+'/assets/DAPG_hammer.xml', 5)
        utils.EzPickle.__init__(self)

        # change actuator sensitivity
        self.sim.model.actuator_gainprm[self.sim.model.actuator_name2id('A_WRJ1'):self.sim.model.actuator_name2id('A_WRJ0')+1,:3] = np.array([10, 0, 0])
        self.sim.model.actuator_gainprm[self.sim.model.actuator_name2id('A_FFJ3'):self.sim.model.actuator_name2id('A_THJ0')+1,:3] = np.array([1, 0, 0])
        self.sim.model.actuator_biasprm[self.sim.model.actuator_name2id('A_WRJ1'):self.sim.model.actuator_name2id('A_WRJ0')+1,:3] = np.array([0, -10, 0])
        self.sim.model.actuator_biasprm[self.sim.model.actuator_name2id('A_FFJ3'):self.sim.model.actuator_name2id('A_THJ0')+1,:3] = np.array([0, -1, 0])

        # Initialize nail management
        self._initialize_nail_management()

        self.target_obj_sid = self.sim.model.site_name2id('S_target')
        self.S_grasp_sid = self.sim.model.site_name2id('S_grasp')
        self.obj_bid = self.sim.model.body_name2id('Object')
        self.tool_sid = self.sim.model.site_name2id('tool')
        self.goal_sid = self.sim.model.site_name2id('nail_goal')
        self.act_mid = np.mean(self.model.actuator_ctrlrange, axis=1)
        self.act_rng = 0.5 * (self.model.actuator_ctrlrange[:, 1] - self.model.actuator_ctrlrange[:, 0])
        self.action_space.high = np.ones_like(self.model.actuator_ctrlrange[:,1])
        self.action_space.low  = -1.0 * np.ones_like(self.model.actuator_ctrlrange[:,0])

    def _initialize_nail_management(self):
        """Initialize nail management system for distraction nails"""
        # Find all nail bodies, joints, and sites
        for i in range(10):  # We have nail_0 to nail_9
            try:
                body_name = f'nail_{i}'
                joint_name = f'nail_{i}_dir'

                body_id = self.sim.model.body_name2id(body_name)
                joint_id = self.sim.model.joint_name2id(joint_name)

                self.nail_bodies.append(body_id)
                self.nail_joints.append(joint_id)

                # For distraction nails (i > 0), also store site ids
                if i > 0:
                    site_name = f'S_dist_{i}'
                    site_id = self.sim.model.site_name2id(site_name)
                    self.nail_sites.append(site_id)

            except ValueError:
                # Body/joint/site not found, skip
                continue

        # Initially deactivate all distraction nails (move them underground)
        self._deactivate_all_distraction_nails()

    def _deactivate_all_distraction_nails(self):
        """Move all distraction nails underground"""
        for i in range(1, len(self.nail_bodies)):  # Skip nail_0 (target nail)
            body_id = self.nail_bodies[i]
            # Move to underground position
            self.model.body_pos[body_id] = np.array([0.0, 0.0, -0.5])
        self.active_nails = []
        self.sim.forward()

    def set_distraction_nails(self, num_nails):
        """
        Set the number of active distraction nails.
        Nails are placed at predefined positions around the target nail.

        Args:
            num_nails (int): Number of distraction nails to activate (0-9)
        """
        num_nails = max(0, min(num_nails, 9))  # Clamp to valid range

        # Deactivate all current distraction nails
        self._deactivate_all_distraction_nails()

        if num_nails > 0:
            # Define positions for distraction nails (relative to nail board center)
            # These positions form a pattern around the target nail
            positions = [
                [0.02, 0.02, 0.0],   # Position for nail_1
                [-0.02, 0.02, 0.0],  # Position for nail_2
                [0.02, -0.02, 0.0],  # Position for nail_3
                [-0.02, -0.02, 0.0], # Position for nail_4
                [0.04, 0.0, 0.0],    # Position for nail_5
                [-0.04, 0.0, 0.0],   # Position for nail_6
                [0.0, 0.04, 0.0],    # Position for nail_7
                [0.0, -0.04, 0.0],   # Position for nail_8
                [0.03, 0.03, 0.0],   # Position for nail_9
            ]

            # Get nail board position as reference
            board_pos = self.model.body_pos[self.model.body_name2id('nail_board')].copy()

            # Activate the requested number of nails
            for i in range(num_nails):
                body_id = self.nail_bodies[i + 1]  # +1 because nail_0 is target
                # Set position relative to nail board
                self.model.body_pos[body_id] = board_pos + np.array(positions[i])
                self.active_nails.append(i + 1)  # Store nail index (1-9)

        self.sim.forward()

    def get_distraction_nails(self):
        """Get list of currently active distraction nail indices"""
        return self.active_nails.copy()

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

        # Reset distraction nails to current configuration
        self.set_distraction_nails(self.num_distraction_nails)

        self.sim.forward()
        return self.get_obs()

    def get_env_state(self):
        """
        Get state of hand as well as objects and targets in the scene
        """
        qpos = self.data.qpos.ravel().copy()
        qvel = self.data.qvel.ravel().copy()
        board_pos = self.model.body_pos[self.model.body_name2id('nail_board')].copy()
        target_pos = self.data.site_xpos[self.target_obj_sid].ravel().copy()

        # Include distraction nail positions
        distraction_positions = {}
        for i in range(1, len(self.nail_bodies)):
            body_name = f'nail_{i}'
            body_id = self.nail_bodies[i]
            distraction_positions[body_name] = self.model.body_pos[body_id].copy()

        return dict(qpos=qpos, qvel=qvel, board_pos=board_pos, target_pos=target_pos,
                   distraction_positions=distraction_positions, num_distraction_nails=self.num_distraction_nails)

    def set_env_state(self, state_dict):
        """
        Set the state which includes hand as well as objects and targets in the scene
        """
        qp = state_dict['qpos']
        qv = state_dict['qvel']
        board_pos = state_dict['board_pos']
        self.set_state(qp, qv)
        self.model.body_pos[self.model.body_name2id('nail_board')] = board_pos

        # Restore distraction nail positions if available
        if 'distraction_positions' in state_dict:
            for body_name, pos in state_dict['distraction_positions'].items():
                try:
                    body_id = self.sim.model.body_name2id(body_name)
                    self.model.body_pos[body_id] = pos
                except ValueError:
                    pass  # Body not found, skip

        # Restore number of distraction nails if available
        if 'num_distraction_nails' in state_dict:
            self.num_distraction_nails = state_dict['num_distraction_nails']

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
