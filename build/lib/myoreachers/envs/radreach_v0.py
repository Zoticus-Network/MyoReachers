"""=================================================
# Copyright (c) MyoSuite Authors
Authors  :: Vikash Kumar (vikashplus@gmail.com), Vittorio Caggiano (caggiano@gmail.com)

# Modifications Copyright (c) 2026 Nathaniel Ward
Authors  :: Nathaniel Ward (nathaniel@zoticus.network)
================================================="""

import collections

import mujoco
import numpy as np

from myoreachers.envs.base_v0 import BaseV0
from myosuite.utils import gym


class ReachEnvV0(BaseV0):

    DEFAULT_OBS_KEYS = ["qpos", "qvel", "tip_pos", "reach_err"]
    DEFAULT_RWD_KEYS_AND_WEIGHTS = {
        "reach": 1.0,
        "bonus": 4.0,
        "penalty": 50,
    }

    def __init__(
        self, model_path, obsd_model_path=None, seed=None, edit_fn=None, **kwargs
    ):

        # EzPickle.__init__(**locals()) is capturing the input dictionary of the init method of this class.
        # In order to successfully capture all arguments we need to call gym.utils.EzPickle.__init__(**locals())
        # at the leaf level, when we do inheritance like we do here.
        # kwargs is needed at the top level to account for injection of __class__ keyword.
        # Also see: https://github.com/openai/gym/pull/1497
        gym.utils.EzPickle.__init__(
            self, model_path, obsd_model_path, seed, edit_fn=edit_fn, **kwargs
        )

        # This two step construction is required for pickling to work correctly. All arguments to all __init__
        # calls must be pickle friendly. Things like sim / sim_obsd are NOT pickle friendly. Therefore we
        # first construct the inheritance chain, which is just __init__ calls all the way down, with env_base
        # creating the sim / sim_obsd instances. Next we run through "setup"  which relies on sim / sim_obsd
        # created in __init__ to complete the setup.
        super().__init__(
            model_path=model_path,
            obsd_model_path=obsd_model_path,
            seed=seed,
            edit_fn=edit_fn,
            env_credits=self.MYO_CREDIT,
        )

        self.act_change = 0
        self.act_time   = 0
        self.prev_act   = np.zeros(39)

        self._setup(**kwargs)

    def _setup(
        self,
        target_reach_range: dict,
        far_th=0.35,
        obs_keys: list = DEFAULT_OBS_KEYS,
        weighted_reward_keys: dict = DEFAULT_RWD_KEYS_AND_WEIGHTS,
        **kwargs,
    ):
        self.far_th = far_th
        self.target_reach_range = target_reach_range
        super()._setup(
            obs_keys=obs_keys,
            weighted_reward_keys=weighted_reward_keys,
            sites=self.target_reach_range.keys(),
            **kwargs,
        )

    def get_obs_vec(self):
        self.obs_dict["time"] = np.array([self.mj_data.time])
        self.obs_dict["qpos"] = self.mj_data.qpos[:].copy()
        self.obs_dict["qvel"] = self.mj_data.qvel[:].copy() * self.dt
        if self.mj_model.na > 0:
            self.obs_dict["act"] = self.mj_data.act[:].copy()

        # reach error
        self.obs_dict["tip_pos"] = np.array([])
        self.obs_dict["target_pos"] = np.array([])
        for isite in range(len(self.tip_sids)):
            self.obs_dict["tip_pos"] = np.append(
                self.obs_dict["tip_pos"],
                self.mj_data.site_xpos[self.tip_sids[isite]].copy(),
            )
            self.obs_dict["target_pos"] = np.append(
                self.obs_dict["target_pos"],
                self.mj_data.site_xpos[self.target_sids[isite]].copy(),
            )
        self.obs_dict["reach_err"] = np.array(self.obs_dict["target_pos"]) - np.array(
            self.obs_dict["tip_pos"]
        )

        t, obs = self.obsdict2obsvec(self.obs_dict, self.obs_keys)
        return obs

    def get_obs_dict(self, mj_model, mj_data):
        obs_dict = {}
        obs_dict["time"] = np.array([mj_data.time])
        obs_dict["qpos"] = mj_data.qpos[:].copy()
        obs_dict["qvel"] = mj_data.qvel[:].copy() * self.dt
        if mj_model.na > 0:
            obs_dict["act"] = mj_data.act[:].copy()

        # reach error
        obs_dict["tip_pos"] = np.array([])
        obs_dict["tip_vel"] = np.array([])
        obs_dict["target_pos"] = np.array([])
        for isite in range(len(self.tip_sids)):
            obs_dict["tip_pos"] = np.append(
                obs_dict["tip_pos"], mj_data.site_xpos[self.tip_sids[isite]].copy()
            )
            obs_dict["tip_vel"] = np.append(
                obs_dict["tip_vel"], mj_data.sensor(self.tip_vel_sids[isite]).data.copy()*self.unwrapped.dt
            )
            obs_dict["target_pos"] = np.append(
                obs_dict["target_pos"],
                mj_data.site_xpos[self.target_sids[isite]].copy(),
            )
        obs_dict["reach_err"] = np.array(obs_dict["target_pos"]) - np.array(
            obs_dict["tip_pos"]
        )

        # action change
        if mj_data.time != self.act_time:
            self.act_change = self.prev_act - obs_dict["act"]
            self.act_time   = mj_data.time
            self.prev_act   = obs_dict["act"]

        obs_dict["act_change"] = self.act_change

        return obs_dict

    def get_reward_dict(self, obs_dict):
        err = obs_dict["reach_err"]
        reach_dist = np.linalg.norm(err, axis=-1)
        vel = obs_dict["tip_vel"]
        speed = np.linalg.norm(vel, axis=-1)
        current_time = float(np.asarray(obs_dict["time"]).squeeze())
        motion_similarity = (
            np.einsum('ij,ij->i', vel[0], err[0])/(speed*reach_dist)
            if (len(vel.shape) > 1 and vel.shape[1] > 1) else
            np.dot(vel.flatten(), err.flatten())/(speed*reach_dist)
        )
        act_change = (
            np.sum(self.act_change)
            if (self.spec != None and current_time > (self.horizon // 4) * self.dt) else
            0
        )
        act_mag = (
            np.linalg.norm(self.obs_dict["act"], axis=-1) / self.mj_model.na
            if self.mj_model.na != 0
            else 0
        )
        far_th = (
            self.far_th * len(self.tip_sids)
            if np.squeeze(obs_dict["time"]) > 2 * self.dt
            else np.inf
        )
        near_th = len(self.tip_sids) * 0.0125
        rwd_dict = collections.OrderedDict(
            (
                # Optional Keys
                ("reach", -1.0 * reach_dist),
                (
                    "bonus",
                    1.0 * (reach_dist < 2 * near_th) + 1.0 * (reach_dist < near_th),
                ),
                ("act_reg", -1.0 * act_mag),
                ("penalty", -1.0 * (reach_dist > far_th)),
                ("motion", -1.0 * speed),
                ("moving_away", np.mean(motion_similarity) - 0.75), # cos_sim(err,velocity) - 0.75
                ("act_change", act_change), 
                # Must keys
                ("sparse", -1.0*np.log( np.abs( 75*reach_dist )+.1 ) - np.abs(reach_dist/2)),
                ("solved", reach_dist < near_th),
                ("done", reach_dist > far_th),
            )
        )
        rwd_dict["dense"] = np.sum(
            [wt * rwd_dict[key] for key, wt in self.rwd_keys_wt.items()], axis=0
        )
        return rwd_dict

    # generate a valid target
    def generate_target_pose(self):
        for site, span in self.target_reach_range.items():
            sid = self.mj_model.site(site + "_target").id
            radius, theta, phi = self.np_random.uniform(
                low=span[0], high=span[1]
            )
            theta *= np.pi
            phi   *= np.pi
            center = np.array([-0.2021, -0.4464, 1.4284])
            target = np.array([
                radius*np.cos(phi)*np.cos(theta),
                radius*np.cos(phi)*np.sin(theta),
                radius*np.sin(phi),
            ])
            pos = center + target
            self.mj_model.site_pos[sid] = pos
        mujoco.mj_forward(self.mj_model, self.mj_data)

    def reset(self, **kwargs):
        self.generate_target_pose()
        self.robot.sync_sims(
            self.mj_model, self.mj_data, self.obsd_mj_model, self.obsd_mj_data
        )

        self.act_change = 0
        self.act_time   = 0
        self.prev_act   = np.zeros_like(self.action_space) 
        # self.prev_act   = np.zeros(39) 

        obs = super().reset(**kwargs)
        return obs
