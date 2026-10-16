import os
import numpy as np

from myosuite.utils import gym; register=gym.register
from myosuite.envs.env_variants import register_env_variant

curr_dir = os.path.dirname(os.path.abspath(__file__)).replace('\\', '/')
print("Reachers:> Registering Myo Envs")


# utility to register envs with all muscle conditions
def register_env_with_variants(id, entry_point, max_episode_steps, kwargs):
    # register_env_with_variants base env
    register(
        id=id,
        entry_point=entry_point,
        max_episode_steps=max_episode_steps,
        kwargs=kwargs
    )
    #register variants env with sarcopenia  #TODO: needs to be tested with myochallenge models
    if id[:3] == "myo":
        register_env_variant(
            env_id=id,
            variants={'muscle_condition':'sarcopenia'},
            variant_id=id[:3]+"Sarc"+id[3:],
            silent=True
        )
    #register variants with fatigue  #TODO: needs to be tested with myochallenge models
    if id[:3] == "myo":
        register_env_variant(
            env_id=id,
            variants={'muscle_condition':'fatigue'},
            variant_id=id[:3]+"Fati"+id[3:],
            silent=True
        )


# Arm posing ================================
register_env_with_variants(
    id='myoreachersArm-v0',
    entry_point="myoreachers.envs.reacher_v0:ReacherEnvV0",
    max_episode_steps=200,
    kwargs={
        'model_path': curr_dir + '/assets/arm/myoarm_reacher.xml',
        'normalize_act': True,
        'frame_skip': 5,
        'pos_th': 0.1,              # cover entire base of the receptacle
        'rot_th': np.inf,           # ignore rotation errors
        'target_xyz_range': {'high':[0, 0, 1.3], 'low':[-.5, -.5, .9]},
        'target_rxryrz_range': {'high':[0.0, 0.0, 0.0], 'low':[0.0, 0.0, 0.0]}
    }
)

# Elbow posing ==============================
register_env_with_variants(
    id="myoreachersElbow1D6MFixed-v0",
    entry_point="myoreachers.envs.pose_v0:PoseEnvV0",
    max_episode_steps=100,
    kwargs={
        "model_path": curr_dir + "/assets/elbow/myoelbow_1dof6muscles.xml",
        "target_jnt_range": {
            "r_elbow_flex": (2, 2),
        },
        "viz_site_targets": ("wrist",),
        "normalize_act": True,
        "pose_thd": 0.175,
        "reset_type": "random",
    },
)
register_env_with_variants(
    id="myoreachersElbow1D6MRandom-v0",
    entry_point="myoreachers.envs.pose_v0:PoseEnvV0",
    max_episode_steps=100,
    kwargs={
        "model_path": curr_dir + "/assets/elbow/myoelbow_1dof6muscles.xml",
        "target_jnt_range": {
            "r_elbow_flex": (0, 2.27),
        },
        "viz_site_targets": ("wrist",),
        "normalize_act": True,
        "pose_thd": 0.175,
        "reset_type": "random",
    },
)


# Wrist Reaching ============================
register_env_with_variants(
    id="myoreachersWristFixed-v0",
    entry_point="myoreachers.envs.radreach_v0:ReachEnvV0",
    max_episode_steps=100,
    kwargs={
        "model_path": curr_dir + "/assets/wrist/myowrist_pose.xml",
        "target_reach_range": {
            "MFtip": ((0.1739132007385475, 1, 0), (0.1739132007385475, 1, 0)),
        },
        "normalize_act": True,
        "far_th": 0.044,
    },
)
register_env_with_variants(
    id="myoreachersWristRandom-v0",
    entry_point="myoreachers.envs.radreach_v0:ReachEnvV0",
    max_episode_steps=100,
    kwargs={
        "model_path": curr_dir + "/assets/wrist/myowrist_pose.xml",
        "target_reach_range": {
            "MFtip": (
                #radius, theta, phi
                (0.1739, 1.35, 0),
                (0.1739, 1.75, 0),
            ),
        },
        "normalize_act": True,
        "far_th": 0.034,
    },
)


