import ray
from ray import tune
from ray.rllib.models import ModelCatalog
from ray.tune.registry import register_env

from env.envs import LazyMsgListenersTrainEnv
from model.lazy_listener_mlp import InfoLazyMLP

if __name__ == "__main__":

    do_debug = False
    # do_debug = True

    if do_debug:
        ray.init(local_mode=True)

    num_agents = 20
    env_config = {
        "num_agents_pool": num_agents,
    }
    env_name = "lazy_msg_listener_env"  # for train tho
    register_env(env_name, lambda cfg: LazyMsgListenersTrainEnv(cfg))

    # register your custom model
    model_name = "lazy_listener_model_mlp"
    ModelCatalog.register_custom_model(model_name, InfoLazyMLP)

    # train
    tune.run(
        "PPO",
        name="info-lazy_mlp_250802",
        # resume=True,
        stop={"training_iteration": 780},
        checkpoint_freq=1,
        keep_checkpoints_num=32,
        checkpoint_at_end=True,
        checkpoint_score_attr="episode_reward_mean",
        config={
            "env": env_name,
            "env_config": env_config,
            "framework": "torch",
            "model": {
                "custom_model": model_name,
            },
            "num_gpus": 1,
            "num_workers": 8,
            "num_envs_per_worker": 2,
            "rollout_fragment_length": 1024,
            "train_batch_size": 1024*16,
            "sgd_minibatch_size": 256,
            "num_sgd_iter": 10,
            "lr": 2e-5,
            "lr_schedule": [[0, 2e-5],
                            [1e7, 1e-7],
                            ],
            "vf_loss_coeff": 0.5,
            "use_critic": True,
            "use_gae": True,
            "gamma": 0.99,
            "lambda": 0.95,
            "kl_coeff": 0,
            "clip_param": 0.2,
            "vf_clip_param": 256,
            "grad_clip": 0.5,
            "kl_target": 0.01,
        },
    )

