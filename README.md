# Relational Deep Reinforcement Learning

Implementation of actor-critic architecture which incorporates relational inductive biases.

## Usage

Training and testing of PPO with relational module can be done through command line. For example -

```bash
python main --epochs 50 --episodes_per_epoch 32 -e Pong-v0 -v --enable_atari_wrapper  --enable_cuda
```

Full usage of cli -

```bash
usage: main.py [-h] [--env ENV] [--mode {train,test}] [--xpid XPID] [--epochs E] [--episodes_per_epoch EPISODES_PER_EPOCH] [--max_traj_length MAX_TRAJ_LENGTH] [--n_policy_updates N_POLICY_UPDATES]
               [--n_value_updates N_VALUE_UPDATES] [--p_lr P_LR] [--v_lr V_LR] [--discounting DISCOUNTING] [--epsilon EPSILON] [--grad_norm_clip GRAD_NORM_CLIP] [--verbose] [--render]
               [--checkpoint_every CHECKPOINT_EVERY] [--logdir LOGDIR] [--log_tensorboard] [--enable_atari_wrapper] [--enable_cuda]

PPO with relational inductive bias module.

optional arguments:
  -h, --help            show this help message and exit
  --env ENV, -e ENV     Gym environment.
  --mode {train,test}, -m {train,test}
                        Training or test mode.
  --xpid XPID           Experiment id (default: None).
  --epochs E            Total epochs to train for.
  --episodes_per_epoch EPISODES_PER_EPOCH
                        Episodes to run per epoch.
  --max_traj_length MAX_TRAJ_LENGTH
                        Maximum number of timesteps in a trajectory.
  --n_policy_updates N_POLICY_UPDATES
                        Number of times to take gradient steps on the policy each update.
  --n_value_updates N_VALUE_UPDATES
                        Number of times to take gradient steps on the value function each update.
  --p_lr P_LR           Policy learning rate.
  --v_lr V_LR           Value function learning rate.
  --discounting DISCOUNTING
                        Discounting factor.
  --epsilon EPSILON     Small threshold.
  --grad_norm_clip GRAD_NORM_CLIP
                        Global gradient norm clip.
  --verbose, -v         Enable to log progres to console.
  --render, -r          Enables rendering of environment.
  --checkpoint_every CHECKPOINT_EVERY
                        Epochs to checkpoint model every.
  --logdir LOGDIR       Root dir where experiment data will be saved.
  --log_tensorboard     Enable tensorboard logging
  --enable_atari_wrapper
                        Enable atari specific wrapper for env.
  --enable_cuda         Enable CUDA.
```

## References

1. Zambaldi, V., David R., Santoro A. et al. [Deep reinforcement learning with relational inductive biases](https://openreview.net/pdf?id=HkxaFoC9KQ). ICLR 2019.
