from rllab.algos.ddpg_polyRL import DDPG
from rllab.envs.normalized_env import normalize
from rllab.misc.instrument import run_experiment_lite
from rllab.exploration_strategies.ou_strategy import OUStrategy
from rllab.exploration_strategies.persistence_length_2D_v1 import Persistence_Length_Exploration
from rllab.policies.deterministic_mlp_policy import DeterministicMLPPolicy
from rllab.q_functions.continuous_mlp_q_function import ContinuousMLPQFunction

from rllab.envs.mujoco.swimmer_env import SwimmerEnv

env = normalize(SwimmerEnv())

print ("Action High", env.action_space.high)
print ("Action Low", env.action_space.low)
print("Observation Space", env.observation_space)



def run_task(*_):


    """
    DPG on Swimmer environment
    """
    env = normalize(SwimmerEnv())

    """
    Initialise the policy as a neural network policy
    """
    policy = DeterministicMLPPolicy(
        env_spec=env.spec,
        # The neural network policy should have two hidden layers, each with 32 hidden units.
        hidden_sizes=(32, 32)
    )


    """
    Defining exploration strategy : OUStrategy - 
    """
    """
    This strategy implements the Ornstein-Uhlenbeck process, which adds
    time-correlated noise to the actions taken by the deterministic policy.
    The OU process satisfies the following stochastic differential equation:
    dxt = theta*(mu - xt)*dt + sigma*dWt
    where Wt denotes the Wiener process
    """
    es = OUStrategy(env_spec=env.spec)


    """
    Defining the Q network
    """
    qf = ContinuousMLPQFunction(env_spec=env.spec)


    w = qf.get_param_values(regularizable=True)

    """
    Persistence Length Exploration
    """
    lp = Persistence_Length_Exploration(env=env, qf=qf, policy=policy)


    """
    Using the DDPG algorithm
    """
    algo = DDPG(
        env=env,
        policy=policy,
        es=es,
        qf=qf,
        lp=lp,
        batch_size=32,
        max_path_length=1000,
        epoch_length=1000,
        min_pool_size=10000,
        n_epochs=15000,
        discount=0.99,
        scale_reward=0.01,
        qf_learning_rate=1e-3,
        policy_learning_rate=1e-4,
        #Uncomment both lines (this and the plot parameter below) to enable plotting
        plot=True,
    )


    """
    Training the networks based on the DDPG algorithm
    """
    algo.train()

run_experiment_lite(
    run_task,
    # Number of parallel workers for sampling
    n_parallel=1,
    # Only keep the snapshot parameters for the last iteration
    snapshot_mode="last",
    # Specifies the seed for the experiment. If this is not provided, a random seed
    # will be used
    seed=1,
    plot=True,
)
