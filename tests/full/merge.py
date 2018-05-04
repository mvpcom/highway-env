from __future__ import division, print_function

import gym
import highway_env

from highway_env.agent.mcts import MCTSAgent, RobustMCTSAgent
from highway_env.agent.ttc_vi import TTCVIAgent
from highway_env.vehicle.behavior import LinearVehicle, IDMVehicle
from highway_env.wrappers.simulation import Simulation
from highway_env.wrappers.monitor import MonitorV2


def test():
    env = gym.make('highway-merge-v0')
    monitor = MonitorV2(env, 'out', force=True)
    agent = MCTSAgent(prior_policy=MCTSAgent.fast_policy,
                      rollout_policy=MCTSAgent.idle_policy,
                      iterations=50,
                      assume_vehicle_type=LinearVehicle)
    # agent = RobustMCTSAgent(models=[LinearVehicle, IDMVehicle],
    #                         prior_policy=MCTSAgent.fast_policy,
    #                         rollout_policy=MCTSAgent.idle_policy,
    #                         iterations=50)

    # agent = TTCVIAgent()
    sim = Simulation(monitor, agent, highway_env=env, env_seed=0, episodes=1)
    sim.run()


if __name__ == '__main__':
    import multiprocessing
    for _ in range(4):
        p = multiprocessing.Process(target=test)
        p.start()
