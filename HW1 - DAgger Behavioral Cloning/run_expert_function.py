#!/usr/bin/env python

"""

*** Modification fron original file : turned into a function ***

Code to load an expert policy and generate roll-out data for behavioral cloning.
Example usage:
    python run_expert.py experts/Humanoid-v1.pkl Humanoid-v1 --render \
            --num_rollouts 20

Author of this script and included expert policies: Jonathan Ho (hoj@openai.com)

"""


'''
# Install mujoco_py

pip install -r requirements.txt
pip install -r requirements.dev.txt

#Now install mujoco-py

python setup.py install
'''


import os
import pickle
import tensorflow as tf
import numpy as np
import tf_util
import gym
import load_policy

def run_expert(envname, num_rollouts=20, render=True, max_timesteps=1000, export_path='./expert_data', expert_policy_path='./experts/'):

    print('loading and building expert policy')
    policy_fn = load_policy.load_policy(expert_policy_path+envname+'.pkl')
    print('loaded and built')

    with tf.Session():
        tf_util.initialize()
        
        env = gym.make(envname)
        max_steps = max_timesteps or env.spec.timestep_limit

        returns = []
        observations = []
        actions = []
        for i in range(num_rollouts):
            print('iter', i)
            obs = env.reset()
            done = False
            totalr = 0.
            steps = 0
            while not done:
                action = policy_fn(obs[None,:])
                observations.append(obs)
                actions.append(action)
                obs, r, done, _ = env.step(action)
                totalr += r
                steps += 1
                if render:
                    env.render()
                if steps % 100 == 0: print("%i/%i"%(steps, max_steps))
                if steps >= max_steps:
                    break
            returns.append(totalr)

        # print('returns', returns)
        # print('mean return', np.mean(returns))
        # print('std of return', np.std(returns))

        expert_data = {'observations': np.array(observations),
                       'actions': np.array(actions)}

        with open(os.path.join(export_path, envname + '.pkl'), 'wb') as f:
            pickle.dump(expert_data, f, pickle.HIGHEST_PROTOCOL)
           

def run_expert_on_obsv(observations, envname, expert_policy_path='./experts/'):

    print('loading and building expert policy')
    policy_fn = load_policy.load_policy(expert_policy_path+envname+'.pkl')
    print('loaded and built')

    actions = []
        
    with tf.Session():
        tf_util.initialize()
        
        for i,obs in enumerate(observations):
            print("Observation : {}/{}".format(i,len(observations)))
            action = policy_fn(obs[None,:])
            actions.append(action)

    expert_data = {'observations': np.array(observations), 'actions': np.array(actions)}

    return expert_data         