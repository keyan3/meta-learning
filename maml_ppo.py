import math
import datetime
import argparse

import torch
import torch.nn as nn
from torch.distributions import Categorical
from torch.utils.tensorboard import SummaryWriter
from torchvision.models import resnet18
import gym
from retro_contest.local import make
from tqdm import tqdm
import numpy as np

import sonic_util
from PPO import Memory, ActorCritic, PPO
from sonic_util import get_sonic_specific_actions, sample_level_batch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser()
parser.add_argument('--run_name', type=str)
parser.add_argument('--render', type=bool)
parser.add_argument('--dry_run', type=bool)
args = parser.parse_args()

if not args.dry_run:
    curr_time = str(datetime.datetime.now()).replace(' ', '_')
    writer = SummaryWriter(log_dir='../runs/MAML_{}_{}'.format(args.run_name, curr_time))

def train_meta_ppo(level, meta_policy, meta_policy_lr, end_of_meta_iter):
    ############## Hyperparameters ##############
    env = make(game=game, state=level)
    state_dim = env.observation_space.shape[0]
    action_dim = 8
    sonic_actions = get_sonic_specific_actions()
    render = args.render
    solved_reward = 4e3           # stop training if avg_reward > solved_reward
    log_interval = 1              # print avg reward in the interval
    num_episodes = 3              # max training episodes
    max_timesteps = 3000          # max timesteps in one episode
    # update_timestep = 9000      # update policy every n timesteps
    lr = 0.0001
    batch_size = 400
    betas = (0.9, 0.999)
    gamma = 0.99                # discount factor
    K_epochs = 1                # update policy for K epochs
    eps_clip = 0.2              # clip parameter for PPO
    random_seed = None
    #############################################
    
    if random_seed:
        torch.manual_seed(random_seed)
        env.seed(random_seed)

    memory = Memory()
    ppo = PPO(state_dim, action_dim, n_latent_var, lr, betas, gamma, K_epochs, 
              eps_clip, batch_size, load_saved, meta_policy, meta_policy_lr)
    
    # training loop
    for i_episode in range(1, max_episodes+2):
        state = env.reset()
        running_reward = 0
        avg_length = 0
        for t in range(max_timesteps):
            
            # Running policy_old:
            action = ppo.policy_old.act(state, memory)
            # print(action)
            state, reward, done, _ = env.step(sonic_actions[action].astype(int))
            
            # Saving reward and is_terminal:
            memory.rewards.append(reward * 0.005)
            memory.is_terminals.append(done)
            
            running_reward += reward
            if render:
                env.render()
            if done:
                break
                
        avg_length += t
                
        # logging
        if i_episode % log_interval == 0:
            avg_length = int(avg_length/log_interval)
            running_reward = int((running_reward/log_interval))
            
            print('Episode {} \t avg length: {} \t reward: {}'.format(i_episode, avg_length, running_reward))
            writer.add_scalar('Avg_Reward', running_reward, i_episode)
            writer.flush()
        
        if i_episode == max_episodes:
            print('Updating {} policy'.format(level))
            ppo.update(memory)
            memory.clear_memory()
        
        if i_episode == max_episodes + 1:
            print('Collecting gradients in meta-policy')
            ppo.update_meta(memory, end_of_meta_iter)
            memory.clear_memory()
            return running_reward


def main():
    ############## Hyperparameters ##############
    meta_iterations = 80
    meta_policy_lr = 0.001
    #############################################

    meta_policy = ActorCritic((224, 320, 3), action_dim).to(device)

    for i_metaiter in range(meta_iterations):
        
        levels = sample_level_batch()
        total_reward = 0

        for i, level in enumerate(levels):
            post_update_reward = train_meta_ppo(level=level, meta_policy, meta_policy_lr, i == 2)
            total_reward += post_update_reward

        # save model
        torch.save(meta_policy.state_dict(), 'preTrained/meta_PPO.pth')
        
        # logging        
        print('Metaiter {} \t 3-level reward: {}'.format(i_metaiter, total_reward))
        writer.add_scalar('Avg_Reward_Task_Batch', total_reward, i_metaiter)
        writer.flush()


if __name__ == '__main__':
    main()
