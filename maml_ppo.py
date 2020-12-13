import math
import datetime
import argparse
from typing import List

import torch
import torch.nn as nn
from torch.distributions import Categorical
from torch.utils.tensorboard import SummaryWriter
from torchvision.models import resnet18
import gym
from retro_contest.local import make
from tqdm import tqdm
import numpy as np
import higher

import sonic_util
from PPO import Memory, ActorCritic
from sonic_util import get_sonic_specific_actions, sample_level_batch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser()
parser.add_argument('--run_name', type=str)
parser.add_argument('--render', dest='render', action='store_true')
parser.add_argument('--resnet', dest='resnet', action='store_true')
parser.add_argument('--dry_run', dest='dry_run', action='store_true')
args = parser.parse_args()


def gradadd(grads1: List[torch.Tensor], grads2: List[torch.Tensor]):
    for i in range(len(grads1)):
        grads1[i] += grads2[i]
    return grads1

class PPOMeta:
    def __init__(self, state_dim, action_dim, lr, betas, gamma, K_epochs, eps_clip, net_batch_size, 
                 load_saved, entropy_loss_weight, meta_policy):
        self.lr = lr
        self.betas = betas
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        self.net_batch_size = net_batch_size
        self.entropy_loss_weight = entropy_loss_weight
        self.meta_policy = meta_policy
        self.optimizer = torch.optim.Adam(self.meta_policy.parameters(), lr=lr, betas=betas)
        self.MseLoss = nn.MSELoss()
    
    def update(self, memory, meta_policy_update=False):   
        # Monte Carlo estimate of state rewards:
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(memory.rewards), reversed(memory.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)
        
        # Normalizing the rewards:
        rewards = torch.tensor(rewards, dtype=torch.float32)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)

        # convert list to tensor
        old_states = memory.states
        old_actions = memory.actions
        old_logprobs = memory.logprobs
        
        self.meta_policy.train_mode()

        with higher.innerloop_ctx(self.meta_policy, self.optimizer) as (fmeta_policy, diffopt):

            total_loss = 0
            mem_size = len(old_states)
            epoch_indices = np.arange(mem_size)
            grad_wrt_original = None

            if not meta_policy_update:
                self.fmeta_policy = fmeta_policy
            else:
                self.fmeta_policy.train_mode()

            for i in range(math.ceil(mem_size / self.net_batch_size)):
                batch_indices = epoch_indices[i * self.net_batch_size : min((i + 1) * self.net_batch_size, mem_size)]
                old_states_batch = torch.stack([old_states[i] for i in batch_indices]).to(device).detach()
                old_actions_batch = torch.stack([old_actions[i] for i in batch_indices]).to(device).detach()
                old_logprobs_batch = torch.stack([old_logprobs[i] for i in batch_indices]).to(device).detach()
                rewards_batch = rewards[batch_indices].to(device)

                # Evaluating old actions and values :
                logprobs, state_values, dist_entropy = self.fmeta_policy.evaluate(old_states_batch, old_actions_batch)
                
                # Finding the ratio (pi_theta / pi_theta__old):
                ratios = torch.exp(logprobs - old_logprobs_batch.detach())
                    
                # Finding Surrogate Loss:
                advantages = rewards_batch - state_values.detach()
                surr1 = ratios * advantages
                surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
                loss = (
                        -torch.min(surr1, surr2) 
                        + 0.5 * self.MseLoss(state_values, rewards_batch) 
                        - self.entropy_loss_weight * dist_entropy
                )
                total_loss += loss.mean().item()

                if not meta_policy_update:
                    diffopt.step(loss.mean())
                else:
                    # collect gradient of loss w.r.t original theta
                    if grad_wrt_original is None:
                        grad_wrt_original = list(torch.autograd.grad(loss.mean(), self.fmeta_policy.parameters(time=0)))
                    else:
                        grad_curr = list(torch.autograd.grad(loss.mean(), self.fmeta_policy.parameters(time=0)))
                        grad_wrt_original = gradadd(grad_wrt_original, grad_curr)

            print('Loss: ', total_loss)
        
        # Copy new weights into old policy:
        self.meta_policy.eval_mode()
        self.fmeta_policy.eval_mode()

        if meta_policy_update:
            return grad_wrt_original

def train_meta_ppo(level, meta_policy):
    
    if level in sonic_util.LEVELS1:
        game = 'SonicTheHedgehog-Genesis'
    elif level in sonic_util.LEVELS2:
        game = 'SonicTheHedgehog2-Genesis'
    elif level in sonic_util.LEVELS3:
        game = 'SonicAndKnuckles3-Genesis'
    env = make(game=game, state=level)

    ############## Hyperparameters ##############
    state_dim = env.observation_space.shape[0]
    action_dim = 8
    sonic_actions = get_sonic_specific_actions()
    render = args.render
    log_interval = 1              # print avg reward in the interval
    num_episodes = 3              # max training episodes
    max_timesteps = 3000          # max timesteps in one episode
    lr = 0.0001
    batch_size = 200
    betas = (0.9, 0.999)
    gamma = 0.99                # discount factor
    K_epochs = 1                # update policy for K epochs
    eps_clip = 0.2              # clip parameter for PPO
    entropy_loss_weight = 0.1
    random_seed = None
    #############################################
    
    if random_seed:
        torch.manual_seed(random_seed)
        env.seed(random_seed)

    memory = Memory()
    ppo = PPOMeta(state_dim, action_dim, lr, betas, gamma, K_epochs, eps_clip, batch_size, False, entropy_loss_weight, meta_policy)
    # training loop
    for i_episode in range(1, num_episodes+2):
        state = env.reset()
        running_reward = 0
        avg_length = 0
        for t in range(max_timesteps):
            
            # Running policy_old:
            if i_episode == num_episodes + 1:
                action = ppo.fmeta_policy.act(state, memory)
            else:
                action = ppo.meta_policy.act(state, memory)
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
        
        if i_episode == num_episodes:
            print('Updating {} policy'.format(level))
            ppo.update(memory)
            memory.clear_memory()
        
        if i_episode == num_episodes + 1:
            print('Collecting gradients in meta-policy')
            grad_wrt_original_meta = ppo.update(memory, meta_policy_update=True)
            memory.clear_memory()
            return running_reward, grad_wrt_original_meta


def main():
    ############## Hyperparameters ##############
    meta_iterations = 80
    meta_policy_lr = 0.00001
    #############################################

    if not args.dry_run:
        curr_time = str(datetime.datetime.now()).replace(' ', '_')
        writer = SummaryWriter(log_dir='../runs/MAML_{}_{}'.format(args.run_name, curr_time))

    if args.resnet:
        meta_policy = ActorCritic((224, 320, 3), 8, resnet_expl=True).to(device)
    else:
        meta_policy = ActorCritic((224, 320, 3), 8, eps_greedy=0.02, resnet_expl=True).to(device)

    for i_metaiter in range(meta_iterations):
        
        levels = sample_level_batch()
        total_reward = 0

        meta_gradient_total = None

        for i, level in enumerate(levels):
            post_update_reward, gradient_wrt_theta = train_meta_ppo(level, meta_policy)
            if meta_gradient_total is None:
                meta_gradient_total = gradient_wrt_theta
            else:
                meta_gradient_total = gradadd(meta_gradient_total, gradient_wrt_theta)
            
            total_reward += post_update_reward
        
        print(meta_gradient_total)

        print('Updating meta-policy')
        with torch.no_grad():
            for i, p in enumerate(meta_policy.parameters()):
                p.copy_(p - meta_policy_lr * meta_gradient_total[i])

        # save model
        torch.save(meta_policy.state_dict(), 'preTrained/meta_PPO.pth')
        
        # logging        
        print('Metaiter {} \t 3-level reward: {}'.format(i_metaiter, total_reward))
        print('\n')
        if not args.dry_run:
            writer.add_scalar('Avg_Reward_Task_Batch', total_reward, i_metaiter)
            writer.flush()


if __name__ == '__main__':
    main()
