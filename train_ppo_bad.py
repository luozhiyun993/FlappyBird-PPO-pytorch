
import argparse
import os
from torch.utils.data import   BatchSampler, SubsetRandomSampler
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from src.flappy_bird import FlappyBird

def get_args():
    parser = argparse.ArgumentParser(
        """Implementation of PPO to play Flappy Bird""")
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--num_iters", type=int, default=20000)
    parser.add_argument("--log_path", type=str, default="tensorboard_ppo")
    parser.add_argument("--saved_path", type=str, default="trained_models")
    parser.add_argument("--lmbda", type=float, default=0.95)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--eps", type=float, default=0.2)
    parser.add_argument("--batch_size",type=int, default=2048 )
    parser.add_argument("--mini_batch_size",type=int, default=64 )

    args = parser.parse_args()
    return args


class PolicyNet(nn.Module):
    def __init__(self):
        super(PolicyNet, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(4, 32, kernel_size=8, stride=4), nn.ReLU())
        self.conv2 = nn.Sequential(nn.Conv2d(32, 64, kernel_size=4, stride=2), nn.ReLU())
        self.conv3 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, stride=1), nn.ReLU())
        self.flat = nn.Flatten()
        self.fc1 = nn.Sequential(nn.Linear(7 * 7 * 64, 512), nn.Tanh())
        self.drop = nn.Dropout(0.5)
        self.fc3 = nn.Sequential(nn.Linear(512, 2))
    def forward(self, input):
        output = self.conv1(input)
        output = self.conv2(output)
        output = self.conv3(output)
        output = self.flat(output)
        output = self.drop(output)
        output = self.fc1(output)
        return nn.functional.softmax(self.fc3(output), dim=1)



class ValueNet(nn.Module):
    def __init__(self):
        super(ValueNet, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=8, stride=4), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2), nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1), nn.ReLU(),
            nn.Flatten(),
            nn.Linear(7 * 7 * 64, 512), nn.Tanh(),
            nn.Dropout(0.5),
            nn.Linear(512, 1),
        )

    def forward(self, input):
        return self.net(input)

def compute_advantage(gamma, lmbda, td_delta):
    td_delta = td_delta.detach().numpy()
    advantage_list = []
    advantage = 0.0
    for delta in td_delta[::-1]:
        advantage = gamma * lmbda * advantage + delta
        advantage_list.append(advantage)
    advantage_list.reverse()
    return torch.tensor(advantage_list, dtype=torch.float)
def train(opt):
    if torch.cuda.is_available():
        torch.cuda.manual_seed(1993)
    else:
        torch.manual_seed(123)
    actor = PolicyNet().cuda()
    critic = ValueNet().cuda()

    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=opt.lr)
    critic_optimizer = torch.optim.Adam(critic.parameters(), lr=opt.lr)

    if os.path.exists("{}/flappy_bird_actor".format(opt.saved_path)):
        checkpoint = torch.load("{}/flappy_bird_actor".format(opt.saved_path))
        actor.load_state_dict(checkpoint['net'])
        actor_optimizer.load_state_dict(checkpoint['optimizer'])
        print("load actor succ")

    if os.path.exists("{}/flappy_bird_critic".format(opt.saved_path)):
        checkpoint = torch.load("{}/flappy_bird_critic".format(opt.saved_path))
        critic.load_state_dict(checkpoint['net'])
        critic_optimizer.load_state_dict(checkpoint['optimizer'])
        print("load critic succ")

    writer = SummaryWriter(opt.log_path)
    game_state = FlappyBird("ppo")
    state, reward, terminal = game_state.step(0)
    max_reward = 0
    iter = 0
    replay_memory = []
    evaluate_num = 0  # Record the number of evaluations
    evaluate_rewards = []
    while iter < opt.num_iters:
        terminal = False
        episode_return = 0.0

        while not terminal:
            prediction = actor(state)
            action_dist = torch.distributions.Categorical(prediction)
            action_sample = action_dist.sample()
            action = action_sample.item()
            next_state, reward, terminal = game_state.step(action)
            replay_memory.append([state, action, reward, next_state, terminal])
            state = next_state
            episode_return += reward

        if episode_return > max_reward:
            max_reward = episode_return
            print(" max_reward Iteration: {}/{}, Reward: {}".format(iter + 1, opt.num_iters, episode_return))

        if len(replay_memory) > opt.batch_size:
            state_batch, action_batch, reward_batch, next_state_batch, terminal_batch = zip(*replay_memory)
            states = torch.cat(state_batch,dim=0).cuda()
            actions = torch.tensor(action_batch).view(-1, 1).cuda()
            rewards = torch.tensor(reward_batch).view(-1, 1).cuda()
            dones = torch.tensor(terminal_batch).view(-1, 1).int().cuda()
            next_states = torch.cat(next_state_batch,dim=0).cuda()

            with torch.no_grad():
                td_target = rewards + opt.gamma * critic(next_states) * (1 - dones)
                td_delta = td_target - critic(states)
                advantage = compute_advantage(opt.gamma, opt.lmbda,  td_delta.cpu()).cuda()
                old_log_probs = torch.log(actor(states).gather(1, actions)).detach()

            for _ in range(opt.epochs):
                for index in BatchSampler(SubsetRandomSampler(range(opt.batch_size)), opt.mini_batch_size, False):
                    log_probs = torch.log(actor(states[index]).gather(1, actions[index]))
                    ratio = torch.exp(log_probs - old_log_probs[index])
                    surr1 = ratio * advantage[index]
                    surr2 = torch.clamp(ratio, 1 - opt.eps, 1 + opt.eps) * advantage[index]  # 截断
                    actor_loss = torch.mean(-torch.min(surr1, surr2))
                    critic_loss = torch.mean(
                        nn.functional.mse_loss(critic(states[index]), td_target[index].detach()))
                    actor_optimizer.zero_grad()
                    critic_optimizer.zero_grad()
                    actor_loss.backward()
                    critic_loss.backward()
                    actor_optimizer.step()
                    critic_optimizer.step()
            replay_memory = []

        iter += 1
        if (iter+1) % 10 == 0:
            evaluate_num += 1
            evaluate_rewards.append(episode_return)
            print("evaluate_num:{} \t episode_return:{} \t".format(evaluate_num, episode_return))
            writer.add_scalar('step_rewards', evaluate_rewards[-1], global_step= iter)
        if (iter+1) % 1000 == 0:
            actor_dict = {"net": actor.state_dict(), "optimizer": actor_optimizer.state_dict()}
            critic_dict = {"net": critic.state_dict(), "optimizer": critic_optimizer.state_dict()}
            torch.save(actor_dict, "{}/flappy_bird_actor".format(opt.saved_path))
            torch.save(critic_dict, "{}/flappy_bird_critic".format(opt.saved_path))



if __name__ == "__main__":
    opt = get_args()
    train(opt)
