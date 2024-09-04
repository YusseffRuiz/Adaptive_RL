import numpy as np


class ReplayBuffer:
    def __init__(self, matsuoka=False):

        # buffers
        self.start_idx_of_episode = []
        self.idx_to_episode_idx = []
        self.episodes = []
        self.tmp_episode_buff = []
        self.matsuoka = matsuoka

    def store_step(self, state, action, next_state, reward, params):
        if self.matsuoka:
            self.tmp_episode_buff.append((state, action, next_state, reward, params))
        else:
            self.tmp_episode_buff.append((state, action, next_state, reward))

    def done_episode(self):
        if self.matsuoka:
            states, actions, next_states, rewards, dones, params = zip(*self.tmp_episode_buff)
        else:
            states, actions, next_states, rewards = zip(*self.tmp_episode_buff)
        episode_len = len(states)
        usable_episode_len = episode_len - 1
        self.start_idx_of_episode.append(len(self.idx_to_episode_idx))
        self.idx_to_episode_idx.extend([len(self.episodes)] * usable_episode_len)
        if self.matsuoka:
            self.episodes.append((states, actions, next_states, rewards, params))
        else:
            self.episodes.append((states, actions, next_states, rewards))
        self.tmp_episode_buff = []

    def store_episodes(self, episodes):
        for episode in episodes:
            if self.matsuoka:
                states, actions, next_states, rewards, params = zip(*episode)
            else:
                states, actions, next_states, rewards = zip(*episode)
            episode_len = len(states)
            usable_episode_len = episode_len - 1
            self.start_idx_of_episode.append(len(self.idx_to_episode_idx))
            self.idx_to_episode_idx.extend([len(self.episodes)] * usable_episode_len)
            if self.matsuoka:
                self.episodes.append((states, actions, next_states, rewards, params))
            else:
                self.episodes.append((states, actions, next_states, rewards))

    def clear(self):
        self.start_idx_of_episode = []
        self.idx_to_episode_idx = []
        self.episodes = []
        self.tmp_episode_buff = []

    def __getitem__(self, idx):
        episode_idx = self.idx_to_episode_idx[idx]
        start_idx = self.start_idx_of_episode[episode_idx]
        i = idx - start_idx
        if self.matsuoka:
            states, actions, next_states, rewards, params = self.episodes[episode_idx]
            state, action, next_state, reward, params = states[i], actions[i], next_states[i], rewards[i], params[i]
            return state, action, next_state, reward, params
        else:
            states, actions, next_states, rewards = self.episodes[episode_idx]
            state, action, next_state, reward = states[i], actions[i], next_states[i], rewards[i]
            return state, action, next_state, reward

    def __len__(self):
        return len(self.idx_to_episode_idx)

    def mean_reward(self):
        if self.matsuoka:
            _, _, _, rewards, _ = zip(*self.episodes)
        else:
            _, _, _, rewards = zip(*self.episodes)
        return np.mean([np.mean(reward) for reward in rewards])

    def mean_return(self):
        if self.matsuoka:
            _, _, _, rewards, _ = zip(*self.episodes)
        else:
            _, _, _, rewards = zip(*self.episodes)
        return np.mean([np.sum(reward) for reward in rewards])
