import numpy as np
import torch
import random

class ReplayBuffer:
    """
    Replay buffer used for off-policy learning calculating n-step returns.
    This class provides functionality to store experiences and compute n-step returns,
    which can be used for off-policy reinforcement learning. It supports efficient batch
    sampling and allows for experience reuse to improve learning stability and convergence.
    """

    def __init__(
        self, size=int(1e6), return_steps=3, batch_iterations=30,
        batch_size=100, discount_factor=0.99, steps_before_batches=1e4,
        steps_between_batches=50
    ):
        """
        Initializes the replay buffer.
        Parameters:
            - size (int): Maximum number of experiences to store in the buffer. Default is 1e5.
            - return_steps (int): Number of steps to consider for n-step return calculations. Default is 1.
            - batch_iterations (int): Number of times to sample batches during training. Default is 20.
            - batch_size (int): Number of experiences to sample in each batch. Default is 256.
            - discount_factor (float): Discount factor (gamma) for calculating n-step returns. Default is 0.99.
            - steps_before_batches (int): Minimum number of steps before starting to sample batches. Default is 1e4.
            - steps_between_batches (int): Minimum number of steps between consecutive batch sampling operations.
        """

        self.full_max_size = int(size)
        self.return_steps = return_steps
        self.batch_iterations = batch_iterations
        self.batch_size = batch_size
        self.discount_factor = discount_factor
        self.steps_before_batches = steps_before_batches
        self.steps_between_batches = steps_between_batches

    def initialize(self):
        self.np_random = np.random.RandomState()
        self.buffers = None
        self.index = 0
        self.size = 0
        self.last_steps = 0

    def ready(self, steps):
        """
        Checks if the buffer is ready to provide batches for training.
        Parameters:
            - steps (int): The current number of steps taken by the agent.
        Returns:
            - bool: True if the buffer is ready to provide batches, False otherwise.
        """
        if steps < self.steps_before_batches:
            return False
        return (steps - self.last_steps) >= self.steps_between_batches

    def push(self, **kwargs):
        """
        Stores a new experience in the buffer.
        If 'terminations' is provided in kwargs, it is used to calculate discount factors.
        Parameters:
            - kwargs (dict): Experience data to store (e.g., 'observations', 'actions', 'rewards', etc.).
        """
        if 'terminations' in kwargs:
            continuations = np.float32(1 - kwargs['terminations'])
            kwargs['discounts'] = continuations * self.discount_factor

        # Create the named buffers.
        if self.buffers is None:
            self.num_workers = len(list(kwargs.values())[0])
            self.max_size = self.full_max_size // self.num_workers
            self.buffers = {}  # Initialize buffers as a dictionary
            for key, val in kwargs.items():
                shape = (self.max_size,) + np.array(val).shape
                self.buffers[key] = np.full(shape, np.nan, dtype=np.float32)

        # Store the new values.
        for key, val in kwargs.items():
            self.buffers[key][self.index] = val

        # Accumulate values for n-step returns.
        if self.return_steps > 1:
            self.accumulate_n_steps(kwargs)

        self.index = (self.index + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

        assert self.size <= self.max_size, "Replay buffer size exceeded maximum capacity!"

        if self.size > self.full_max_size:
            self.clean_buffer()

    def accumulate_n_steps(self, kwargs):
        """
        Computes n-step returns and updates stored rewards, discounts, and next observations in the buffer.
        Parameters:
            - kwargs (dict): Contains rewards, next observations, and discounts for the current batch of experiences.
        """
        rewards = kwargs['rewards']
        next_observations = kwargs['next_observations']
        discounts = kwargs['discounts']
        # Convert masks to a torch tensor (same as your buffers)
        masks = np.ones(self.num_workers, dtype=np.float32)

        for i in range(min(self.size, self.return_steps - 1)):
            index = (self.index - i - 1) % self.max_size
            masks *= (1 - self.buffers['resets'][index])

            # Vectorized update for rewards and discounts
            self.buffers['rewards'][index] = (
                    masks * (self.buffers['rewards'][index] + self.buffers['discounts'][index] * rewards) +
                    (1 - masks) * self.buffers['rewards'][index]
            )

            self.buffers['discounts'][index] = (
                    masks * (self.buffers['discounts'][index] * discounts) +
                    (1 - masks) * self.buffers['discounts'][index]
            )

            self.buffers['next_observations'][index] = (
                    masks[:, None] * next_observations +
                    (1 - masks[:, None]) * self.buffers['next_observations'][index]
            )


    def get(self, *keys, steps):
        """
        Get batches from named buffers.
        Parameters:
        - keys (str): The keys representing which data (e.g., 'observations', 'actions', etc.) to include in the batch.
        - steps (int): The current step count to update the last sampled steps.
        Yields:
        - dict: A dictionary containing a batch of experience data.
        """
        for _ in range(self.batch_iterations):
            total_size = self.size * self.num_workers
            indices = self.np_random.randint(total_size, size=self.batch_size)
            rows = indices // self.num_workers
            columns = indices % self.num_workers
            yield {k: self.buffers[k][rows, columns] for k in keys}

        self.last_steps = steps

    def clean_buffer(self):
        """
        Clears the replay buffer to release memory.
        """
        self.buffers = None
        self.index = 0
        self.size = 0
        torch.cuda.empty_cache()  # Clear CUDA memory if needed
        print("Replay buffer cleared.")



class Segment:
    """
    Replay storing recent transitions for on-policy learning.
    """

    def __init__(self, size=4096, batch_iterations=80, batch_size=None, discount_factor=0.99, trace_decay=0.97):
        self.max_size = int(size)
        self.batch_iterations = batch_iterations
        self.batch_size = batch_size
        self.discount_factor = discount_factor
        self.trace_decay = trace_decay

    def initialize(self, seed=None):
        self.np_random = np.random.RandomState(seed)
        self.buffers = None
        self.index = 0

    def ready(self):
        return self.index == self.max_size

    def store(self, **kwargs):
        if self.buffers is None:
            self.num_workers = len(list(kwargs.values())[0])
            self.buffers = {}
            for key, val in kwargs.items():
                shape = (self.max_size,) + np.array(val).shape
                self.buffers[key] = np.zeros(shape, np.float32)
        for key, val in kwargs.items():
            self.buffers[key][self.index] = val
        self.index += 1

    def get_full(self, *keys):
        self.index = 0

        if 'advantages' in keys:
            advs = self.buffers['returns'] - self.buffers['values']
            std = advs.std()
            if std != 0:
                advs = (advs - advs.mean()) / std
            self.buffers['advantages'] = advs

        return {k: flatten_batch(self.buffers[k]) for k in keys}

    def get(self, *keys):
        """
        Get mini-batches from named buffers.
        """
        batch = self.get_full(*keys)

        if self.batch_size is None:
            for _ in range(self.batch_iterations):
                yield batch
        else:
            size = self.max_size * self.num_workers
            all_indices = np.arange(size)
            for _ in range(self.batch_iterations):
                self.np_random.shuffle(all_indices)
                for i in range(0, size, self.batch_size):
                    indices = all_indices[i:i + self.batch_size]
                    yield {k: v[indices] for k, v in batch.items()}

    def compute_returns(self, values, next_values):
        shape = self.buffers['rewards'].shape
        self.buffers['values'] = values.reshape(shape)
        self.buffers['next_values'] = next_values.reshape(shape)
        self.buffers['returns'] = lambda_returns(
            values=self.buffers['values'],
            next_values=self.buffers['next_values'],
            rewards=self.buffers['rewards'],
            resets=self.buffers['resets'],
            terminations=self.buffers['terminations'],
            discount_factor=self.discount_factor,
            trace_decay=self.trace_decay)

    def clean_buffer(self):
        """
        Clears the replay buffer to release memory.
        """
        self.buffers = None
        self.index = 0
        torch.cuda.empty_cache()  # Clear CUDA memory if needed
        print("Replay buffer cleared.")


def lambda_returns(values, next_values, rewards, resets, terminations, discount_factor, trace_decay):
    """
    Function used to calculate lambda-returns on parallel buffers.
    """

    returns = np.zeros_like(values)
    last_returns = next_values[-1]
    for t in reversed(range(len(rewards))):
        bootstrap = (
            (1 - trace_decay) * next_values[t] + trace_decay * last_returns)
        bootstrap *= (1 - resets[t])
        bootstrap += resets[t] * next_values[t]
        bootstrap *= (1 - terminations[t])
        returns[t] = last_returns = rewards[t] + discount_factor * bootstrap
    return returns


def flatten_batch(values):
    shape = values.shape
    new_shape = (np.prod(shape[:2], dtype=int),) + shape[2:]
    return values.reshape(new_shape)