import numpy as np


class ReplayBuffer:
    """
    Replay buffer used for off-policy learning calculating n-step returns.
    This class provides functionality to store experiences and compute n-step returns,
    which can be used for off-policy reinforcement learning. It supports efficient batch
    sampling and allows for experience reuse to improve learning stability and convergence.
    """

    def __init__(
        self, size=int(1e6), return_steps=1, batch_iterations=64,
        batch_size=256, discount_factor=0.99, steps_before_batches=int(1e4),
        steps_between_batches=50
    ):
        """
        Initializes the replay buffer.
        Parameters:
            - size (int): Maximum number of experiences to store in the buffer. Default is 1e6.
            - return_steps (int): Number of steps to consider for n-step return calculations. Default is 1.
            - batch_iterations (int): Number of times to sample batches during training. Default is 50.
            - batch_size (int): Number of experiences to sample in each batch. Default is 100.
            - discount_factor (float): Discount factor (gamma) for calculating n-step returns. Default is 0.99.
            - steps_before_batches (int): Minimum number of steps before starting to sample batches. Default is 1e4.
            - steps_between_batches (int): Minimum number of steps between consecutive batch sampling operations. Default is 50.
        """
        self.max_size = None
        self.num_workers = None
        self.full_max_size = size
        self.return_steps = return_steps
        self.batch_iterations = batch_iterations
        self.batch_size = batch_size
        self.discount_factor = discount_factor
        self.steps_before_batches = steps_before_batches
        self.steps_between_batches = steps_between_batches
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
            self.buffers = {}
            for key, val in kwargs.items():
                shape = (self.max_size,) + np.array(val).shape
                self.buffers[key] = np.full(shape, np.nan, np.float32)

        # Store the new values.
        for key, val in kwargs.items():
            self.buffers[key][self.index] = val

        # Accumulate values for n-step returns.
        if self.return_steps > 1:
            self.accumulate_n_steps(kwargs)

        self.index = (self.index + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def accumulate_n_steps(self, kwargs):
        """
        Computes n-step returns and updates stored rewards, discounts, and next observations in the buffer.
        Parameters:
            - kwargs (dict): Contains rewards, next observations, and discounts for the current batch of experiences.
        """
        rewards = kwargs['rewards']
        next_observations = kwargs['next_observations']
        discounts = kwargs['discounts']
        masks = np.ones(self.num_workers, np.float32)

        for i in range(min(self.size, self.return_steps - 1)):
            index = (self.index - i - 1) % self.max_size
            masks *= (1 - self.buffers['resets'][index])
            new_rewards = (self.buffers['rewards'][index] +
                           self.buffers['discounts'][index] * rewards)
            self.buffers['rewards'][index] = (
                (1 - masks) * self.buffers['rewards'][index] +
                masks * new_rewards)
            new_discounts = self.buffers['discounts'][index] * discounts
            self.buffers['discounts'][index] = (
                (1 - masks) * self.buffers['discounts'][index] +
                masks * new_discounts)
            self.buffers['next_observations'][index] = (
                (1 - masks)[:, None] *
                self.buffers['next_observations'][index] +
                masks[:, None] * next_observations)

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
