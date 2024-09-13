import abc


class BaseAgent(abc.ABC):
    """
    Abstract base class used to build agents.
    These are the required methods used to build any agent.
    """

    def initialize(self, observation_space, action_space, seed=None):
        pass

    @abc.abstractmethod
    def step(self, observations, steps):
        """
        Returns actions during training.
        """
        pass

    def update(self, observations, rewards, resets, terminations, steps):
        """
        Informs the agent of the latest transitions during training.
        """
        pass

    @abc.abstractmethod
    def test_step(self, observations):
        """Returns actions during testing."""
        pass

    def test_update(self, observations, rewards, resets, terminations, steps):
        """Informs the agent of the latest transitions during testing."""
        pass

    def save(self, path):
        """Saves the agent weights during training."""
        pass

    def load(self, path):
        """Reloads the agent weights from a checkpoint."""
        pass


def trainable_variables(model):
    return [p for p in model.parameters() if p.requires_grad]
