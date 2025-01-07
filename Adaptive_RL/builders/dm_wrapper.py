from .wrappers import ExceptionWrapper


class DMWrapper(ExceptionWrapper):
    """
    Wrapper for general DeepMind ControlSuite environments.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        from dm_control.rl.control import PhysicsError

        self.error = PhysicsError

    def muscle_lengths(self):
        length = self.unwrapped.environment.physics.data.actuator_length
        return length

    def muscle_forces(self):
        return self.unwrapped.environment.physics.data.actuator_force

    def muscle_velocities(self):
        return self.unwrapped.environment.physics.data.actuator_velocity

    def muscle_activity(self):
        return self.unwrapped.environment.physics.data.act

    @property
    def _max_episode_steps(self):
        return self.unwrapped.max_episode_steps
