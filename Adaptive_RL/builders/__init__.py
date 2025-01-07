from .environments import Gym, MyoSuite
from .wrappers import ActionRescaler, TimeFeature, GymWrapper
from .cpg_wrapper import CPGWrapper
from .dm_wrapper import DMWrapper

def apply_wrapper(env):
    if "control" in str(env).lower():
        return DMWrapper(env)
    else:
        return GymWrapper(env)

__all__ = [
    Gym, MyoSuite,
    ActionRescaler, TimeFeature, CPGWrapper, DMWrapper]
