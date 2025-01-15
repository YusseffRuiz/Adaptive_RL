from .environments import Gym, MyoSuite
from .wrappers import ActionRescaler, TimeFeature, GymWrapper
from .cpg_wrapper import CPGWrapper

def apply_wrapper(env, muscle_flag=False, cpg_flag=False, cpg_model=None, myo_flag=False, direct=False):
    if direct:
        return GymWrapper(env)
    else:
        if myo_flag:
            import myosuite
            env = f"MyoSuite({env})"
        else:
            env = f"Gym({env})"
        if muscle_flag:
            if cpg_flag:
                return GymWrapper(CPGWrapper(eval(env), cpg_model=cpg_model, use_cpg=True))
            else:
                return GymWrapper(eval(env))
        else:
            if cpg_flag:
                return CPGWrapper(eval(env), cpg_model=cpg_model, use_cpg=True)
            else:
                return eval(env)

__all__ = [
    Gym, MyoSuite,
    ActionRescaler, TimeFeature, CPGWrapper]
