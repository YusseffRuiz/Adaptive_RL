from .environments import Gym, MyoSuite
from .wrappers import ActionRescaler, TimeFeature, GymWrapper
from .cpg_wrapper import CPGWrapper

def apply_wrapper(env, muscle_flag=False, cpg_flag=False, cpg_model=None, myo_flag=False, direct=False, separate_flag=False):
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
                output_env = GymWrapper(CPGWrapper(eval(env), cpg_model=cpg_model, use_cpg=True))
                if separate_flag:
                    output_env.remove_action_osl(output_env.params)
                return output_env
            else:
                output_env = GymWrapper(eval(env))
                if separate_flag:
                    output_env.remove_action_osl()
                return output_env
        else:
            if cpg_flag:
                return CPGWrapper(eval(env), cpg_model=cpg_model, use_cpg=True)
            else:
                return eval(env)

__all__ = [
    Gym, MyoSuite,
    ActionRescaler, TimeFeature, CPGWrapper]
