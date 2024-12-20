import os
import time
import numpy as np
import torch
import platform
from playsound import playsound

from Adaptive_RL import logger


class Trainer:
    """Trainer used to train and evaluate an agent on an environment."""

    def __init__(
        self, steps=int(1e7), epoch_steps=int(2e4), save_steps=int(5e5),
        test_episodes=5, show_progress=True, replace_checkpoint=False, early_stopping=False,
    ):
        self.max_steps = steps
        self.epoch_steps = epoch_steps
        self.save_steps = save_steps
        self.test_episodes = test_episodes
        self.show_progress = show_progress
        self.replace_checkpoint = replace_checkpoint # If we want to start from scratch all checkpoints

        self.steps = 0
        self.save_cycles = 1
        self.test_environment = None
        self.environment = None
        self.agent = None

        # Early Stop Parameters
        self.best_reward = -float('inf')
        self.patience = 500  # 500 episodes limit if there is no improvement
        self.no_improvement_counter = 0
        self.early_stopping = early_stopping
        self.decay_counter = 0

    def initialize(self, agent, environment, test_environment=None, step_saved=None):
        self.agent = agent
        self.environment = environment
        self.test_environment = test_environment
        if step_saved is not None:
            self.steps = step_saved

    def dump_trainer(self):
        trainer_data = {
            'max_steps': self.max_steps,
            'epoch_steps': self.epoch_steps,
            'save_steps': self.save_steps,
            'test_episodes': self.test_episodes,
            'early_stopping': self.early_stopping,
        }
        return trainer_data

    def run(self):
        """Runs the main training loop."""

        start_time = last_epoch_time = time.time()

        # Start the environments.
        observations = self.environment.start()
        num_workers = len(observations)
        scores = np.zeros(num_workers)
        lengths = np.zeros(num_workers, int)
        self.steps, epoch_steps, epochs, episodes = self.steps, self.steps, self.steps, self.steps
        steps_since_save = self.steps
        stop_training = False

        while not stop_training:
            # Select actions.
            actions = self.agent.step(observations, self.steps)
            assert not np.isnan(actions.sum())
            logger.store('train/action', actions, stats=True)

            # Take a step in the environments.
            observations, infos = self.environment.step(actions)
            self.agent.update(**infos, steps=self.steps)
            scores += infos['rewards']
            lengths += 1
            self.steps += num_workers
            epoch_steps += num_workers
            steps_since_save += num_workers

            # Show the progress bar.
            if self.show_progress:
                logger.show_progress(self.steps, self.epoch_steps, self.max_steps)

            # Check the finished episodes.
            for i in range(num_workers):
                if infos['resets'][i]:
                    logger.store('train/episode_score', scores[i], stats=True)
                    logger.store('train/episode_length', lengths[i], stats=True)
                    scores[i] = 0
                    lengths[i] = 0
                    episodes += 1

            # End of the epoch.
            if epoch_steps >= self.epoch_steps:
                # Evaluate the agent on the test environment.
                if self.test_environment:
                    self._test()

                # Log the data.
                epochs += 1
                current_time = time.time()
                epoch_time = current_time - last_epoch_time
                # sps = epoch_steps / epoch_time
                logger.store('train/episodes', episodes)
                logger.store('train/epochs', epochs)
                logger.store('train/seconds', current_time - start_time)
                logger.store('train/epoch_seconds', epoch_time)
                logger.store('train/epoch_steps', epoch_steps)
                logger.store('train/steps', self.steps)
                logger.store('train/worker_steps', self.steps // num_workers)
                # logger.store('train/steps_per_second', sps)
                logger.dump()
                last_epoch_time = time.time()
                epoch_steps = 0

            # End of training.
            stop_training = self.steps >= self.max_steps
            # Check if no improvement for 'patience' number of epochs
            if self.no_improvement_counter >= self.patience and self.early_stopping:
                print(f"Early stopping at epoch {epochs}")
                stop_training = True

            # Save a checkpoint.
            if stop_training or steps_since_save >= self.save_steps:
                path = os.path.join(logger.get_path(), 'checkpoints')
                if os.path.isdir(path) and self.replace_checkpoint:
                    for file in os.listdir(path):
                        if file.startswith('step_'):
                            os.remove(os.path.join(path, file))
                checkpoint_name = f'step_{self.steps}'
                save_path = os.path.join(path, checkpoint_name)
                self.agent.save(save_path)
                steps_since_save = self.steps % self.save_steps

                tmp_score = 0
                if self.test_environment:
                    tmp_score = self._test()
                if tmp_score > self.best_reward:
                    self.best_reward = tmp_score
                    self.no_improvement_counter = 0  # Reset counter if there's an improvement
                    play_system_sound(time="best") # Play system sound when a best reward was found
                    # Save the best model.
                    best_model_path = os.path.join(path, 'best_model')
                    self.agent.save(best_model_path)
                    logger.log(f"Best model saved with mean reward {self.best_reward} at epoch {epochs}")
                else:
                    self.no_improvement_counter += 1

                self.save_cycles += 1
                if self.save_cycles % 10 == 0:  # Saving everything only every 10% of the total training
                    self.save_cycles = 1
                    save_model_path = os.path.join(path, "model_checkpoint.pth")
                    self.save_model(self.agent.model, self.agent.actor_updater.optimizer, self.agent.replay_buffer,
                                    save_model_path)

                if self.no_improvement_counter >= self.patience * 0.6:
                    self.agent.decay_flag = True  # start decaying the noise
                if self.agent.decay_flag:
                    self.decay_counter += 1
                    if self.decay_counter >= self.patience * 0.1:
                        self.agent.decay_flag = False
                        self.decay_counter = 0
        play_system_sound(time="end")

    def _test(self):
        """Tests the agent on the test environment."""

        # Start the environment.
        if not hasattr(self, 'test_observations'):
            self.test_observations = self.test_environment.start()
            assert len(self.test_observations) == 1

        # Test loop.
        for _ in range(self.test_episodes):
            score, length = 0, 0

            while True:
                # Select an action.
                actions = self.agent.test_step(self.test_observations)
                assert not np.isnan(actions.sum())
                logger.store('test/action', actions, stats=True)

                # Take a step in the environment.
                self.test_observations, infos = self.test_environment.step(actions)
                self.agent.test_update(**infos, steps=self.steps)

                score += infos['rewards'][0]
                length += 1

                if infos['resets'][0]:
                    break

            # Log the data.
            logger.store('test/episode_score', score, stats=True)
            logger.store('test/episode_length', length, stats=True)
            return score

    @staticmethod
    def save_model(agent, optimizer, replay_buffer, save_path):
        save_data = {
            'model_state_dict': agent.state_dict(),  # Save model weights
            'optimizer_state_dict': optimizer.state_dict(),  # Save optimizer state
            'replay_buffer': replay_buffer,  # Save replay buffer
        }
        # Save the entire state in a single file
        torch.save(save_data, save_path)
        logger.log(f"Model, optimizer, and replay buffer saved at {save_path}")

    @staticmethod
    def load_model(agent, actor_updater, replay_buffer, save_path):
        try:
            # Load the saved data from the file
            save_path = save_path + '/model_checkpoint.pth'
            checkpoint = torch.load(save_path)

            # Load model state
            agent.model.load_state_dict(checkpoint['model_state_dict'])

            # Load optimizer state
            if 'optimizer_state_dict' in checkpoint:
                actor_updater.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

            # Load replay buffer (optional)
            if 'replay_buffer' in checkpoint:
                replay_buffer = checkpoint['replay_buffer']

            logger.log(f"Model, optimizer, and replay buffer loaded from {save_path}")
            return replay_buffer
        except FileNotFoundError:
            print(f"Checkpoint not found at {save_path}")
        except KeyError as e:
            print(f"Key missing in the saved checkpoint: {e}")



# Function to play system notification sound
def play_system_sound(time="best"):
    system = platform.system()
    if time == "best":
        sound = "bell"
    else:
        sound = "complete"
    if system == "Windows":
        import winsound
        winsound.MessageBeep(winsound.MB_ICONASTERISK)  # Windows notification sound
    elif system == "Linux":
        playsound(f'/usr/share/sounds/freedesktop/stereo/{sound}.oga')  # Common system notification sound
    elif system == "Darwin":  # macOS
        playsound('/System/Library/Sounds/Glass.aiff')  # macOS system sound
    else:
        print("Unsupported platform for system sound.")
