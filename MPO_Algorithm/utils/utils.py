import os
from MPO_Algorithm import logger


def load_checkpoint(checkpoint, path):
    logger.log(f'Trying to load experiment from {path}')

    # Use no checkpoint, the agent is freshly created.
    if checkpoint == 'none':
        logger.log('Not loading any weights')
        checkpoint_path = None
    else:
        checkpoint_path = os.path.join(path, 'checkpoints')
        if not os.path.isdir(checkpoint_path):
            logger.error(f'{checkpoint_path} is not a directory, starting from scratch')
            checkpoint_path = None
        else:
            # List all the checkpoints.
            checkpoint_ids = []
            for file in os.listdir(checkpoint_path):
                if file[:5] == 'step_':
                    checkpoint_id = file.split('.')[0]
                    checkpoint_ids.append(int(checkpoint_id[5:]))

            if checkpoint_ids:
                # Use the last checkpoint.
                if checkpoint == 'last':
                    checkpoint_id = max(checkpoint_ids)
                    checkpoint_path = os.path.join(
                        checkpoint_path, f'step_{checkpoint_id}')

                # Use the specified checkpoint.
                else:
                    checkpoint_id = int(checkpoint)
                    if checkpoint_id in checkpoint_ids:
                        checkpoint_path = os.path.join(
                            checkpoint_path, f'step_{checkpoint_id}')
                    else:
                        logger.error(f'Checkpoint {checkpoint_id} '
                                           f'not found in {checkpoint_path}')
                        checkpoint_path = None

            else:
                logger.error(f'No checkpoint found in {checkpoint_path}, starting from scratch')
                checkpoint_path = None
    return checkpoint_path