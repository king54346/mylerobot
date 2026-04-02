import datasets
import torch


# TODO(aliberts): remove
def calculate_episode_data_index(hf_dataset: datasets.Dataset) -> dict[str, torch.Tensor]:
    """
    Calculate episode data index for the provided HuggingFace Dataset. Relies on episode_index column of hf_dataset.

    Parameters:
    - hf_dataset (datasets.Dataset): A HuggingFace dataset containing the episode index.

    Returns:
    - episode_data_index: A dictionary containing the data index for each episode. The dictionary has two keys:
        - "from": A tensor containing the starting index of each episode.
        - "to": A tensor containing the ending index of each episode.
    """
    episode_data_index = {"from": [], "to": []}

    current_episode = None
    """
    The episode_index is a list of integers, each representing the episode index of the corresponding example.
    For instance, the following is a valid episode_index:
      [0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 2]

    Below, we iterate through the episode_index and populate the episode_data_index dictionary with the starting and
    ending index of each episode. For the episode_index above, the episode_data_index dictionary will look like this:
        {
            "from": [0, 3, 7],
            "to": [3, 7, 12]
        }
    """
    if len(hf_dataset) == 0:
        episode_data_index = {
            "from": torch.tensor([]),
            "to": torch.tensor([]),
        }
        return episode_data_index
    for idx, episode_idx in enumerate(hf_dataset["episode_index"]):
        if episode_idx != current_episode:
            # We encountered a new episode, so we append its starting location to the "from" list
            episode_data_index["from"].append(idx)
            # If this is not the first episode, we append the ending location of the previous episode to the "to" list
            if current_episode is not None:
                episode_data_index["to"].append(idx)
            # Let's keep track of the current episode index
            current_episode = episode_idx
        else:
            # We are still in the same episode, so there is nothing for us to do here
            pass
    # We have reached the end of the dataset, so we append the ending location of the last episode to the "to" list
    episode_data_index["to"].append(idx + 1)

    for k in ["from", "to"]:
        episode_data_index[k] = torch.tensor(episode_data_index[k])

    return episode_data_index
