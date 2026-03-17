# SEAMLESS INTERACTION DATASET

This is a forked version of the code resources that download samples from Meta's Seamless Interaction Dataset. For detailed information, please refer to the [original README](https://github.com/facebookresearch/seamless_interaction/blob/main/README.md).

## Adopted pipeline:

1. Aggregate sample information using [`aggregate_assets`](/scripts/aggregate_assets.py).

2. Subset the data (optional) and create a JSON list of session keys using [`subset_data`](/scripts/subset_data.py).
    - The subset currently implemented in the script is: Naturalistic samples tagged as IPC conversations that have participant relationships and personalities annotation available

3. Pass the JSON list of session keys to `download_session_exploration` in [`download_s3`](/scripts/download_s3.py). 
    - The default script behaviour is to download all interactions from all sessions contained in the list, however it is also possible to download only a fixed number of samples by adjusting `num_sessions` and `interactions_per_session`.

    - If you're only interested in the audio modality, use `include_video = False` to skip video download.

    - Each interaction comes with associated metadata and features:
        - 'metadata': textual transcripts and VAD information
        - 'annotations': first-person and third-person internal state and behavior annotations
        - 'features': movement data and pre-extracted features

        Use `features_to_download` to pass a list with the desired information. E.g., ['metadata'] or ['annotations', 'features'].

4. Preprocess the downloaded data using [`preprocess`](/scripts/preprocess.py). We employ the following steps:
    - 48kHz -> 16kHz resampling
    - Channel mixing: two mono files from each participant in a session become one stereo file. The script creates a `channel_map` associating each channel of the resulting stereo wav to their respective speaker ID.
    - VAD mixing: combine VAD from both channels into one JSON file structured as:
        ```python
        [
            [[0.1, 2.11], [2.5, 3.9], ...], # Channel 1 VAD segments
            [[2.2, 2.45], [4.8, 5.2], ...], # Channel 2 VAD segments
        ]
        ```