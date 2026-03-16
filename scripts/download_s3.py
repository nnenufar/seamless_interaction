# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from seamless_interaction.fs import DatasetConfig, SeamlessInteractionFS
import json

SESSION_KEYS = "assets/sessions.json"

def download_single_example(file_id: str | None = None):
    """
    Download a single interaction example (~50-100MB).

    Perfect for quick exploration and understanding data structure.
    Auto-samples from vendors with smaller files (V00, V01) if no file_id
    provided.

    :param file_id: Specific file ID to download, or None to auto-sample
    """
    config = DatasetConfig(label="naturalistic", split="dev", preferred_vendors_only=True)
    fs = SeamlessInteractionFS(config=config)

    if file_id is None:
        # Auto-sample a random file ID from preferred vendors
        file_ids = fs.sample_random_file_ids(num_samples=1)
        file_id = file_ids[0]
        print(f"🎲 Auto-sampled file ID: {file_id}")

    # Download single interaction
    fs.gather_file_id_data_from_s3(file_id)
    print(f"✅ Downloaded single example: {file_id}")


def download_interaction_pair(interaction_key: str | None = None):
    """
    Download a pair of interactions from the same session (~100-200MB).

    Ideal for studying conversational dynamics between participants.
    Auto-samples interaction pairs if no interaction_key provided.

    :param interaction_key: Interaction key (V00_S0809_I00000582) or None to auto-sample
    """
    config = DatasetConfig(label="improvised", split="dev", preferred_vendors_only=True)
    fs = SeamlessInteractionFS(config=config)

    if interaction_key is None:
        # Auto-sample interaction pairs from preferred vendors
        pairs = fs.get_interaction_pairs(num_pairs=1)
        file_ids = pairs[0]  # Take first pair
        print(f"🎲 Auto-sampled interaction pair: {file_ids}")
    else:
        # Use specific interaction key
        pairs = fs.get_interaction_pairs(interaction_keys=[interaction_key])
        file_ids = pairs[0] if pairs else []
        print(f"📍 Using interaction key: {interaction_key} -> {file_ids}")

    if not file_ids:
        print("❌ No interaction pairs found")
        return

    # Download both participants from same interaction
    fs.download_batch_from_s3(file_ids)
    print(f"✅ Downloaded interaction pair: {file_ids}")


def download_samples_1gb(file_ids: list[str] | None = None, num_samples: int = 10):
    """
    Download approximately 1GB of samples (~10 interactions).

    Good for initial exploration and prototyping.
    Auto-samples diverse interactions if no file_ids provided.

    :param file_ids: Specific file IDs to download, or None to auto-sample
    :param num_samples: Number of samples to download (if auto-sampling)
    """
    config = DatasetConfig(
        label="improvised",
        split="test",
        preferred_vendors_only=True,
        seed=42,  # For reproducible sampling
        num_workers=4,
    )
    fs = SeamlessInteractionFS(config=config)

    if file_ids is None:
        # Auto-sample diverse file IDs from preferred vendors
        file_ids = fs.sample_random_file_ids(num_samples=num_samples)
        print(f"🎲 Auto-sampled {len(file_ids)} file IDs from preferred vendors")
        ids_preview = file_ids[:3] if len(file_ids) > 3 else file_ids
        print(
            f"Sample IDs: {ids_preview}..."
            if len(file_ids) > 3
            else f"Sample IDs: {ids_preview}"
        )

    fs.download_batch_from_s3(file_ids)
    print(f"✅ Downloaded {len(file_ids)} samples (~{len(file_ids) * 100}MB)")


def download_session_exploration(
    session_keys: str | list[str] | None = None, 
    interactions_per_session: int = 0, # All
    num_sessions: int = 0,             # All
    include_video: bool = False,
    features_to_download: list[str] | None = None,
):
    """
    Download complete session groups for deeper exploration (~400MB per session).

    Perfect for studying conversational context and session dynamics.
    Auto-samples sessions with rich interaction content if no session_key provided.

    :param session_key: Session key (V00_S0809) or List of keys or None to auto-sample
    :param interactions_per_session: Target interactions per session. 0 to download all interactions.
    :param include_video: Whether to include video data in the download
    :param features_to_download: List of specific features to download (None for all)
    """
    config = DatasetConfig(
        label="naturalistic", split="dev", preferred_vendors_only=True, num_workers=4,
        local_dir="inspect/samples"
    )
    fs = SeamlessInteractionFS(config=config)

    if session_keys is None:
        # Auto-sample session groups from preferred vendors
        session_groups = fs.get_session_groups(
            num_sessions=num_sessions, interactions_per_session=interactions_per_session
        )
        all_file_ids = session_groups[0] if session_groups else []
        print(f"🎲 Auto-sampled session: {len(all_file_ids)} interactions")
    else:
        # Use specific session key(s)
        if isinstance(session_keys, str):
            session_keys = [session_keys]
            
        session_groups = fs.get_session_groups(
            session_keys=session_keys,
            num_sessions=num_sessions,
            interactions_per_session=interactions_per_session,
        )
        all_file_ids = [file_id for group in session_groups for file_id in group]
        print(
            f"📍 Using session keys: {session_keys} -> {len(all_file_ids)} interactions"
        )

    if not all_file_ids:
        print("❌ No session interactions found")
        return

    fs.download_batch_from_s3(
        all_file_ids, 
        include_video=include_video, 
        features_to_download=features_to_download
    )
    print(f"✅ Downloaded sessions with {len(all_file_ids)} interactions")


def main():
    """
    Demonstrate S3-based flexible download options with intelligent sampling.

    All functions support both manual key specification and automatic sampling.
    Auto-sampling prioritizes smaller vendors (V00, V01).
    """
    print("🔍 S3 Download Options with Intelligent Sampling:")
    print("1. Single example (~100MB) - Quick exploration")
    print("2. Interaction pair (~200MB) - Conversational dynamics")
    print("3. Sample set (~1GB) - Initial prototyping")
    print("4. Session exploration (~400MB/session) - Deep context study")

    # Uncomment desired download scenario:
    # download_single_example()  # Auto-samples if no file_id provided
    # download_single_example("V01_S0223_I00000127_P1505")  # Specific file
    # download_interaction_pair()  # Auto-samples interaction pairs
    # download_interaction_pair("V00_S0809_I00000582")  # Specific interaction
    # download_samples_1gb()  # Auto-samples 10 diverse files
    # download_samples_1gb(num_samples=20)  # Auto-samples 20 files (~2GB)
    # download_session_exploration()  # Auto-samples 1 rich session
    # download_session_exploration("V00_S0809")  # Specific session

    with open(SESSION_KEYS, "r") as f:
        session_keys = json.load(f)
    
    download_session_exploration(session_keys, include_video=False, features_to_download=["metadata"], num_sessions=0,interactions_per_session=0)  # Multiple specific sessions

if __name__ == "__main__":
    main()
