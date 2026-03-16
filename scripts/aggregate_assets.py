from pathlib import Path
import pandas as pd
import re

ASSETS = Path(__file__).resolve().parents[1] / "assets"


def parse_file_id(fid):
    m = re.match(r"V(\d+)_S(\d+)_I(\d+)_P(\w+)", str(fid))
    if not m:
        return None, None, None, None
    return m.group(1), m.group(2), m.group(3), m.group(4)


def build_interaction_table():

    filelist = pd.read_csv(ASSETS / "filelist.csv", dtype=str)
    interactions = pd.read_csv(ASSETS / "interactions.csv", dtype=str)
    relationships = pd.read_csv(ASSETS / "relationships.csv", dtype=str)
    relationships["vendor_id"] = relationships["vendor_id"].str.replace("^V", "", regex=True)
    participants = pd.read_csv(ASSETS / "participants.csv", dtype=str)

    # -----------------------------
    # Parse IDs from file_id
    # -----------------------------
    parsed = filelist["file_id"].apply(parse_file_id)
    parsed = pd.DataFrame(parsed.tolist(),
                          columns=["vendor_id", "session_id",
                                   "interaction_id", "participant_id"])

    filelist = pd.concat([filelist, parsed], axis=1)

    keys = ["vendor_id", "session_id", "interaction_id"]

    # -----------------------------
    # Participants per interaction
    # -----------------------------
    part_table = (
        filelist[keys + ["participant_id"]]
        .drop_duplicates()
        .groupby(keys)["participant_id"]
        .apply(lambda x: sorted(set(x)))
        .reset_index(name="participant_ids")
    )

    part_table["participant_1_id"] = part_table["participant_ids"].apply(
        lambda x: x[0] if len(x) > 0 else None
    )

    part_table["participant_2_id"] = part_table["participant_ids"].apply(
        lambda x: x[1] if len(x) > 1 else None
    )

    # -----------------------------
    # Interaction-level stats
    # -----------------------------
    stats = (
        filelist.groupby(keys)
        .agg(
            label=("label", lambda x: "|".join(sorted(set(x.dropna())))),
        )
        .reset_index()
    )

    interaction_df = part_table.merge(stats, on=keys)

    # -----------------------------
    # Join relationship metadata
    # -----------------------------
    interaction_df = interaction_df.merge(
        relationships,
        on=["vendor_id", "session_id"],
        how="left"
    )

    # -----------------------------
    # Join prompt metadata
    # -----------------------------
    interactions = interactions.rename(columns={"prompt_hash": "interaction_id"})

    interaction_df = interaction_df.merge(
        interactions,
        on="interaction_id",
        how="left"
    )

    # -----------------------------
    # Join participant personalities
    # -----------------------------
    p1 = participants.add_prefix("participant_1_").rename(
        columns={
            "participant_1_vendor_id": "vendor_id",
            "participant_1_participant_id": "participant_1_id"
        }
    )

    p2 = participants.add_prefix("participant_2_").rename(
        columns={
            "participant_2_vendor_id": "vendor_id",
            "participant_2_participant_id": "participant_2_id"
        }
    )

    interaction_df = interaction_df.merge(
        p1,
        on=["vendor_id", "participant_1_id"],
        how="left"
    )

    interaction_df = interaction_df.merge(
        p2,
        on=["vendor_id", "participant_2_id"],
        how="left"
    )

    return interaction_df.sort_values(
        ["vendor_id", "session_id", "interaction_id"]
    ).reset_index(drop=True)


def sanitize_for_csv(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize object columns so the output CSV stays parser-friendly."""
    out = df.copy()

    # Expand list-valued columns to a delimiter-safe string representation.
    if "participant_ids" in out.columns:
        out["participant_ids"] = out["participant_ids"].apply(
            lambda x: "|".join(x) if isinstance(x, list) else x
        )

    # Ensure column names are valid strings (some editors crash on non-string names).
    out.columns = [str(c) if c is not None else "" for c in out.columns]
    return out


if __name__ == "__main__":

    df = build_interaction_table()
    df = sanitize_for_csv(df)

    output_path = ASSETS / "interaction_aggregated.csv"
    df.to_csv(output_path, index=False, encoding="utf-8", lineterminator="\n")

    print(f"Saved: {output_path}")
    print(df.head())