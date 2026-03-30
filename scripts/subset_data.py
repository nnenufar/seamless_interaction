import pandas as pd
import json

df = pd.read_csv("assets/interaction_aggregated.csv")
total_num_samples = len(df)

df = df.replace("Undisclosed", pd.NA)
df = df.replace("", pd.NA)

df_nat_ipc_rel = df[
                (df["label"] == "naturalistic") &
                (df["interaction_type"] == "ipc_conversation") &
                (df["relationship"].notna())
               ]

natural_num_samples = len(df_nat_ipc_rel)
nat_total_ratio = natural_num_samples / total_num_samples

perso_columns = [
    "participant_1_extraversion_raw",
    "participant_1_agreeableness_raw",
    "participant_1_conscientiousness_raw",
    "participant_1_neuroticism_raw",
    "participant_1_openness_raw",
    "participant_2_extraversion_raw",
    "participant_2_agreeableness_raw",
    "participant_2_conscientiousness_raw",
    "participant_2_neuroticism_raw",
    "participant_2_openness_raw"
]

df_perso = df_nat_ipc_rel[
    df_nat_ipc_rel[perso_columns].notna().all(axis=1)
]

perso_num_samples = len(df_perso)
perso_nat_ratio = perso_num_samples / natural_num_samples
perso_total_ratio = perso_num_samples / total_num_samples

output_data = {
    "naturalisticSamples_RelTotal": f"{nat_total_ratio:.2%}",
    "relationship_detail_counts": df_nat_ipc_rel["relationship_detail"].value_counts().to_dict(),
    "personalitySamples_RelNaturalistic": f"{perso_nat_ratio:.2%}",
    "personalitySamples_RelTotal": f"{perso_total_ratio:.2%}"
}

print(f"Naturalistic conversations: {nat_total_ratio:.2%} of the dataset\n")
print(df_nat_ipc_rel["relationship_detail"].value_counts(), "\n")
print(f"Personality data available: {perso_nat_ratio:.2%} of naturalistic conversations ({perso_total_ratio:.2%} of total dataset)")

with open("assets/subset_stats.json", "w") as f:
    json.dump(output_data, f, indent=2)

sessions_to_keep = (
    "V" + df["vendor_id"].astype(str).str.zfill(2) +
    "_S" + df["session_id"].astype(str).str.zfill(4)
    ).unique().tolist()

json_out_path = "assets/subset_sessions.json"

with open(json_out_path, "w") as f:
    json.dump(sessions_to_keep, f, indent=2)

