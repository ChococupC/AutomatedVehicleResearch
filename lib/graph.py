import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set global Seaborn style
sns.set_theme(style="whitegrid")


def extract_df_from_participant(p_obj) -> pd.DataFrame:
    """Helper to extract a pandas DataFrame from various participant object structures."""
    if hasattr(p_obj, "data") and isinstance(p_obj.data, pd.DataFrame):
        return p_obj.data.copy()
    elif hasattr(p_obj, "file"):
        if hasattr(p_obj.file, "seek"):
            p_obj.file.seek(0)
        return pd.read_csv(p_obj.file)
    elif isinstance(p_obj, pd.DataFrame):
        return p_obj.copy()
    else:
        raise ValueError(f"Unable to extract DataFrame from participant object: {p_obj}")


def process_toc_experiment_data(df: pd.DataFrame, ttc_threshold: float = 1.5) -> pd.DataFrame:
    """
    Processes multi-participant telemetry:
    1. Filters out unknown scenarios (allows only A, B, C, D).
    2. Calculates Manual Driving % relative to the first Sim_End timestep.
    3. Flags prior failure experience if Scenario B/D occurred AND TTC <= 1.5s.
    """
    valid_scenarios = {'A', 'B', 'C', 'D'}
    records = []

    for participant_id, part_df in df.groupby('Participant'):
        run_groups = part_df.groupby('Folder_Num')
        sorted_folders = sorted(run_groups.groups.keys())

        had_prior_failure_exp = False

        for trial_idx, folder in enumerate(sorted_folders):
            group = run_groups.get_group(folder).sort_values('Timestep')

            scenario = group['Scenario_Name'].iloc[0]
            if scenario not in valid_scenarios:
                continue

            # Find timestep where Sim_End is first True
            sim_end_rows = group[group['Sim_End'] == True]
            if not sim_end_rows.empty:
                sim_end_timestep = sim_end_rows['Timestep'].iloc[0]
                valid_group = group[group['Timestep'] <= sim_end_timestep]
            else:
                valid_group = group
                sim_end_timestep = group['Timestep'].max()

            if sim_end_timestep <= 0:
                continue

            # Calculate manual driving percentage
            manual_timesteps = (valid_group['Driver_TO'] == True).sum()
            manual_pct = (manual_timesteps / sim_end_timestep) * 100.0

            # Option 2: Critical TTC Near-Miss Check (0 < TTC <= ttc_threshold)
            valid_ttc = valid_group[valid_group['TTC'] > 0]
            experienced_near_miss = not valid_ttc[valid_ttc['TTC'] <= ttc_threshold].empty
            min_ttc = valid_ttc['TTC'].min() if not valid_ttc.empty else np.nan

            records.append({
                'Participant': participant_id,
                'Folder_Num': folder,
                'Scenario_Name': scenario,
                'Trial_Order': trial_idx + 1,
                'Had_Prior_Failure_Exp': had_prior_failure_exp,
                'Manual_Pct': manual_pct,
                'Near_Miss_Occurred': experienced_near_miss,
                'Min_TTC': min_ttc
            })

            # Flag failure history ONLY if scenario was B or D AND near-miss occurred
            if scenario in ['B', 'D'] and experienced_near_miss:
                had_prior_failure_exp = True

    return pd.DataFrame(records)


def process_takeover_aligned_telemetry(df: pd.DataFrame, window_sec: float = 4.0, fps: float = 10.0) -> pd.DataFrame:
    """Slices frame-by-frame telemetry centered at takeover moment (Driver_TO == True at T = 0)."""
    valid_scenarios = {'A', 'B', 'C', 'D'}
    window_frames = int(window_sec * fps)
    aligned_records = []

    for (participant_id, folder), group in df.groupby(['Participant', 'Folder_Num']):
        group = group.sort_values('Timestep')
        scenario = group['Scenario_Name'].iloc[0]

        if scenario not in valid_scenarios:
            continue

        to_rows = group[group['Driver_TO'] == True]
        if to_rows.empty:
            continue

        to_step = to_rows['Timestep'].iloc[0]
        to_idx = group.index[group['Timestep'] == to_step][0]
        pos_in_group = group.index.get_loc(to_idx)

        start_pos = max(0, pos_in_group - window_frames)
        end_pos = min(len(group), pos_in_group + window_frames)

        sub_df = group.iloc[start_pos:end_pos].copy()
        sub_df['Relative_Time_Sec'] = (sub_df['Timestep'] - to_step) / fps
        sub_df['Participant'] = participant_id
        aligned_records.append(sub_df)

    return pd.concat(aligned_records, ignore_index=True) if aligned_records else pd.DataFrame()


# --- Plotting Functions ---

def plot_manual_pct(summary_df: pd.DataFrame, p_name: str) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(6, 4))

    sns.barplot(
        data=summary_df,
        x='Scenario_Name',
        y='Manual_Pct',
        order=['A', 'B', 'C', 'D'],
        palette='Blues_d',
        capsize=0.1,
        ax=ax
    )

    means = summary_df.groupby('Scenario_Name')['Manual_Pct'].mean()
    for i, sc in enumerate(['A', 'B', 'C', 'D']):
        if sc in means:
            ax.text(i, means[sc] + 1.5, f"{means[sc]:.1f}%", ha='center', va='bottom', fontsize=9, fontweight='bold')

    ax.set_title(f"Manual Control % by Scenario ({p_name})", fontsize=11, fontweight='bold')
    ax.set_xlabel('Scenario Name', fontsize=9)
    ax.set_ylabel('Manual Control Time (%)', fontsize=9)
    ax.set_ylim(0, 105)
    plt.tight_layout()
    return fig


def plot_failure_impact(summary_df: pd.DataFrame, p_name: str) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(6, 4))

    plot_df = summary_df.copy()
    plot_df['Failure_Exp_Label'] = plot_df['Had_Prior_Failure_Exp'].map({
        False: 'No Prior Silent Near-Miss',
        True: r'After Near-Miss (1.5s)'
    })

    sns.barplot(
        data=plot_df,
        x='Scenario_Name',
        y='Manual_Pct',
        hue='Failure_Exp_Label',
        order=['A', 'B', 'C', 'D'],
        palette=['#2b5c8f', '#d95f02'],
        capsize=0.1,
        ax=ax
    )

    ax.set_title(f"Silent Failure Impact ({p_name})", fontsize=11, fontweight='bold')
    ax.set_xlabel('Scenario Name', fontsize=9)
    ax.set_ylabel('Manual Control Time (%)', fontsize=9)
    ax.set_ylim(0, 105)
    ax.legend(title='Prior Experience', fontsize=8, title_fontsize=8, loc='upper left')
    plt.tight_layout()
    return fig


def plot_takeover_trajectory(aligned_df: pd.DataFrame, p_name: str) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(6, 4))

    sns.lineplot(
        data=aligned_df,
        x='Relative_Time_Sec',
        y='Brake_Contr',
        hue='Scenario_Name',
        hue_order=['A', 'B', 'C', 'D'],
        palette='Set1',
        errorbar=None,
        ax=ax
    )

    ax.axvline(0, color='black', linestyle='--', linewidth=1, label='Takeover ($T=0$)')
    ax.set_title(f"Takeover Control Profile ({p_name})", fontsize=11, fontweight='bold')
    ax.set_xlabel('Time Relative to Takeover (s)', fontsize=9)
    ax.set_ylabel('Brake Control Input', fontsize=9)
    ax.legend(title='Scenario', fontsize=8, title_fontsize=8, loc='upper left')
    plt.tight_layout()
    return fig


# --- Main Entry Point for Streamlit ---

def create_participant_graphs(participant_objs) -> dict:
    """
    Generates research figures for each participant.
    Returns a dictionary matching Streamlit requirement:
    { p_name: [ {'id': str, 'title': str, 'fig': matplotlib.figure.Figure}, ... ] }
    """
    dfs = []
    p_names = []

    for p_obj in participant_objs:
        p_name = p_obj.get_name() if hasattr(p_obj, 'get_name') else getattr(p_obj, 'name', 'Unknown')
        p_names.append(p_name)

        df = extract_df_from_participant(p_obj)
        if 'Participant' not in df.columns:
            df['Participant'] = p_name
        dfs.append(df)

    if not dfs:
        return {}

    combined_df = pd.concat(dfs, ignore_index=True)

    # Process metrics
    summary_df = process_toc_experiment_data(combined_df, ttc_threshold=1.5)
    aligned_df = process_takeover_aligned_telemetry(combined_df)

    graphs_dict = {}

    # Overall Participants Graph
    p_graphs = [
        {
            "id": "Overall_manual_pct",
            "title": "Manual Driving % - Overall",
            "fig": plot_manual_pct(summary_df, "Overall")
        },
        {
            "id": "Overall_failure_impact",
            "title": "Silent Failure Impact (TTC <= 1.5s) - Overall",
            "fig": plot_failure_impact(summary_df, "Overall")
        }
    ]

    if not aligned_df.empty:
        p_graphs.append({
            "id": f"Overall_takeover_profile",
            "title": f"Takeover Control Profile - Overall",
            "fig": plot_takeover_trajectory(aligned_df, "Overall")
        })

    graphs_dict["Overall"] = p_graphs

    # for p_name in p_names:
    #     p_summary = summary_df[summary_df['Participant'] == p_name]
    #     p_aligned = aligned_df[aligned_df['Participant'] == p_name] if not aligned_df.empty else pd.DataFrame()
    #
    #     # Use fallback to group summary if single participant data is too small
    #     target_summary = p_summary if len(p_summary) > 0 else summary_df
    #
    #     p_graphs = [
    #         {
    #             # Crucial: ID MUST contain p_name to avoid clearing selected_graph in Streamlit
    #             "id": f"{p_name}_manual_pct",
    #             "title": f"Manual Driving % - {p_name}",
    #             "fig": plot_manual_pct(target_summary, p_name)
    #         },
    #         {
    #             "id": f"{p_name}_failure_impact",
    #             "title": f"Silent Failure Impact (TTC <= 1.5s) - {p_name}",
    #             "fig": plot_failure_impact(target_summary, p_name)
    #         }
    #     ]
    #
    #     if not p_aligned.empty:
    #         p_graphs.append({
    #             "id": f"{p_name}_takeover_profile",
    #             "title": f"Takeover Control Profile - {p_name}",
    #             "fig": plot_takeover_trajectory(p_aligned, p_name)
    #         })
    #
    #     graphs_dict[p_name] = p_graphs

    return graphs_dict