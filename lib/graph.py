import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set global Seaborn style
sns.set_theme(style="whitegrid")

def extract_df_from_participant(p_obj) -> pd.DataFrame:
    """Helper to extract a pandas DataFrame from various participant object structures."""
    if hasattr(p_obj, "data_frame") and isinstance(p_obj.data_frame, pd.DataFrame):
        return p_obj.data_frame.copy()
    elif hasattr(p_obj, "file"):
        if hasattr(p_obj.file, "seek"):
            p_obj.file.seek(0)
        return pd.read_csv(p_obj.file)
    elif isinstance(p_obj, pd.DataFrame):
        return p_obj.copy()
    else:
        raise ValueError(f"Unable to extract DataFrame from participant object: {p_obj}")


def filter_experiment_data(df: pd.DataFrame, **kwargs) -> pd.DataFrame:
    """
    Find participant whose specific dataframe match the applied filter
    :param df:
    :param kwargs:
    :return:
    """
    conditions = []

    for key, value in kwargs.items():
        if key.endswith('_min'):
            col = key.replace('_min', '')
            conditions.append(f"{col} >= @kwargs['{key}']")
        elif key.endswith('_max'):
            col = key.replace('_max', '')
            conditions.append(f"{col} <= @kwargs['{key}']")
        else:
            conditions.append(f"{key} == @kwargs['{key}']")

    query_str = " & ".join(conditions)
    return df.query(query_str)

def sanitize_scenario_data(df: pd.DataFrame, scenario: str = "ABCD") -> pd.DataFrame:
    df_filtered = df[
        df["Scenario_Name"].isin(list(scenario.upper()))
        # & ~df["Sim_End"]
    ]
    return df_filtered

# Process Experiment data
def process_toc_experiment_data(df: pd.DataFrame, ttc_threshold: float = 5) -> pd.DataFrame:
    """
    Processes multi-participant telemetry for TOC plot:
    1. Filters out unknown scenarios (allows only A, B, C, D).
    2. Calculates Manual Driving % relative to the first Sim_End timestep.
    3. Flags prior failure experience if Scenario B/D occurred AND TTC Threshold Triggered.
    """
    df_filtered = sanitize_scenario_data(df)
    scenarios_independent_variables = {
        'A' : {"Warning": True, "Traffic_Level": "Low"},
        'B': {"Warning": False, "Traffic_Level": "Low"},
        'C': {"Warning": True, "Traffic_Level": "High"},
        'D': {"Warning": False, "Traffic_Level": "High"},
    }
    records = []

    for participant_id, part_df in df_filtered.groupby('Participant'):
        run_groups = part_df.groupby('Folder_Num')
        sorted_folders = sorted(run_groups.groups.keys())

        had_prior_failure_exp = False

        for trial_idx, folder in enumerate(sorted_folders[3:]):
            group = run_groups.get_group(folder).sort_values('Timestep')
            initial_row = group.iloc[0]

            scenario = initial_row['Scenario_Name']

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

            # Critical TTC Near-Miss Check (0 < TTC <= ttc_threshold)
            tailgate_onset = initial_row['Tailgate_Onset']
            valid_ttc = filter_experiment_data(valid_group,Timestep_min=tailgate_onset)
            experienced_near_miss = not valid_ttc[valid_ttc['TTC'] <= ttc_threshold].empty
            min_ttc = valid_ttc['TTC'].min() if not valid_ttc.empty else np.nan

            # Create new record
            records.append({
                'Participant': participant_id,
                'Folder_Num': folder,
                'Scenario_Name': scenario,
                'Warning': scenarios_independent_variables.get(scenario)['Warning'],
                'Traffic_Level': scenarios_independent_variables.get(scenario)['Traffic_Level'],
                'Trial_Order': trial_idx + 1,
                'Had_Prior_Failure_Exp': had_prior_failure_exp,
                'Manual_Pct': manual_pct,
                'Near_Miss_Occurred': experienced_near_miss,
                'Min_TTC': min_ttc
            })

            # Flag failure if scenario was B or D
            if scenario in ['B', 'D']:
                had_prior_failure_exp = True

    return pd.DataFrame(records)


def process_brake_aligned_telemetry(
    df: pd.DataFrame,
    pre_window_sec: float = 4.0,
    post_window_sec: float = 4.0,
    fps: float = 20.0,
    brake_threshold: float = 0.05,
    min_brake_gap_sec: float = 3.0
) -> pd.DataFrame:

    df_filtered = sanitize_scenario_data(df)

    pre_frames = int(pre_window_sec * fps)
    post_frames = int(post_window_sec * fps)

    aligned_records = []

    for (participant_id, folder), group in df_filtered.groupby(
        ['Participant', 'Folder_Num']
    ):
        group = group.sort_values('Timestep').reset_index(drop=True)

        # Make sure there is exactly one scenario
        scenarios = group['Scenario_Name'].dropna().unique()

        if len(scenarios) != 1:
            continue

        scenario = scenarios[0]

        # Find Tailgate Onset
        tailgate_positions = group.index[
            group['Tailgate_Onset'].notna()
        ]

        if len(tailgate_positions) == 0:
            continue

        tailgate_pos = tailgate_positions[0]
        tailgate_step = group.loc[tailgate_pos, 'Timestep']

        # Filter out all brake with min_brake_gap_sec apart
        is_braking = group['Brake_Contr'] >= brake_threshold

        candidate_positions = group.index[
            (group['Timestep'] >= tailgate_step) &
            is_braking &
            (~is_braking.shift(1, fill_value=False))
        ]

        brake_positions = []

        last_brake_step = None

        for brake_pos in candidate_positions:

            brake_step = group.loc[brake_pos, 'Timestep']

            if last_brake_step is None:
                # Always accept the first brake
                brake_positions.append(brake_pos)
                last_brake_step = brake_step

            elif (brake_step - last_brake_step) / fps >= min_brake_gap_sec:
                # Accept only if enough time has passed
                brake_positions.append(brake_pos)
                last_brake_step = brake_step

        # Extract window around EVERY accepted brake event

        for brake_pos in brake_positions:

            brake_step = group.loc[brake_pos, 'Timestep']

            try:
                # Require complete window
                if brake_pos < pre_frames:
                    raise ValueError("miss pre_frame")

                if brake_pos + post_frames >= len(group):
                    raise ValueError("miss post_frame", brake_pos + post_frames)
            except ValueError as e:
                print(f"Skipping {participant_id} at {brake_step} in {scenario}, brake_pos exceed window frame: {e}")
                continue

            # Extract window
            sub_df = group.iloc[
                brake_pos - pre_frames:
                brake_pos + post_frames + 1
            ].copy()

            # Align brake onset to T = 0
            sub_df['Relative_Time_Sec'] = (
                sub_df['Timestep'] - brake_step
            ) / fps

            # Identifiers
            sub_df['Participant'] = participant_id
            sub_df['Brake_Event_Step'] = brake_step
            sub_df['Scenario_Name'] = scenario

            sub_df['Brake_Event_ID'] = (
                f"{participant_id}_{folder}_{brake_step}"
            )

            aligned_records.append(sub_df)

    if not aligned_records:
        return pd.DataFrame()

    return pd.concat(
        aligned_records,
        ignore_index=True
    )


def process_ttc_data(
    df: pd.DataFrame, scenario: str = "A", default_ttc: int = 5,
) -> pd.DataFrame:
    # Filter scenario using existing sanitizer
    filtered_df = sanitize_scenario_data(df, scenario).copy()

    # Replace 1000 default values with NaN between the observed TTC
    valid_sorted = filtered_df[filtered_df["TTC"] != 1000]

    if not valid_sorted.empty:
        t_start = valid_sorted.iloc[0]["Timestep"]
        t_end = valid_sorted.iloc[-1]["Timestep"]

        time_cond = filtered_df["Timestep"].between(t_start, t_end)
        ttc_cond = filtered_df["TTC"] == 1000

        filtered_df.loc[time_cond & ttc_cond, "TTC"] = np.nan

    # Clean df to needed columns and sort in timestep:
    processed_df = (
        filtered_df[
            ["Timestep", "Scenario_Name", "TTC", "Driver_TO", "Warning", "Collision"]
        ]
        .sort_values("Timestep")
    )

    # Interpolate missing TTC values with linear
    processed_df["TTC"] = processed_df["TTC"].interpolate(
        method="linear", limit_direction="forward"
    )
    # Fill remaining Null Values (Usually in the front) to a default_ttc
    processed_df["TTC"] = processed_df["TTC"].replace(1000, default_ttc)

    if df["Participant"].iloc[0] == "P72":
        processed_df.to_csv("./chat.csv")
        print("FOund")
    return processed_df


def plot_ttc_timestep(
    processed_df: pd.DataFrame,
    scenario: str,
    true_color: str = "red",
    false_color: str = "blue",
    show_collision: bool = True,
    show_warning: bool = True,
    plot_mode: str = "individual",  # "individual", "ci", or "iqr"
) -> plt.Figure:

    fig, ax = plt.subplots(figsize=(6, 4))

    color_map = {
        True: true_color,
        False: false_color,
    }

    # ============================================================
    # 1. Plot TTC
    # ============================================================

    if plot_mode == "individual":

        participant_amount = 0

        for participant, participant_df in processed_df.groupby("Participant"):
            participant_amount += 1

            participant_df = participant_df.copy()
            participant_df = process_ttc_data(
                participant_df,
                scenario=scenario
            )

            # Identify contiguous Driver_TO blocks
            change_mask = (
                participant_df["Driver_TO"]
                != participant_df["Driver_TO"].shift()
            )

            participant_df["block_id"] = change_mask.cumsum()

            blocks = list(
                participant_df.groupby("block_id")
            )

            for i, (_, group) in enumerate(blocks):

                # Connect to first point of next block
                if i < len(blocks) - 1:
                    next_first_row = blocks[i + 1][1].iloc[[0]]
                    segment_df = pd.concat(
                        [group, next_first_row]
                    )
                else:
                    segment_df = group

                driver_to_val = group["Driver_TO"].iloc[0]

                ax.plot(
                    segment_df["Timestep"],
                    segment_df["TTC"],
                    color=color_map[driver_to_val],
                    linewidth=1,
                    alpha=0.5,
                )

    elif plot_mode == "ci":

        # Process each participant first
        participant_data = []

        for participant, participant_df in processed_df.groupby("Participant"):

            participant_df = participant_df.copy()
            participant_df = process_ttc_data(
                participant_df,
                scenario=scenario
            )

            participant_data.append(participant_df)

        # Put all participant data back together
        ci_df = pd.concat(participant_data)

        # --------------------------------------------------------
        # Calculate mean + 95% CI at each timestep
        # --------------------------------------------------------
        summary = (
            ci_df
            .groupby(["Timestep"])["TTC"]
            .agg(
                mean="mean",
                std="std",
                n="count",
            )
            .reset_index()
        )
        print(summary)
        # Standard error
        summary["sem"] = (
            summary["std"] / np.sqrt(summary["n"])
        )

        # 95% CI using normal approximation
        summary["ci"] = 1.96 * summary["sem"]

        summary["lower"] = (
            summary["mean"] - summary["ci"]
        )

        summary["upper"] = (
            summary["mean"] + summary["ci"]
        )

        # --------------------------------------------------------
        # Plot each Driver_TO state
        # --------------------------------------------------------


        group = summary.sort_values("Timestep")

        color = color_map[0]

        ax.plot(
            group["Timestep"],
            group["mean"],
            color=color,
            linewidth=2,
            label="label",
        )

        ax.fill_between(
            group["Timestep"],
            group["lower"],
            group["upper"],
            color=color,
            alpha=0.2,
        )

    elif plot_mode == "iqr":
        # Process each participant first
        participant_data = []

        for participant, participant_df in processed_df.groupby("Participant"):
            participant_df = participant_df.copy()
            participant_df = process_ttc_data(
                participant_df,
                scenario=scenario
            )

            participant_data.append(participant_df)

        # Put all participant data back together
        iqr_df = pd.concat(participant_data)

        summary = (
            iqr_df
            .groupby(["Timestep"])["TTC"]
            .agg(
                median="median",
                q1=lambda x: x.quantile(0.025),
                q3=lambda x: x.quantile(0.975),
            )
            .reset_index()
        )

        summary["iqr"] = summary["q3"] - summary["q1"]

        summary["lower"] = summary["q1"] - 1.5 * summary["iqr"]
        summary["upper"] = summary["q3"] + 1.5 * summary["iqr"]

        group = summary.sort_values("Timestep")

        color = color_map[0]

        ax.plot(
            group["Timestep"],
            group["median"],
            color=color,
            linewidth=2,
            label="label",
        )

        ax.fill_between(
            group["Timestep"],
            group["lower"],
            group["upper"],
            color=color,
            alpha=0.2,
        )

    else:
        raise ValueError("Plot mode must be standard, ci, or iqr")
    # ============================================================
    # 2. Threshold & event reference lines
    # ============================================================

    ax.axhline(
        y=5,
        color="yellow",
        linestyle="--",
        linewidth=1,
        label="5s TTC Threshold",
    )

    tailgate_onset = 3080

    if scenario in ["C", "D"]:
        tailgate_onset = 2620

    ax.axvline(
        x=tailgate_onset,
        color="purple",
        linestyle="--",
        linewidth=2,
        label="Tailgate-Trigger",
    )

    # ============================================================
    # 3. Warning
    # ============================================================

    if show_warning:

        warning_df = processed_df.sort_values(
            ["Participant", "Timestep"]
        )

        warning_rising = (
            warning_df["Warning"].eq(True)
            & warning_df.groupby("Participant")["Warning"]
            .shift(1)
            .eq(False)
        )

        warning_timesteps = warning_df.loc[
            warning_rising,
            "Timestep"
        ].unique()

        for i, timestep in enumerate(warning_timesteps):

            ax.axvline(
                x=timestep,
                color="orange",
                linestyle=":",
                linewidth=1.5,
                label=(
                    "Warning Trigger"
                    if i == 0
                    else None
                ),
            )

    # ============================================================
    # 4. Collision
    # ============================================================

    if show_collision:

        collision_df = processed_df.sort_values(
            ["Participant", "Timestep"]
        )

        collision_rising = (
            collision_df["Collision"].eq(True)
            & collision_df.groupby("Participant")["Collision"]
            .shift(1)
            .eq(False)
        )

        collision_timesteps = collision_df.loc[
            collision_rising,
            "Timestep"
        ].unique()

        for i, timestep in enumerate(collision_timesteps):

            ax.axvline(
                x=timestep,
                color="black",
                linestyle="-",
                linewidth=2,
                label=(
                    "First Collision"
                    if i == 0
                    else None
                ),
            )

    # ============================================================
    # 5. Deduplicate legend
    # ============================================================

    handles, labels = ax.get_legend_handles_labels()

    by_label = dict(zip(labels, handles))

    ax.legend(
        by_label.values(),
        by_label.keys(),
        loc="best",
        fontsize=8,
    )

    # ============================================================
    # 6. Title
    # ============================================================

    if plot_mode == "individual":
        participant_amount = processed_df["Participant"].nunique()

        participant_label = (
            f"{participant_amount} participants"
            if participant_amount > 1
            else str(processed_df["Participant"].iloc[0])
        )

        title = (
            f"Scenario {scenario} Continuous TTC "
            f"({participant_label})"
        )

    elif plot_mode == "ci":
        participant_amount = (
            processed_df["Participant"].nunique()
        )

        title = (
            f"Scenario {scenario} Continuous TTC "
            f"(Mean ± 95% Confidence Interval, "
            f"{participant_amount} participants)"
        )
    else:
        participant_amount = (
            processed_df["Participant"].nunique()
        )

        title = (
            f"Scenario {scenario} Continuous TTC "
            f"(Median ± 1.5 IQR, "
            f"{participant_amount} participants)"
        )

    ax.set_title(
        title,
        fontsize=11,
        fontweight="bold",
    )

    ax.set_xlabel(
        "Simulation Time",
        fontsize=9,
    )

    ax.set_ylabel(
        "TTC",
        fontsize=9,
    )

    return fig


def plot_ttc_first_warning_aligned_ci(
    processed_df: pd.DataFrame,
    scenario: str,
    pre_window_sec: float = 8.0,
    post_window_sec: float = 4.0,
    fps: float = 20.0,
    ttc_threshold: float = 5,
) -> plt.Figure:

    fig, ax = plt.subplots(figsize=(6, 4))

    pre_frames = int(pre_window_sec * fps)
    post_frames = int(post_window_sec * fps)

    aligned_records = []

    # ============================================================
    # 1. Process each participant + folder
    # ============================================================

    for (participant, p_scenario), group in processed_df.groupby(
            ["Participant", "Scenario_Name"]
    ):

        group = group.copy()

        print(f"\n--- Participant {participant}, Scenario {p_scenario} ---")

        # ------------------------------------------------------------
        # Scenario check
        # ------------------------------------------------------------

        if p_scenario != scenario:
            print(f"SKIP: wrong scenario ({p_scenario})")
            continue

        # ------------------------------------------------------------
        # Scenario onset
        # ------------------------------------------------------------

        scenario_onset_step = 3080

        if scenario in ["C", "D"]:
            scenario_onset_step = 2620

        print(
            f"Scenario onset: {scenario_onset_step}"
        )

        # ------------------------------------------------------------
        # Process TTC (Initial Check)
        # ------------------------------------------------------------

        below_threshold = group["TTC"] < ttc_threshold
        warning_rows = group[below_threshold]

        if not warning_rows.empty:
            closest_warning_index = (warning_rows["Timestep"] - scenario_onset_step).abs().idxmin()
            target_warning_step = group.loc[closest_warning_index, "Timestep"]
            if scenario_onset_step - target_warning_step >= 10 * fps:
                closest_warning_index = None
                target_warning_step = None
        else:
            closest_warning_index = None
            target_warning_step = None

        # ------------------------------------------------------------
        # No warning
        # ------------------------------------------------------------

        if closest_warning_index is None:

            print(
                "SKIP: NO warning crossing near scenario onset"
            )

            # Useful diagnostic
            after_onset = group[
                (group["Timestep"] >= scenario_onset_step) |
                (group["Warning"] == True)
                ]

            if not after_onset.empty:
                print(
                    f"TTC after onset: "
                    f"min={after_onset['TTC'].min()}, "
                    f"max={after_onset['TTC'].max()}"
                )

            continue

        original_len = len(group)

        # ------------------------------------------------------------
        # Data Cleaning & Reset Index
        # ------------------------------------------------------------

        group = process_ttc_data(
            group,
            scenario=scenario
        )

        print(f"Rows before process_ttc_data: {original_len}")
        print(f"Rows after process_ttc_data: {len(group)}")

        if group.empty:
            print("SKIP: process_ttc_data returned empty")
            continue

        group = (
            group
            .sort_values("Timestep")
            .reset_index(drop=True)
        )

        # ------------------------------------------------------------
        # Check data actually reaches scenario onset
        # ------------------------------------------------------------

        min_step = group["Timestep"].min()
        max_step = group["Timestep"].max()

        print(f"Timestep range: {min_step} → {max_step}")

        if max_step < scenario_onset_step:
            print("SKIP: participant data ends BEFORE scenario onset")
            continue

        # ------------------------------------------------------------
        # First warning (Relocate after index reset)
        # ------------------------------------------------------------

        matching_indices = group.index[group["Timestep"] == target_warning_step]

        if matching_indices.empty:
            print(f"SKIP: The closest warning step ({target_warning_step}) was removed by process_ttc_data")
            continue

        warning_pos = matching_indices[0]
        warning_step = target_warning_step

        print(
            f"First warning: timestep={warning_step}, "
            f"position/iloc={warning_pos}"
        )

        # ------------------------------------------------------------
        # Window checks
        # ------------------------------------------------------------

        print(
            f"Required window: "
            f"{pre_frames} frames before + "
            f"{post_frames} frames after"
        )

        if warning_pos < pre_frames:
            print(f"SKIP: insufficient PRE window ({warning_pos} < {pre_frames})")
            continue

        if warning_pos + post_frames >= len(group):
            print(f"SKIP: insufficient POST window ({warning_pos + post_frames} >= {len(group)})")
            continue

        # ------------------------------------------------------------
        # Extract window
        # ------------------------------------------------------------

        sub_df = group.iloc[
                 warning_pos - pre_frames:
                 warning_pos + post_frames + 1
                 ].copy()

        sub_df["Relative_Time_Sec"] = (
                                              sub_df["Timestep"] - warning_step
                                      ) / fps

        sub_df["Participant"] = participant
        sub_df["Scenario_Name"] = scenario
        sub_df["Scenario_Onset_Step"] = scenario_onset_step
        sub_df["First_Warning_Step"] = warning_step

        sub_df["First_Warning_Event_ID"] = (
            f"{participant}_{scenario}_{warning_step}"
        )

        aligned_records.append(sub_df)

        print("ACCEPTED")

    # ============================================================
    # 8. Check whether valid data exists
    # ============================================================

    if not aligned_records:
        raise ValueError(
            f"No valid first-warning events found "
            f"for Scenario {scenario}."
        )

    aligned_df = pd.concat(
        aligned_records,
        ignore_index=True
    )

    participant_count = (
        aligned_df["Participant"].nunique()
    )

    # ============================================================
    # 9. Calculate mean + 95% CI
    # ============================================================

    summary = (
        aligned_df
        .groupby("Relative_Time_Sec")["TTC"]
        .agg(
            mean="mean",
            std="std",
            n="count",
        )
        .reset_index()
    )

    summary["sem"] = (
        summary["std"] /
        np.sqrt(summary["n"])
    )

    summary["ci"] = (
        1.96 * summary["sem"]
    )

    summary["lower"] = (
        summary["mean"] -
        summary["ci"]
    )

    summary["upper"] = (
        summary["mean"] +
        summary["ci"]
    )

    summary = summary.sort_values(
        "Relative_Time_Sec"
    )

    # ============================================================
    # 10. Plot mean TTC
    # ============================================================

    SCENARIO_COLORS = {
        "A": "#1F77B4",
        "B": "#FF7F0E",
        "C": "#2CA02C",
        "D": "#D62728"
    }

    current_color = SCENARIO_COLORS[scenario]

    ax.plot(
        summary["Relative_Time_Sec"],
        summary["mean"],
        linewidth=2,
        label="Mean TTC",
        color=current_color,
    )

    # ============================================================
    # 11. Plot 95% CI
    # ============================================================

    ax.fill_between(
        summary["Relative_Time_Sec"],
        summary["lower"],
        summary["upper"],
        alpha=0.2,
        label="95% CI",
        color=current_color,
    )

    # ============================================================
    # 12. TTC threshold
    # ============================================================

    ax.axhline(
        y=ttc_threshold,
        linestyle="--",
        linewidth=1,
        label=f"{ttc_threshold}s TTC Threshold",
    )

    # ============================================================
    # 13. First warning trigger
    # ============================================================

    ax.axvline(
        x=0,
        linestyle=":",
        linewidth=2,
        label="First Warning Trigger",
    )

    # ============================================================
    # 14. Formatting
    # ============================================================

    ax.set_title(
        f"Scenario {scenario} First-Warning TTC "
        f"(Mean ± 95% CI, "
        f"{participant_count} participants)",
        fontsize=11,
        fontweight="bold",
    )

    ax.set_xlabel(
        "Time Relative to First Warning (s)",
        fontsize=9,
    )

    ax.set_ylabel(
        "TTC (s)",
        fontsize=9,
    )

    ax.legend(
        loc="best",
        fontsize=8,
    )

    ax.grid(alpha=0.2)

    return fig

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
        True: r'After Near-Miss'
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


def plot_takeover_trajectory(
    aligned_df: pd.DataFrame,
    p_name: str
) -> list[plt.Figure]:

    scenarios = ['A', 'B', 'C', 'D']
    figures = []

    palette = sns.color_palette('Set1', n_colors=4)
    color_dict = dict(zip(scenarios, palette))

    for scenario in scenarios:

        df_sub = aligned_df[
            aligned_df['Scenario_Name'] == scenario
        ].copy()

        if df_sub.empty:
            continue

        fig, ax = plt.subplots(figsize=(6, 4))

        # --------------------------------------------------
        # Calculate mean + central 66% interval
        # --------------------------------------------------

        summary = (
            df_sub
            .groupby('Relative_Time_Sec')
            .agg(
                mean=('Brake_Contr', 'mean'),
                lower=('Brake_Contr',
                       lambda x: x.quantile(0.17)),
                upper=('Brake_Contr',
                       lambda x: x.quantile(0.83))
            )
            .reset_index()
            .sort_values('Relative_Time_Sec')
        )

        x = summary['Relative_Time_Sec']
        mean = summary['mean']
        lower = summary['lower']
        upper = summary['upper']

        color = color_dict[scenario]

        # --------------------------------------------------
        # 66% interval
        # --------------------------------------------------

        ax.fill_between(
            x,
            lower,
            upper,
            color=color,
            alpha=0.20,
            label='Central 66%'
        )

        # --------------------------------------------------
        # Mean trajectory
        # --------------------------------------------------

        ax.plot(
            x,
            mean,
            color=color,
            linewidth=2.5,
            label='Mean'
        )

        # --------------------------------------------------
        # Brake onset
        # --------------------------------------------------

        ax.axvline(
            0,
            color='black',
            linestyle='--',
            linewidth=1.2,
            label='Brake onset ($T=0$)'
        )

        # --------------------------------------------------
        # Number of brake events
        # --------------------------------------------------

        n_events = df_sub['Brake_Event_ID'].nunique()

        ax.text(
            0.98,
            0.95,
            f'n = {n_events} brake events',
            transform=ax.transAxes,
            ha='right',
            va='top',
            fontsize=9
        )

        # --------------------------------------------------
        # Formatting
        # --------------------------------------------------

        ax.set_title(
            f'Scenario {scenario}',
            fontsize=11,
            fontweight='bold'
        )

        ax.set_xlabel(
            'Time Relative to Brake Onset (s)',
            fontsize=9
        )

        ax.set_ylabel(
            'Brake Control Input',
            fontsize=9
        )

        ax.set_xlim(
            x.min(),
            x.max()
        )

        ax.set_ylim(
            0,
            1.05
        )

        ax.grid(
            True,
            alpha=0.25
        )

        ax.legend(
            fontsize=8,
            loc='upper left'
        )

        fig.suptitle(
            f'Brake Control Profile — {p_name}',
            fontsize=12,
            fontweight='bold'
        )

        plt.tight_layout()

        figures.append(fig)

    return figures

def plot_takeover_trajectory_overall(
    aligned_df: pd.DataFrame,
    p_name: str
) -> plt.Figure:

    scenarios = ['A', 'B', 'C', 'D']

    fig, ax = plt.subplots(figsize=(8, 5))

    # --------------------------------------------------
    # 1. Count brake events for each scenario
    # --------------------------------------------------
    event_counts = (
        aligned_df
        .groupby('Scenario_Name')['Brake_Event_ID']
        .nunique()
        .reindex(scenarios)
        .fillna(0)
    )

    # Normalize event counts from 0 → 1
    min_count = event_counts.min()
    max_count = event_counts.max()

    if max_count == min_count:
        normalized_counts = pd.Series(
            0.7,
            index=event_counts.index
        )
    else:
        normalized_counts = (
            (event_counts - min_count)
            / (max_count - min_count)
        )

    # --------------------------------------------------
    # 2. Create colors based on amount of data
    # --------------------------------------------------
    cmap = plt.cm.coolwarm

    color_dict = {
        scenario: cmap(0.35 + 0.55 * normalized_counts[scenario])
        for scenario in scenarios
    }

    # --------------------------------------------------
    # 3. Plot each scenario
    # --------------------------------------------------
    for scenario in scenarios:

        df_sub = aligned_df[
            aligned_df['Scenario_Name'] == scenario
        ].copy()

        if df_sub.empty:
            continue

        summary = (
            df_sub
            .groupby('Relative_Time_Sec')
            .agg(
                mean=('Brake_Contr', 'mean'),
                lower=('Brake_Contr',
                       lambda x: x.quantile(0.17)),
                upper=('Brake_Contr',
                       lambda x: x.quantile(0.83))
            )
            .reset_index()
            .sort_values('Relative_Time_Sec')
        )

        x = summary['Relative_Time_Sec']
        mean = summary['mean']
        lower = summary['lower']
        upper = summary['upper']

        color = color_dict[scenario]
        n_events = event_counts[scenario]

        # Central 66% interval boundaries
        ax.plot(
            x,
            lower,
            color=color,
            linestyle=':',
            linewidth=1.5,
            alpha=0.8
        )

        ax.plot(
            x,
            upper,
            color=color,
            linestyle=':',
            linewidth=1.5,
            alpha=0.8
        )

        # Mean trajectory
        ax.plot(
            x,
            mean,
            color=color,
            linewidth=2.5,
            label=f'Scenario {scenario} (n={int(n_events)})'
        )

    # --------------------------------------------------
    # 4. Takeover / alignment point
    # --------------------------------------------------
    ax.axvline(
        0,
        color='black',
        linestyle='--',
        linewidth=1.2,
        label='Brake onset ($T=0$)'
    )

    # --------------------------------------------------
    # 5. Formatting
    # --------------------------------------------------
    ax.set_title(
        f'Brake Control Profiles — {p_name}',
        fontsize=12,
        fontweight='bold'
    )

    ax.set_xlabel(
        'Time Relative to Brake Onset (s)',
        fontsize=10
    )

    ax.set_ylabel(
        'Brake Control Input',
        fontsize=10
    )

    ax.set_xlim(
        aligned_df['Relative_Time_Sec'].min(),
        aligned_df['Relative_Time_Sec'].max()
    )

    ax.set_ylim(0, 1.05)

    ax.grid(
        True,
        alpha=0.25
    )

    ax.legend(
        fontsize=9,
        loc='upper left'
    )

    plt.tight_layout()

    return fig

def regression_plot(
    df: pd.DataFrame,
    x: str,
    y: str,
) -> plt.Figure:
    """
    Create a scatter plot with a linear regression line.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    x : str
        Column name for the x-axis.
    y : str
        Column name for the y-axis.

    Returns
    -------
    plt.Figure
        The matplotlib figure.
    """

    # Remove missing values
    data = df[[x, y]].dropna()

    # Create figure
    fig, ax = plt.subplots(figsize=(8, 6))

    # Scatter points + regression line
    sns.regplot(
        data=data,
        x=x,
        y=y,
        ax=ax,
        scatter_kws={"alpha": 0.6},
        line_kws={"linewidth": 2},
    )

    # Labels
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.set_title(f"{y} vs. {x}")

    fig.tight_layout()

    return fig

def mean_ci_plot(
    df: pd.DataFrame,
    x: str,
    y: str,
) -> plt.Figure:

    fig, ax = plt.subplots(figsize=(8, 6))

    sns.lineplot(
        data=df,
        x=x,
        y=y,
        estimator="mean",
        errorbar=("ci", 95),
        marker="o",
        ax=ax,
    )

    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.set_title(f"Mean {y} by {x}")

    fig.tight_layout()

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
    aligned_df = process_brake_aligned_telemetry(combined_df)

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
            "title": "Silent Failure Impact - Overall",
            "fig": plot_failure_impact(summary_df, "Overall")
        }
    ]

    if not aligned_df.empty:
        figs = plot_takeover_trajectory(aligned_df, "Overall")
        for i in range(0,len(figs)):
            p_graphs.append({
                "id": f"takeover_profile - {i}",
                "title": f"Takeover Control Profile - {i}",
                "fig": figs[i]
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

