from lib.file_tools import create_participants_by_data_folder, ExcelConverter
from lib.graph import (
    process_toc_experiment_data, plot_manual_pct,
    process_brake_aligned_telemetry, plot_takeover_trajectory_overall, plot_takeover_trajectory,
    process_ttc_data, plot_ttc_timestep, filter_experiment_data, sanitize_scenario_data, plot_ttc_first_warning_aligned_ci
)
from pathlib import Path
import random
import pandas as pd
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt

from scipy.stats import chi2
def lr_test(reduced, full):
    lr = 2 * (full.llf - reduced.llf)
    df_diff = len(full.fe_params) - len(reduced.fe_params)
    p = chi2.sf(lr, df_diff)

    print(f"LR statistic: {lr:.3f}")
    print(f"df difference: {df_diff}")
    print(f"p-value: {p:.4f}")

def mixedlm_fit(formula, data_frame, groups, reml=True):
    model = smf.mixedlm(
        formula,
        data=data_frame,
        groups=groups,
    )
    return model.fit(reml=reml)

def test0(df):
    m1 = mixedlm_fit(
        "Manual_Pct ~ Had_Prior_Failure_Exp + Trial_Order",
        data_frame = df,
        groups = df["Participant"],
        reml = False,
    )

    m2 = mixedlm_fit(
        "Manual_Pct ~ Had_Prior_Failure_Exp * Scenario_Name + Trial_Order",
        data_frame=df,
        groups=df["Participant"],
        reml=False,
    )
    lr_test(m1, m2)

def t_test(df):
    # Actual Model
    result = mixedlm_fit(
        """Manual_Pct ~ Traffic_Level
                              * Warning""",
        data_frame=df,
        groups= df["Participant"]
    )

    print(result.summary())

def test2(df):
    reduced_result = mixedlm_fit(
        "Manual_Pct ~ Had_Prior_Failure_Exp + Trial_Order",
        data_frame = df,
        groups = df["Participant"],
        reml = False,
    )


    full_result = mixedlm_fit(
        "Manual_Pct ~ Had_Prior_Failure_Exp * Scenario_Name + Trial_Order",
        data_frame=df,
        groups=df["Participant"],
        reml = False,
    )


    lr_test(reduced_result,full_result)

def test3(df):
    m2 = mixedlm_fit(
        "Manual_Pct ~ Had_Prior_Failure_Exp * Scenario_Name + Trial_Order",
        data_frame=df,
        groups=df["Participant"],
        reml=False,
    )

    print("m2 AIC:", m2.aic)
    print("m2 BIC:", m2.bic)

    m3 = mixedlm_fit(
        """Manual_Pct ~ Had_Prior_Failure_Exp
                      * Traffic_Level
                      * Warning
                      + Trial_Order""",
        data_frame=df,
        groups=df["Participant"],
        reml=False,
    )


    print("m3 AIC:", m3.aic)
    print("m3 BIC:", m3.bic)

def test4(df):
    m1 = mixedlm_fit(
        "Manual_Pct ~ Had_Prior_Failure_Exp + Trial_Order",
        data_frame=df,
        groups=df["Participant"],
        reml=False,
    )

    m2 = mixedlm_fit(
        "Manual_Pct ~ Had_Prior_Failure_Exp + Traffic_Level * Warning + Trial_Order",
        data_frame=df,
        groups=df["Participant"],
        reml=False,
    )

    m3 = mixedlm_fit(
        "Manual_Pct ~ Had_Prior_Failure_Exp * Scenario_Name + Trial_Order",
        data_frame=df,
        groups=df["Participant"],
        reml=False,
    )
    print("Test between base and warning")
    lr_test(m1, m2)
    print("Test between base and Scenario")
    lr_test(m1, m3)

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    datas = []

    p_objs = create_participants_by_data_folder("data")
    for p_obj in p_objs:
        # if p_obj.get_name() == "P72":
            datas.append(p_obj.get_data_frame())

    combined_df = pd.concat(datas, ignore_index=True)
    # Left for plotting
    # df = process_toc_experiment_data(combined_df)
    # plot_manual_pct(df, "a")
    sanitized_df = sanitize_scenario_data(combined_df, "D")
    plot_ttc_first_warning_aligned_ci(sanitized_df, "D")

    # df = process_brake_aligned_telemetry(combined_df)
    # plot_takeover_trajectory_overall(df,"overall")

    plt.show()
 
    # t_test(df)


