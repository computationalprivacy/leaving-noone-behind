import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from lnb.distance import top_n_vulnerable_dists


def plot_achilles(distances: dict, n: int):
    sns.set_theme(style="white")

    top_n = top_n_vulnerable_dists(distances, n)

    thresh = np.min(top_n)

    plot_df = pd.DataFrame(
        {"Achilles score": distances.values()}, index=distances.keys()
    )

    plot_df = plot_df.assign(
        Vulnerability=plot_df["Achilles score"].map(
            lambda x: "Low" if x < thresh else "High"
        )
    )
    # plot_df.loc[top_n,'Vulnerability'] = 'High'

    fig, axs = plt.subplots(1, 1)

    sns.histplot(
        plot_df,
        x="Achilles score",
        hue="Vulnerability",
        ax=axs,
        alpha=1,
        hue_order=["High", "Low"],
        palette=["#ff0083", "#89d98c"],
        stat="probability",
    )

    plt.savefig("achilles_scores.png")


def calculate_statistics(distances: dict):
    mean_score = np.mean(list(distances.values()))
    q3 = np.quantile(list(distances.values()), 0.75)

    perc_below_avg = (
        len([k for k in distances if distances[k] < mean_score])
        / len(distances.keys())
        * 100
    )

    print(
        f"{perc_below_avg:.2f}% of the records in the target dataset have a below-average Achilles score."
    )
    print(
        f"The third quantile is {q3:.2f}, i.e. three quarters (75%) of the records have an Achilles score below {q3:.2f}."
    )


def plot_mia_scores(mia_results: list, output_path: str = None):
    plot_df = mia_results_to_df(mia_results)
    _, axs = plt.subplots(1, 2, figsize=(15, 5), sharey=True)

    sns.barplot(
        plot_df, x="record", y="auc", hue="Model", ax=axs[0], palette="tab10"
    )
    sns.barplot(
        plot_df,
        x="record",
        y="accuracy",
        hue="Model",
        ax=axs[1],
        palette="tab10",
    )
    axs[0].set_ylim(0.3, 1.0)

    if output_path is not None:
        plt.savefig(output_path + "mia_scores_barplot.png")


def mia_results_to_df(mia_results: list):
    m = mia_results[0]
    df = pd.DataFrame()
    for model in m[1].keys():
        df_temp = pd.concat(
            [
                pd.DataFrame(
                    {
                        "auc": mia_results[i][1][model]["auc"],
                        "accuracy": mia_results[i][1][model]["accuracy"],
                        "model": model,
                    },
                    index=[mia_results[i][0]],
                )
                for i in range(len(mia_results))
            ]
        )
        df = pd.concat([df, df_temp])
    df = df.reset_index().rename({"index": "record"}, axis=1)
    df = df.assign(Model=df.model.map(lambda x: model_names[x]))
    return df


model_names = {
    "random_forest": "Random Forest",
    "logistic_regression": "Logistic Regression",
    "mlp": "MLP",
}
