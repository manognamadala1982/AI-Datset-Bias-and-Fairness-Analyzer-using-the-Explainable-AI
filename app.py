from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from analyzer import (
    MODEL_FACTORIES,
    load_dataset,
    train_and_compare,
)


st.set_page_config(
    page_title="Crime Dataset Fairness & XAI Analyzer",
    page_icon="⚖️",
    layout="wide",
)

st.title("Crime Dataset Fairness & Explainable AI Analyzer")
st.markdown(
    "Explore model performance, group-level error rates, and global feature "
    "importance. This is a classroom/research demonstration, **not** a tool for "
    "predicting individual behavior or making policing, charging, sentencing, "
    "or other real-world decisions."
)
st.warning(
    "The data contains sensitive demographic attributes and criminal-justice "
    "outcomes. Fairness metrics are descriptive and depend on the chosen target, "
    "positive outcome, groups, and sample sizes. They do not prove that a model "
    "is fair or suitable for use."
)


def selection_rate_parity(group_metrics: pd.DataFrame) -> float:
    """Calculate selection-rate parity without relying on cached analyzer code."""
    required_columns = {"Support", "Selection rate"}
    if not required_columns.issubset(group_metrics.columns):
        return float("nan")

    supported = group_metrics.loc[group_metrics["Support"] == "Included"]
    rates = pd.to_numeric(supported["Selection rate"], errors="coerce").dropna()
    if len(rates) < 2 or rates.max() <= 0:
        return float("nan")
    return float(rates.min() / rates.max())


@st.cache_data(show_spinner=False)
def get_dataset(uploaded_file, default_path: str):
    if uploaded_file is not None:
        uploaded_file.seek(0)
        return load_dataset(uploaded_file)
    return load_dataset(default_path)


dataset_path = Path(__file__).with_name("crime_data.csv")
uploaded_file = st.file_uploader(
    "Upload a CSV (or use the included crime_data.csv)", type=["csv"]
)

if uploaded_file is None and not dataset_path.exists():
    st.error("crime_data.csv was not found beside app.py. Upload your CSV to continue.")
    st.stop()

try:
    dataset_info = get_dataset(uploaded_file, str(dataset_path))
except (pd.errors.ParserError, UnicodeDecodeError, OSError, ValueError) as error:
    st.error(f"Could not load the CSV: {error}")
    st.stop()

data = dataset_info.data
if data.empty:
    st.error("The selected CSV has no data rows.")
    st.stop()

total, retained, duplicates = st.columns(3)
total.metric("Rows in CSV", f"{dataset_info.original_rows:,}")
retained.metric("Rows after exact deduplication", f"{len(data):,}")
duplicates.metric("Exact duplicate rows removed", f"{dataset_info.duplicate_rows:,}")

with st.expander("Dataset overview", expanded=True):
    overview = pd.DataFrame(
        {
            "Column": data.columns,
            "Data type": [str(data[column].dtype) for column in data.columns],
            "Unique values": [int(data[column].nunique(dropna=True)) for column in data.columns],
            "Missing values": [int(data[column].isna().sum()) for column in data.columns],
        }
    )
    st.dataframe(overview, width="stretch", hide_index=True)

if "Offender_Gender" in data.columns:
    st.subheader("Recorded entries by offender gender")
    st.warning(
        "These are counts of deduplicated dataset rows, not counts of unique "
        "people or per-capita crime rates. They cannot establish that one gender "
        "commits more crime: the dataset has no population denominator and may "
        "reflect reporting, recording, and enforcement patterns."
    )
    gender_counts = (
        data["Offender_Gender"]
        .fillna("Missing")
        .astype(str)
        .value_counts()
        .rename_axis("Recorded offender gender")
        .rename("Dataset rows")
        .reset_index()
    )
    gender_counts["Share of rows"] = gender_counts["Dataset rows"] / len(data)
    gender_columns = st.columns(min(len(gender_counts), 4))
    for column, row in zip(gender_columns, gender_counts.itertuples(index=False)):
        column.metric(
            f"{row[0]} rows",
            f"{row[1]:,}",
            f"{row[2]:.1%} of deduplicated rows",
        )
    st.bar_chart(gender_counts, x="Recorded offender gender", y="Dataset rows")
    st.caption(
        "This is a descriptive summary of the supplied records. It is separate "
        "from model fairness metrics, which compare the model's selected "
        "prediction rates and error rates across groups."
    )

candidate_sensitive = [
    column
    for column in data.columns
    if any(token in column.casefold() for token in ("race", "gender", "sex", "age"))
]
default_sensitive = [
    column
    for column in ("Offender_Race", "Offender_Gender", "Victim_Race", "Victim_Gender")
    if column in data.columns
]

st.subheader("Configure the analysis")
with st.form("analysis_form"):
    target = st.selectbox(
        "Outcome to model",
        options=list(data.columns),
        index=list(data.columns).index("Disposition")
        if "Disposition" in data.columns
        else 0,
        help=(
            "The outcome is what the models learn to predict. The default is "
            "Disposition (OPEN/CLOSED) for a retrospective demonstration only."
        ),
    )
    sensitive_columns = st.multiselect(
        "Sensitive attributes for fairness evaluation",
        options=candidate_sensitive,
        default=default_sensitive or candidate_sensitive[:1],
        help="These columns are excluded from model inputs and used only for evaluation.",
    )
    target_labels = sorted(data[target].dropna().astype(str).unique().tolist())
    positive_default = "OPEN" if "OPEN" in target_labels else target_labels[-1]
    positive_label = st.selectbox(
        "Positive outcome for fairness rates",
        options=target_labels,
        index=target_labels.index(positive_default),
        help="Selection rate, true-positive rate, and false-positive rate refer to this class.",
    )
    selected_models = st.multiselect(
        "Models to compare",
        options=list(MODEL_FACTORIES),
        default=list(MODEL_FACTORIES),
    )
    minimum_group_size = st.number_input(
        "Minimum test-set group size for fairness gap summaries",
        min_value=2,
        max_value=500,
        value=30,
        step=1,
        help=(
            "Smaller groups remain visible in per-group results but are omitted "
            "from disparity gap summaries due to unstable estimates."
        ),
    )
    submitted = st.form_submit_button("Train models and analyze", type="primary")

if submitted:
    if not sensitive_columns:
        st.error("Select at least one sensitive attribute.")
        st.stop()
    if not selected_models:
        st.error("Select at least one model.")
        st.stop()

    with st.spinner("Training models and calculating holdout explanations..."):
        try:
            results = train_and_compare(
                data=data,
                target=target,
                sensitive_columns=sensitive_columns,
                model_names=selected_models,
                positive_label=positive_label,
                minimum_group_size=int(minimum_group_size),
            )
        except (ValueError, KeyError) as error:
            st.error(str(error))
            st.stop()

    st.session_state["analysis_results"] = results
    st.session_state["analysis_target"] = target

results = st.session_state.get("analysis_results")
if results is not None:
    st.subheader("Model comparison")
    st.caption(
        f"Target: {st.session_state.get('analysis_target')} · "
        f"Positive outcome: {results['positive_label']} · "
        f"Holdout rows: {results['test_size']:,}"
    )
    st.dataframe(
        results["comparison"].style.format(
            {
                "Accuracy": "{:.3f}",
                "Balanced accuracy": "{:.3f}",
                "Macro F1": "{:.3f}",
            }
        ),
        width="stretch",
        hide_index=True,
    )
    if results["excluded_constant_columns"]:
        st.caption(
            "Constant input columns excluded automatically: "
            + ", ".join(results["excluded_constant_columns"])
        )

    st.subheader("Fairness by selected attribute")
    st.caption(
        "Gaps are max-minus-min differences across groups meeting the support "
        "threshold; rates range from 0 to 1. Selection-rate parity is the lowest "
        "group selection rate divided by the highest (closer to 100% means rates "
        "are more similar). These are separate metrics, not an overall fairness "
        "score or guarantee."
    )
    for attribute, model_results in results["fairness"].items():
        st.markdown(f"#### {attribute}")
        for model_name, fairness in model_results.items():
            with st.expander(model_name, expanded=True):
                gap_values = fairness["gaps"]
                gap_columns = st.columns(4)
                for column, metric in zip(
                    gap_columns,
                    (
                        "Selection rate",
                        "True positive rate",
                        "False positive rate",
                        "Selection-rate parity",
                    ),
                ):
                    value = gap_values.get(metric, float("nan"))
                    if metric == "Selection-rate parity" and pd.isna(value):
                        # Recover the metric for results cached in the current
                        # Streamlit session before this metric was introduced.
                        value = selection_rate_parity(fairness["by_group"])
                    column.metric(
                        (
                            "Fairness rate"
                            if metric == "Selection-rate parity"
                            else f"{metric} gap"
                        ),
                        (
                            "N/A"
                            if pd.isna(value)
                            else (
                                f"{value:.1%}"
                                if metric == "Selection-rate parity"
                                else f"{value:.3f}"
                            )
                        ),
                        help=(
                            "Lowest eligible group's selection rate divided by "
                            "the highest eligible group's rate. N/A means fewer "
                            "than two supported groups or all rates are zero."
                            if metric == "Selection-rate parity"
                            else None
                        ),
                    )
                st.dataframe(
                    fairness["by_group"].style.format(
                        {
                            "Selection rate": "{:.3f}",
                            "True positive rate": "{:.3f}",
                            "False positive rate": "{:.3f}",
                        },
                        na_rep="N/A",
                    ),
                    width="stretch",
                    hide_index=True,
                )

    st.subheader("Explainable AI: global permutation importance")
    model_choice = st.selectbox("Explain this model", list(results["models"]))
    importance = results["models"][model_choice]["importance"]
    st.caption(
        "Importance measures the change in holdout balanced accuracy when one "
        "feature is shuffled. It describes this fitted model and dataset; it is "
        "not causal evidence."
    )
    st.dataframe(
        importance.head(20).style.format(
            {"Importance": "{:.4f}", "Std. dev.": "{:.4f}"}
        ),
        width="stretch",
        hide_index=True,
    )
    plot_data = importance.head(15).sort_values("Importance")
    figure, axis = plt.subplots(figsize=(9, max(4, len(plot_data) * 0.32)))
    axis.barh(
        plot_data["Feature"],
        plot_data["Importance"],
        xerr=plot_data["Std. dev."],
        color="#376a9f",
        alpha=0.9,
    )
    axis.set_xlabel("Mean change in balanced accuracy after shuffling")
    axis.set_title(f"Top global feature importance — {model_choice}")
    figure.tight_layout()
    st.pyplot(figure)
    plt.close(figure)

    st.download_button(
        "Download model comparison CSV",
        data=results["comparison"].to_csv(index=False).encode("utf-8"),
        file_name="model_comparison.csv",
        mime="text/csv",
    )
else:
    st.info("Configure the analysis above, then select **Train models and analyze**.")
