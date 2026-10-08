import pandas as pd
import plotly.express as px


TITANIC_URL = (
    "https://raw.githubusercontent.com/leontoddjohnson/datasets/main/data/titanic.csv"
)
AGE_GROUPS = ["Child", "Teen", "Adult", "Senior"]


def load_titanic():
    """Load the Titanic data and normalize column names."""
    df = pd.read_csv(TITANIC_URL)
    df.columns = (
        df.columns.str.strip()
        .str.lower()
        .str.replace(r"\s+", "_", regex=True)
    )
    return df


def survival_demographics() -> pd.DataFrame:
    """Summarize survival by passenger class, sex, and age group."""
    df = load_titanic()

    df["age_group"] = pd.cut(
        df["age"],
        bins=[-float("inf"), 12, 19, 59, float("inf")],
        labels=AGE_GROUPS,
        ordered=True,
    )

    group_columns = ["pclass", "sex", "age_group"]
    grouped = (
        df.dropna(subset=["age_group"])
        .groupby(group_columns, observed=True)
        .agg(
            n_passengers=("survived", "size"),
            n_survivors=("survived", "sum"),
        )
    )

    all_combinations = pd.MultiIndex.from_product(
        [
            sorted(df["pclass"].dropna().unique()),
            sorted(df["sex"].dropna().unique()),
            AGE_GROUPS,
        ],
        names=group_columns,
    )

    result = grouped.reindex(all_combinations, fill_value=0).reset_index()
    result["age_group"] = pd.Categorical(
        result["age_group"],
        categories=AGE_GROUPS,
        ordered=True,
    )
    result["survival_rate"] = result["n_survivors"].div(
        result["n_passengers"].where(result["n_passengers"] > 0)
    )

    return result.sort_values(
        ["pclass", "sex", "age_group"]
    ).reset_index(drop=True)


def visualize_demographic():
    """Plot survival rates across age groups, sexes, and passenger classes."""
    result = survival_demographics()

    fig = px.bar(
        result,
        x="age_group",
        y="survival_rate",
        color="sex",
        barmode="group",
        facet_col="pclass",
        category_orders={
            "age_group": AGE_GROUPS,
            "pclass": [1, 2, 3],
        },
        labels={
            "age_group": "Age group",
            "survival_rate": "Survival rate",
            "sex": "Sex",
            "pclass": "Passenger class",
        },
        title="Titanic survival rate by age, sex, and passenger class",
    )
    fig.update_yaxes(tickformat=".0%", range=[0, 1])
    return fig


def family_groups() -> pd.DataFrame:
    """Summarize passenger counts and fares by family size and class."""
    df = load_titanic()
    df["family_size"] = df["sibsp"] + df["parch"] + 1

    return (
        df.groupby(["pclass", "family_size"], as_index=False)
        .agg(
            n_passengers=("fare", "size"),
            avg_fare=("fare", "mean"),
            min_fare=("fare", "min"),
            max_fare=("fare", "max"),
        )
        .sort_values(["pclass", "family_size"])
        .reset_index(drop=True)
    )


def last_names() -> pd.Series:
    """Return passenger counts indexed by surname."""
    surnames = (
        load_titanic()["name"]
        .str.split(",", n=1)
        .str[0]
        .str.strip()
    )
    return surnames.value_counts()