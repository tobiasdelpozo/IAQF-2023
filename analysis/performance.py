import pandas as pd
import numpy as np
import statsmodels.api as sm
from itertools import pairwise
from typing import List, Tuple
def perf_summ(data: pd.DataFrame, adj: int = 12, mkt: pd.Series=None) -> pd.DataFrame:
    """
    Calculate performance summary for all needed statistics.

    Specifically:
        - Annualized rets
        - Annualized Volatility (std of rets)
        - Annualized Sharpe Ratio (mean/vol)
        - Annualized Sortino (mean/downside vol)
        - Market Beta (regression of rets on mkt)
        - Downside Beta (regression of rets on downside mkt)
        - Skewness (third moment)
        - Excess Kurtosis (fourth moment - remember normal has kurtosis of 3)
        - VaR (empirically observed VaR, just take the 5th quantile)
        - CVaR (empiracally observed CVaR, mean value conditioned on being in 5th quantile)
        - Min (worst rets seen)
        - Max (best rets seen)
        - Max Drawdown (biggest peak to trough)
        - Calmar Ratio (annualized rets / max drawdown)
        - Peak (largest peak of cumulative rets)
        - Bottom (lowest point of cumulative rets)
        - Recovery (max drawdown to new peak)

    Args:
        data (pd.DataFrame): rets data for a single asset (make sure to change adj=X accordingly)
        adj (int, optional): Date adjustment period. Defaults to 12 (monthly -> annual).

    rets:
        pd.DataFrame: Summary statistics for the rets data.
    """
    summary = data.mean().to_frame("Annualized Return").apply(lambda x: x * adj)
    summary["Annualized Volatility"] = data.std().apply(lambda x: x * np.sqrt(adj))
    summary["Annualized Sharpe Ratio"] = (
            summary["Annualized Return"] / summary["Annualized Volatility"]
    )
    summary["Annualized Sortino Ratio"] = summary["Annualized Return"] / (
        (data[data < 0]).std() * np.sqrt(adj)
    )

    if mkt is not None:
        for asset in data.columns:
            summary.loc[asset, "Market Beta"] = sm.OLS(
                data[asset], sm.add_constant(mkt)
            ).fit().params[1]
            summary.loc[asset, "Market Alpha"] = sm.OLS(
                data[asset], sm.add_constant(mkt)
            ).fit().params[0]
            mkt_d = mkt[mkt < 0]
            summary.loc[asset, "Downside Beta"] = sm.OLS(
                data.loc[mkt_d.index, asset], sm.add_constant(mkt_d)
            ).fit().params[1]

    summary["Skewness"] = data.skew()
    summary["Excess Kurtosis"] = data.kurtosis()
    summary["VaR (0.05)"] = data.quantile(0.05, axis=0)
    summary["CVaR (0.05)"] = data[data <= data.quantile(0.05, axis=0)].mean()
    summary["Min"] = data.min()
    summary["Max"] = data.max()

    wealth_index = 1000 * (1 + data).cumprod()
    previous_peaks = wealth_index.cummax()
    drawdowns = (wealth_index - previous_peaks) / previous_peaks

    summary["Max Drawdown"] = drawdowns.min()
    summary["Calmar Ratio"] = ((data.mean() * adj) / drawdowns.min()).abs()
    summary["Peak"] = [
        previous_peaks[col][: drawdowns[col].idxmin()].idxmax()
        for col in previous_peaks.columns
    ]
    summary["Bottom"] = drawdowns.idxmin()

    recovery_date = []
    for col in wealth_index.columns:
        prev_max = previous_peaks[col][: drawdowns[col].idxmin()].max()
        recovery_wealth = pd.DataFrame(
            [wealth_index[col][drawdowns[col].idxmin():]]
        ).T
        recovery_date.append(
            recovery_wealth[recovery_wealth[col] >= prev_max].index.min()
        )
    summary["Recovery"] = recovery_date
    summary["Recovery"] = summary["Recovery"].replace(
        to_replace=pd.NaT, value="Not yet recovered"
    )
    return summary

def multi_perf_summ(
        data: pd.DataFrame,
        dates: List[str] = None,
        multi_asset: bool = True,
        adj: int = 12,
        mkt: pd.Series = None,
) -> pd.DataFrame:
    """
    Multi-asset/multi-date performance summary. Slices the performance summary according to date and asset, returning either
    a singly-indexed DataFrame (in the case of only one asset with several dates), or a multi-indexed DataFrame in the case
    of multiple assets and multiple dates.

    Args:
        data (pd.DataFrame): rets DataFrame, each columns corresponds to an asset.
        dates (List[str], optional): Dates to slice by - Full, and XX-end included. Defaults to None.
        multi_asset (bool, optional): Logic for reseting index in the case of a single asset. Defaults to False.
        adj (int, optional): _description_. Defaults to 12.

    rets:
        pd.DataFrame: (singly/multi)-indexed DataFrame with corresponding performance summary.
    """
    stats = []
    labels = []
    if dates is None:
        return perf_summ(data, adj=adj, mkt=mkt)
    else:
        labels.append("Full")
        stats.append(perf_summ(data, adj=adj, mkt=mkt))
    labels.append(f"Beg-{dates[0]}")
    stats.append(perf_summ(data.loc[: dates[0]], adj=adj, mkt=mkt.loc[: dates[0]]))

    if len(dates) > 2:
        for i, j in pairwise(dates):
            data_sub = data.loc[i:j]
            stats.append(perf_summ(data_sub, adj=adj, mkt=mkt.loc[i:j]))
            labels.append(f"{i}-{j}")
    stats.append(perf_summ(data=data.loc[dates[-1]:], adj=adj, mkt=mkt.loc[dates[-1]:]))
    labels.append(f"{dates[-1]}-End")
    return (
        pd.concat(stats, keys=labels)
        if multi_asset
        else pd.concat(stats, keys=labels).reset_index(1).drop("level_1", axis=1)
    )

if __name__ == "__main__":
    data = pd.read_csv("../data/raw_data.csv", index_col=0).pivot_table(values="adj_close", index="date", columns="ticker")
    data = data[["SPY", "VOO"]].pct_change().dropna()
    print(multi_perf_summ(data, adj=252, mkt=data["SPY"]).T)