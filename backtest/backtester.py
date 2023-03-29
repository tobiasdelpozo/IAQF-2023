import pandas as pd
import numpy as np


def backtest(sample_data, contract_size=100, TRANSACTION_COST=0.0):
    data = sample_data.copy()
    cur_pos = (0, 0)
    init_val, prev_val = 0, 0
    closed = False
    new_pos = False
    for i in data.index:
        cur_prices = [data.loc[i].iloc[0], data.loc[i].iloc[1]]
        signal = data.loc[i, "Signal"]
        if signal > 0:
            if cur_pos[0] == 0 and cur_pos[1] == 0:
                cur_pos = (
                    signal * contract_size,
                    -1 * contract_size * signal * sample_data.loc[i, "Betas"],
                )
                data.loc[i, "Transaction Cost"] = (
                    -np.dot(np.abs(cur_pos), cur_prices) * TRANSACTION_COST
                )
                init_val = np.dot(cur_pos, cur_prices)
                new_pos = True
            elif cur_pos[0] < 0 and cur_pos[1] > 0:
                data.loc[i, "Transaction Cost"] = (
                    -np.dot(np.abs(cur_pos), cur_prices) * TRANSACTION_COST
                )
                closed = True
        elif signal < 0:
            if cur_pos[0] == 0 and cur_pos[1] == 0:
                cur_pos = (
                    signal * contract_size,
                    -signal * contract_size * sample_data.loc[i, "Betas"],
                )
                data.loc[i, "Transaction Cost"] = (
                    -np.dot(np.abs(cur_pos), cur_prices) * TRANSACTION_COST
                )
                init_val = np.dot(cur_pos, cur_prices)
                new_pos = True
            elif cur_pos[0] > 0 and cur_pos[1] < 0:
                data.loc[i, "Transaction Cost"] = (
                    -np.dot(np.abs(cur_pos), cur_prices) * TRANSACTION_COST
                )
                closed = True

        if new_pos == True:
            prev_val = init_val
            new_pos = False

        data.loc[i, "Daily PnL"] = np.dot(cur_pos, cur_prices) - prev_val
        prev_val = np.dot(cur_pos, cur_prices)

        if closed == True:
            cur_pos = (0, 0)
            init_val, prev_val = 0, 0
            closed = False

        data.loc[i, "Pos 1"] = cur_pos[0]
        data.loc[i, "Pos 2"] = cur_pos[1]
        data.loc[i, "Position Value"] = np.dot(cur_pos, cur_prices)

    data["Transaction Cost"] = data["Transaction Cost"].fillna(0)
    data["Total PnL"] = data["Daily PnL"].cumsum()
    data["Total Transaction"] = data["Transaction Cost"].cumsum()
    data["Total PnL"] = data["Total PnL"] + data["Total Transaction"]
    return data
