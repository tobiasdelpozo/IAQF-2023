import pandas as pd
import multiprocessing as mp
import math
from math import ceil
from functools import partial
def simulate_all_pairs(dfs, K=50_000, trading_cost=0, delay_time=1):
    sim_part = partial(simulate_pair, K=K, trading_cost=trading_cost, delay_time=delay_time)
    results = []
    for df in dfs:
        results.append(sim_part(df))

    # with mp.Pool(mp.cpu_count()) as pool:
    #     results = pool.map(sim_part, dfs)
    return aggregate(results)

def aggregate(dfs):
    df_fin = pd.DataFrame()
    for idx, df in enumerate(dfs):
        df_fin[f'Pair {idx + 1} Leg 1 Size'] = df.iloc[:, 0]
        df_fin[f'Pair {idx + 1} Leg 2 Size'] = df.iloc[:, 1]
        df_fin[f'Pair {idx + 1} PnL'] = df['PnL']
        df_fin[f'Pair {idx + 1} Cash'] = df['Cash']
        df_fin[f'Pair {idx + 1} Portfolio'] = df['Portfolio']
        # if df.columns[0] in df_fin.columns:
        #     df_fin[df.columns[0]] += df[df.columns[0]]
        # else:
        #     df_fin[df.columns[0]] = df[df.columns[0]]
        # if df.columns[1] in df_fin.columns:
        #     df_fin[df.columns[1]] += df[df.columns[1]]
        # else:
        #     df_fin[df.columns[1]] = df[df.columns[1]]
    df_fin = df_fin.reindex(dfs[0].index)
    return df_fin

def simulate_pair(data, K, trading_cost=0, delay_time=1):
    d = data.copy()
    leg_1 = data.columns[0]
    leg_2 = data.columns[1]
    df = pd.DataFrame(index=data.index, columns=['Leg 1', 'Leg 2', 'Cash', 'Portfolio', 'PnL'])
    df = df.fillna(0)
    df.iloc[0, 2] = K
    init_prices = False
    closing = False

    # Set signal to be the delay.
    d['Signal'] = d[f'Signal {delay_time}']
    def buy_spread():
        #N1 = 100 * signal
        cash = df.loc[time, 'Cash']

        target_exec = int(ceil(cash / (10 * (curr_prices[0] + beta * curr_prices[1])) * signal))

        N1 = min(target_exec, data.loc[time, f'Leg 1 VWAP Volume {delay_time}'] * 0.05, data.loc[time, f'Leg 2 VWAP Volume {delay_time}'] / beta * 0.05)
        #N1 = target_exec
        df.loc[time, 'Cash'] -= N1 * curr_prices_vwap[0]
        df.loc[time, 'Leg 1'] = N1

        N2 = int(ceil(beta * N1))
        df.loc[time, 'Leg 2'] = -N2

        df.loc[time, 'Cash'] -= (N1 * curr_prices_vwap[0] + N2 * curr_prices_vwap[1]) * trading_cost

    def short_spread():
        #N1 = 100 * abs(signal)
        cash = df.loc[time, 'Cash']
        target_exec = int(ceil(cash / (10 * (curr_prices[0] + beta * curr_prices[1])) * abs(signal)))
        N1 = min(target_exec, data.loc[time, f'Leg 1 VWAP Volume {delay_time}'] * 0.05, data.loc[time, f'Leg 2 VWAP Volume {delay_time}'] / beta * 0.05)
        #N1 = target_exec
        df.loc[time, 'Leg 1'] = -N1

        N2 = int(ceil(beta * N1))
        df.loc[time, 'Cash'] -= N2 * curr_prices_vwap[1]
        df.loc[time, 'Leg 2'] = N2

        df.loc[time, 'Cash'] -= (N1 * curr_prices_vwap[0] + N2 * curr_prices_vwap[1]) * trading_cost

    def close_short():
        df.loc[time, 'Cash'] += -df.loc[time, 'Leg 1'] * (init_prices[0] - curr_prices_vwap[0])
        df.loc[time, 'Cash'] += df.loc[time, 'Leg 2'] * curr_prices_vwap[1]

        df.loc[time, 'Leg 1'], df.loc[time, 'Leg 2'] = 0, 0

        df.loc[time, 'Cash'] -= (-df.loc[time, 'Leg 1'] * curr_prices_vwap[0] + df.loc[time, 'Leg 2'] * curr_prices_vwap[1]) * trading_cost

    def close_long():
        df.loc[time, 'Cash'] += df.loc[time, 'Leg 1'] * curr_prices_vwap[0]
        df.loc[time, 'Cash'] += -df.loc[time, 'Leg 2'] * (init_prices[1] - curr_prices_vwap[1])

        df.loc[time, 'Leg 1'], df.loc[time, 'Leg 2'] = 0, 0

        df.loc[time, 'Cash'] -= (df.loc[time, 'Leg 1'] * curr_prices_vwap[0] - df.loc[time, 'Leg 2'] * curr_prices_vwap[1]) * trading_cost

    def get_paid(ffr=0):
        notional = 0
        if df.loc[time, 'Leg 1'] < 0:
            notional += -df.loc[time, 'Leg 1'] * init_prices[0]
        if df.loc[time, 'Leg 2'] < 0:
            notional += -df.loc[time, 'Leg 2'] * init_prices[1]

        df.loc[time, 'Cash'] += notional * (ffr + 0.005) * (1 / (252 * 390))

    def pay_interest(ffr=0):
        if df.loc[time, 'Cash'] < 0:
            df.loc[time, 'Cash'] -= df.loc[time, 'Cash'] * ffr / (252 * 390)

    def update_port_value():
        if df.loc[time, 'Leg 1'] > 0:
            long_val = df.loc[time, 'Leg 1'] * curr_prices[0]
            short_val = -df.loc[time, 'Leg 2'] * (init_prices[1] - curr_prices[1])
        elif df.loc[time, 'Leg 1'] < 0:
            long_val = df.loc[time, 'Leg 2'] * curr_prices[1]
            short_val = -df.loc[time, 'Leg 1'] * (init_prices[0] - curr_prices[0])
        else:
            long_val, short_val = 0, 0
        df.loc[time, 'Portfolio'] = df.loc[time, 'Cash'] + long_val + short_val

    def update_pnl():
        if init_prices:
            if df.loc[time, 'Leg 1'] > 0:
                long_pnl = df.loc[time, 'Leg 1'] * (curr_prices[0] - init_prices[0])
                short_pnl = -df.loc[time, 'Leg 2'] * (init_prices[1] - curr_prices[1])
            else:
                long_pnl = df.loc[time, 'Leg 2'] * (curr_prices[1] - init_prices[1])
                short_pnl = -df.loc[time, 'Leg 1'] * (init_prices[0] - curr_prices[0])
            df.loc[time, 'PnL'] = short_pnl + long_pnl
        else:
            df.loc[time, 'PnL'] = 0

    for num, time in enumerate(data.index):
        signal = d.loc[time, 'Signal']
        ffr = d.loc[time, 'FFR']
        curr_prices = [d.loc[time].iloc[0], d.loc[time].iloc[1]]
        curr_prices_vwap = [d.loc[time, f'Leg 1 VWAP {delay_time}'], d.loc[time, f'Leg 2 VWAP {delay_time}']]
        beta = d.loc[time, 'Betas']
        if num > 0:
            df.loc[time] = df.iloc[num - 1, :]
            update_pnl()
            pay_interest(ffr=ffr)
            get_paid(ffr=ffr)
        if num == len(df.index) - 1:
            if df.loc[time, 'Leg 1'] > 0:
                close_long()
            elif df.loc[time, 'Leg 1'] < 0:
                close_short()
            update_port_value()
            df.columns = [leg_1, leg_2, 'Cash', 'Portfolio', 'PnL']
            return df
        if closing:
            if data.loc[time, 'Volume 1'] < abs(df.loc[time, 'Leg 1']) or data.loc[time, 'Volume 2'] < abs(df.loc[time, 'Leg 2']):
                continue
            else:
                if df.loc[time, 'Leg 1'] > 0:
                    close_long()
                elif df.loc[time, 'Leg 1'] < 0:
                    close_short()
                closing = False
        if signal > 0 and df.loc[time, 'Leg 1'] == 0:
            # init_prices = curr_prices
            init_prices = curr_prices_vwap
            # ADD VWAP PRICE CHANGE HERE.
            buy_spread()
        elif signal > 0 and df.loc[time, 'Leg 1'] < 0:
            if data.loc[time, 'Volume 1'] * 0.05 >= -df.loc[time, 'Leg 1'] and data.loc[time, 'Volume 2'] * 0.05 >= df.loc[time, 'Leg 2']:
                close_short()
                closing = False
            else:
                closing = True
                continue
            init_prices = curr_prices_vwap
            buy_spread()
        elif signal < 0 and df.loc[time, 'Leg 1'] == 0:
            init_prices = curr_prices_vwap
            short_spread()
        elif signal < 0 and df.loc[time, 'Leg 1'] > 0:
            if data.loc[time, 'Volume 1'] *0.05 >= df.loc[time, 'Leg 1'] and data.loc[time, 'Volume 2'] *0.05 >= -df.loc[time, 'Leg 2']:
                close_long()
                closing = False
            else:
                closing = True
                continue
            init_prices = curr_prices_vwap
            short_spread()
        elif signal == 0:
            if df.loc[time, 'Leg 1'] > 0:
                if data.loc[time, 'Volume 1'] *0.05 >= df.loc[time, 'Leg 1'] and data.loc[time, 'Volume 2'] *0.05 >= -df.loc[time, 'Leg 2']:
                    close_long()
                    closing = False
                else:
                    closing = True
                    continue
            elif df.loc[time, 'Leg 1'] < 0:
                if data.loc[time, 'Volume 1'] * 0.05 >= -df.loc[time, 'Leg 1'] and data.loc[time, 'Volume 2'] * 0.05 >= df.loc[time, 'Leg 2']:
                    close_short()
                    closing = False
                else:
                    closing = True
                    continue
        update_port_value()
