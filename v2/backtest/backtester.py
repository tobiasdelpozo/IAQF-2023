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


# Backtester.
# NOTE: this function comes directly from Tobias Rodriguez del Pozo's
#       Homework 2 for Spread Trading. This will be heavily modified
#       for the final submission of this project, but for now, it's
#       to get a working backtester running.
def backtest(data_o, asset_1, asset_2, starting_bal):
    data = data_o.copy()
    perf = {'Trade Days': [],
            'Trades': [],
            'Balance': [],
            'Paper PnL': [],
            'Realized PnL': [],
            'Gross Exposure': []}

    # Keep track of entry prices to track the PnL
    curr_pos = {asset_1: 0, asset_2: 0, f'{asset_1}_P': 0, f'{asset_2}_P': 0, 'gross': 0, 'long': 0}
    curr_bal = starting_bal
    last_trade_day = data.index[-1]

    for date, val in data.iterrows():
        # Grab values for this day.
        asset_1_p = val[f'{asset_1}']
        asset_2_p = val[f'{asset_2}']
        asset_1_s = val[f'{asset_1}_M']
        asset_2_s = val[f'{asset_2}_M']
        sig = val['signal']

        # Calculate Paper PnL
        paper_pnl = curr_pos[asset_1]*(asset_1_p - curr_pos[f'{asset_1}_P']) + curr_pos[asset_2]*(asset_2_p - curr_pos[f'{asset_2}_P'])

        # paper_pnl = asset_1_p * curr_pos[asset_1] + asset_2_p * curr_pos[asset_2] - curr_pos['gross']
        perf['Paper PnL'].append(paper_pnl)

        if date == last_trade_day:
            perf['Trades'].append(curr_pos['long']* -1)
            curr_bal += asset_1_p * curr_pos[asset_1] + asset_2_p * curr_pos[asset_2]
            perf['Balance'].append(curr_bal)
            perf['Realized PnL'].append(paper_pnl)
            perf['Trade Days'].append(date)
            perf['Gross Exposure'].append(0)
            break

        # Check to close the position.
        if sig == 2 and curr_pos['long'] != 0:
            curr_bal += asset_1_p * curr_pos[asset_1] + asset_2_p * curr_pos[asset_2]
            perf['Trades'].append(-1 if curr_pos['long'] == 1 else 1)
            perf['Balance'].append(curr_bal)
            perf['Realized PnL'].append(paper_pnl)
            curr_pos = {asset_1: 0, asset_2: 0, f'{asset_1}_P': 0, f'{asset_2}_P': 0, 'gross': 0, 'long': 0}
            perf['Trade Days'].append(date)
            perf['Gross Exposure'].append(0)
            continue

        # Check if we can go from nothing -> long.
        if sig == 1 and curr_pos['long'] == 0:
            # No PnL realized, just draw from balance.
            curr_bal += asset_2_s * asset_2_p - asset_1_s * asset_1_p
            perf['Balance'].append(curr_bal)
            perf['Trades'].append(1)
            perf['Realized PnL'].append(0)
            exp = abs(asset_2_s*asset_2_p) + abs(asset_1_p*asset_1_s)
            perf['Gross Exposure'].append(exp)
            curr_pos = {asset_1: asset_1_s, asset_2: -asset_2_s, f'{asset_1}_P': asset_1_p, f'{asset_2}_P': asset_2_p, 'gross': exp, 'long': 1}
            perf['Trade Days'].append(date)
            continue

        # Check if we can go from nothing -> short.
        if sig == -1 and curr_pos['long'] == 0:
            # No PnL realized, just draw from balance.
            curr_bal +=  asset_1_s * asset_1_p - asset_2_s * asset_2_p
            perf['Balance'].append(curr_bal)
            perf['Trades'].append(-1)
            perf['Realized PnL'].append(0)
            exp = abs(asset_2_s*asset_2_p) + abs(asset_1_p*asset_1_s)
            perf['Gross Exposure'].append(exp)
            curr_pos = {asset_1: -asset_1_s, asset_2: asset_2_s, f'{asset_1}_P': asset_1_p, f'{asset_2}_P': asset_2_p, 'gross': exp, 'long': -1}
            perf['Trade Days'].append(date)
            continue

        # Check if we can go from short -> long.
        if sig == 1 and curr_pos['long'] == -1:
            # First close the short by buying.
            curr_bal +=  curr_pos[asset_2] * asset_2_p + curr_pos[asset_1] * asset_1_p

            # Enter the long position.
            curr_bal += asset_2_s * asset_2_p - asset_1_s * asset_1_p
            perf['Balance'].append(curr_bal)
            exp = abs(asset_2_s*asset_2_p) + abs(asset_1_p*asset_1_s)
            perf['Gross Exposure'].append(exp)

            curr_pos = {asset_1: asset_1_s, asset_2: -asset_2_s, f'{asset_1}_P': asset_1_p, f'{asset_2}_P': asset_2_p, 'gross': exp, 'long': 1}

            perf['Trades'].append(1)
            perf['Realized PnL'].append(paper_pnl)
            perf['Trade Days'].append(date)
            continue

        # Check if we can go from long -> short.
        if sig == -1 and curr_pos['long'] == 1:
            # First close the long position by selling.
            curr_bal +=  curr_pos[asset_1] * asset_1_p + curr_pos[asset_2] * asset_2_p

            # Enter the short position.
            curr_bal += asset_1_s * asset_1_p - asset_2_s * asset_2_p
            perf['Balance'].append(curr_bal)

            exp = abs(asset_2_s*asset_2_p) + abs(asset_1_p*asset_1_s)
            perf['Gross Exposure'].append(exp)

            curr_pos = {asset_1: -asset_1_s, asset_2: asset_2_s, f'{asset_1}_P': asset_1_p, f'{asset_2}_P': asset_2_p, 'gross': exp, 'long': -1}

            perf['Trades'].append(-1)
            perf['Realized PnL'].append(paper_pnl)
            perf['Trade Days'].append(date)
            continue

        # If we haven't done anything...
        perf['Balance'].append(perf['Balance'][-1] if len(perf['Balance']) > 0 else starting_bal)
        perf['Trades'].append(0)
        perf['Realized PnL'].append(0)
        perf['Gross Exposure'].append(abs(asset_2_s*asset_2_p) + abs(asset_1_p*asset_1_s))
        perf['Trade Days'].append(date)
    return pd.DataFrame(perf)