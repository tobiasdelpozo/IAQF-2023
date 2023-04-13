import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import scipy.stats as stats
import statsmodels.api as sm
import matplotlib.gridspec as gridspec
pio.renderers.default = "svg"

def plot_trades_single_spread(spread, trades, name):
    fig = make_subplots(rows=2, cols=1)
    fig.add_trace(go.Scatter(x=spread.index, y=spread, name=name), row=1, col=1)
    buys = spread.loc[trades[trades['side'] == 1].index]
    sells = spread.loc[trades[trades['side'] == -1].index]
    fig.add_trace(go.Scatter(x=buys.index, y=buys, mode='markers', name='buys', marker=dict(color='red'),), row=1, col=1)
    fig.add_trace(go.Scatter(x=sells.index, y=sells, mode='markers', name='sells', marker=dict(color='green'),), row=1, col=1)
    fig.add_trace(go.Bar(trades.index, trades['notional'], name='notional'), row=2, col=1)
    fig.add_trace(go.Bar(trades.index, trades['pnl'], name='pnl', base=0), row=2, col=1)
    fig.show()


def standardize_log(price, n = 0):
    result = np.log(price/price.iloc[0])
    return result

def standardize_logma(price, n):
    result =  np.log(price / price.rolling(n).mean()).dropna()
    # result = (np.log(price) - np.log(price).rolling(n).mean()).dropna()
    return result

def get_dist(price, t, n, start, end, standardize = standardize_logma):
    sample_price = price[t]
    data = standardize(sample_price, n)
    
    sample = data.loc[start:end]
    l, sc = stats.norm.fit(sample)
    rv_sample = pd.Series(stats.norm.rvs(size = int(1e6), loc = l , scale = sc))
    fig = plt.figure(figsize=(10, 5))
    sample.plot(kind= 'kde', ylabel = 'Density', title = f'Distribution of {t}, n = {n}')
    sample.plot(kind= 'hist', bins=100, alpha=0.5, density=True)
    rv_sample.plot(kind= 'kde', color='black', xlabel = 'Log Returns')
    fig.legend(['Sample Gaussian KDE', 'Histogram', 'Normal Distribution'], loc='upper right')
    fig.tight_layout()
    m = sample.mean()
    std = sample.std()
    plt.xlim(m - 7*std, m+7*std)
    plt.close()
    return fig


def std_comparison(prices, start, end, test_end, n, k=4):
    data_ma = standardize_logma(prices, n)
    data_log = standardize_log(prices)
    fig, ax = plt.subplots(figsize=(10, 4),  nrows= 1, ncols= 2, squeeze=False)

    in_sample_ma = data_ma.loc[start:end]
    out_sample_ma = data_ma.loc[end:test_end]
    in_sample_ma.plot(kind= 'hist', ylabel = 'Histogram', title = f'Log(Price/Moving Average)', ax=ax[0,0], color = 'maroon', bins=100, alpha = 0.5)
    out_sample_ma.plot(kind= 'hist', ylabel = 'Histogram', ax=ax[0,0], color = 'cornflowerblue', bins = 100, alpha = 0.5)
    
    in_sample_log = data_log.loc[start:end]
    out_sample_log = data_log.loc[end:test_end]
    in_sample_log.plot(kind= 'hist', xlabel = 'Log Returns', ax=ax[0,1], title = f'Log(Price/Initial Price)', color = 'maroon', bins = 100, alpha = 0.5)
    out_sample_log.plot(kind= 'hist', xlabel = 'Log Returns', ax=ax[0,1], color = 'cornflowerblue', bins = 100, alpha = 0.5)
    fig.suptitle(f'Distribution', fontsize=20, y=1)
    fig.legend(['Formulation Period Distribution', 'Trading Period Distribution'],
               loc='upper center', bbox_to_anchor=(0.5, 1.02),
               fancybox=True, shadow=True, ncol=5)
    fig.tight_layout()
    plt.close()
    return fig

def plot_joint_dist(joint_dist, pair_label, cop_name):
    sample = joint_dist.rvs(int(5e3), random_state=20210801)
    h = sns.jointplot(x=sample[:, 0], y=sample[:, 1], kind="scatter", height=4)
    _ = h.set_axis_labels(pair_label[0], pair_label[1])
    _ = h.fig.suptitle(f'Joint Distribution with {cop_name} Copula', y=1)
    plt.close()
    return h


def plot_copula(copula_fitter, pair, copula_type):
    joint_dist = copula_fitter.fit_copula(copula_type, joint_dist = True)
    fig = plot_joint_dist(joint_dist, pair, copula_type)
    plt.close()
    return fig


class SeabornFig2Grid():

    def __init__(self, seaborngrid, fig,  subplot_spec):
        self.fig = fig
        self.sg = seaborngrid
        self.subplot = subplot_spec
        if isinstance(self.sg, sns.axisgrid.FacetGrid) or \
            isinstance(self.sg, sns.axisgrid.PairGrid):
            self._movegrid()
        elif isinstance(self.sg, sns.axisgrid.JointGrid):
            self._movejointgrid()
        self._finalize()

    def _movegrid(self):
        """ Move PairGrid or Facetgrid """
        self._resize()
        n = self.sg.axes.shape[0]
        m = self.sg.axes.shape[1]
        self.subgrid = gridspec.GridSpecFromSubplotSpec(n,m, subplot_spec=self.subplot)
        for i in range(n):
            for j in range(m):
                self._moveaxes(self.sg.axes[i,j], self.subgrid[i,j])

    def _movejointgrid(self):
        """ Move Jointgrid """
        h= self.sg.ax_joint.get_position().height
        h2= self.sg.ax_marg_x.get_position().height
        r = int(np.round(h/h2))
        self._resize()
        self.subgrid = gridspec.GridSpecFromSubplotSpec(r+1,r+1, subplot_spec=self.subplot)

        self._moveaxes(self.sg.ax_joint, self.subgrid[1:, :-1])
        self._moveaxes(self.sg.ax_marg_x, self.subgrid[0, :-1])
        self._moveaxes(self.sg.ax_marg_y, self.subgrid[1:, -1])

    def _moveaxes(self, ax, gs):
        #https://stackoverflow.com/a/46906599/4124317
        ax.remove()
        ax.figure=self.fig
        self.fig.axes.append(ax)
        self.fig.add_axes(ax)
        
        ax._subplotspec = gs
        ax.set_position(gs.get_position(self.fig))
        ax.set_subplotspec(gs)

    def _finalize(self):
        plt.close(self.sg.fig)
        self.fig.canvas.mpl_connect("resize_event", self._resize)
        self.fig.canvas.draw()

    def _resize(self, evt=None):
        self.sg.fig.set_size_inches(self.fig.get_size_inches())
        
        
def plot_linear(formation_price, price, pair, start, trading_end, i):
    beta = sm.OLS(formation_price[pair[0]], sm.add_constant(formation_price[pair[1]])).fit().params[1]
    y = (price[pair[0]] - beta * price[pair[1]]).loc[start:trading_end]

    x_minutes = y.index
    x_plot = range(0, len(x_minutes))
    x_labels = x_minutes[~x_minutes.strftime("%d").duplicated()]
    x_ticks = np.where(np.in1d(x_minutes, x_labels))[0]

    fig, ax = plt.subplots(figsize=(10,5))
    ax.plot(x_plot, y)
    ax.set_xticks(x_ticks, x_labels.strftime("%d"), rotation=45)

    ax.grid(linestyle='--', alpha=0.8)
    ax.set(title=f'Pair {i+1} {pair[0]} - {beta:.2f} * {pair[1]}', ylabel='Price', xlabel='March 2022')
    plt.show()
    
def plot_signal(sig1, sig2, trading_price, pair):
    fig = plt.figure(figsize=(15,3))
    x_minutes = trading_price.index
    x_plot = range(0, len(x_minutes))
    x_labels = x_minutes[~x_minutes.strftime("%d").duplicated()]
    x_ticks = np.where(np.in1d(x_minutes, x_labels))[0]
    plt.plot(x_plot, sig1[pair] )
    plt.plot(x_plot, sig2[pair])
    plt.xticks(x_ticks, x_labels.strftime("%d-%b-%y"), rotation=45)
    plt.axhline(0.95, color='r')
    plt.axhline(0.90, color='tomato')
    plt.axhline(0.85, color='salmon')
    plt.axhline(0.8, color='cyan', linestyle='--')
    plt.axhline(0.2, color='cyan', linestyle='--')
    plt.title(f'Copula Probabilities of Spread: {pair[0]}-{pair[1]}')
    plt.axhline(0.05, color='darkgreen')
    plt.axhline(0.10, color='green')
    plt.axhline(0.15, color='lime')
    plt.legend([f'P({pair[0]}<=x|{pair[1]}=y)', f'P({pair[1]}<=y|{pair[0]}=x)'])
    plt.show()
    
def plot_backtested(pair, trading_price, signals, beta, sig1, sig2, backtest_res):
    pair_leg_1 = pair[0]
    pair_leg_2 = pair[1]

    fig, (a, axes, ax) = plt.subplots(3,1,figsize=(16,9))

    buys = signals[signals['Signal'] > 0].index
    sells = signals[signals['Signal'] < 0].index

    spread = trading_price[pair_leg_1] - trading_price[pair_leg_2] * beta

    x_minutes = spread.index
    x_plot = range(0, len(x_minutes))
    x_labels = x_minutes[~x_minutes.strftime("%d").duplicated()]
    x_ticks = np.where(np.in1d(x_minutes, x_labels))[0]

    xmins = pd.Series(x_minutes.values)
    x_buys =  xmins[xmins.isin(buys)].index
    x_sells = xmins[xmins.isin(sells)].index

    axes.plot(x_plot, spread, label='Spread')
    axes.plot(x_buys, spread.loc[buys], '^', markersize=6, color='g', label='Buy')
    axes.plot(x_sells, spread.loc[sells], 'v', markersize=6, color='r', label='Sell')
    axes.set_xticks(x_ticks, x_labels.strftime("%d-%b-%y"), rotation=45)
    axes.set_title(f'Pair Spread: {pair_leg_1}-{pair_leg_2}')
    axes.legend()

    a.plot(x_plot, sig1[pair])
    a.plot(x_plot, sig2[pair])
    a.set_xticks(x_ticks, x_labels.strftime("%d-%b-%y"), rotation=45)
    a.axhline(0.95, color='r')
    a.axhline(0.90, color='tomato')
    a.axhline(0.85, color='salmon')
    a.axhline(0.8, color='cyan', linestyle='--')
    a.axhline(0.2, color='cyan', linestyle='--')
    a.set_title(f'Copula Probabilities of Spread: {pair_leg_1}-{pair_leg_2}')

    a.axhline(0.05, color='darkgreen')
    a.axhline(0.10, color='green')
    a.axhline(0.15, color='lime')
    a.legend([f'P({pair_leg_1}<=x|{pair_leg_2}=y)', f'P({pair_leg_2}<=y|{pair_leg_1}=x)'])

    ax.plot(x_plot, backtest_res['Portfolio'] - 50_000)
    ax.set_xticks(x_ticks, x_labels.strftime("%d-%b-%y"), rotation=45)
    ax.set_title(f'Cumulative PnL ($): {pair_leg_1}-{pair_leg_2}')
    ax.axhline(0, color='r', linestyle='--')

    fig.subplots_adjust(hspace=0.6)
    plt.show()
    
    