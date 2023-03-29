import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
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


