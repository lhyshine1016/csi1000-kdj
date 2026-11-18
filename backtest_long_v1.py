"""
中证1000 多头回测 v1
买入规则：日线 C→A/B（EMA3↑ + MACD≤0），每次信号买入 1 万元，信号当日收盘价成交
卖出规则：日线卖出 C→A/B（MA3↓），信号当日收盘价全部平仓
"""
import os, json, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import requests
from matplotlib import font_manager
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

for font_path in ["/System/Library/Fonts/PingFang.ttc", "/System/Library/Fonts/STHeiti Light.ttc"]:
    if os.path.exists(font_path):
        font_manager.fontManager.addfont(font_path)
        matplotlib.rcParams["font.family"] = font_manager.FontProperties(fname=font_path).get_name()
        break
matplotlib.rcParams["axes.unicode_minus"] = False

HEADERS = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"}
SYMBOL  = "sh000852"
NAME    = "中证1000"
AMOUNT  = 10_000      # 每次买入金额（元）
OUT_IMG = os.path.join(os.path.dirname(__file__), "backtest_long_v1.png")

# ── 参数 ─────────────────────────────────────────────────────────────────
# 买入
BUY_J_MAX   = 1
BUY_K_MAX   = 45
BUY_KD_MAX  = 5
PRICE_GAP   = 0.03    # 同C期间两次买点最小价差
DAYS_EXEMPT = 5

# 卖出
SELL_J_MIN  = 95
SELL_K_MIN  = 45
SELL_KD_MAX = 5


def fetch_kline(symbol, scale="240", n=1100):
    r = requests.get(
        "http://money.finance.sina.com.cn/quotes_service/api/json_v2.php/CN_MarketData.getKLineData",
        params={"symbol": symbol, "scale": scale, "ma": "no", "datalen": str(n)},
        headers=HEADERS, timeout=20)
    rows = [{"date":   pd.Timestamp(d["day"]),
             "open":   float(d["open"]),
             "high":   float(d["high"]),
             "low":    float(d["low"]),
             "close":  float(d["close"]),
             "volume": int(d["volume"])} for d in json.loads(r.text)]
    return pd.DataFrame(rows).sort_values("date").reset_index(drop=True)


def kdj93(df):
    lo  = df["low"].rolling(9, min_periods=1).min()
    hi  = df["high"].rolling(9, min_periods=1).max()
    rsv = (df["close"] - lo) / (hi - lo).replace(0, np.nan) * 100
    K   = rsv.ewm(alpha=1/3, adjust=False).mean()
    D   = K.ewm(alpha=1/3, adjust=False).mean()
    return K, D, 3*K - 2*D


# ── 加载数据 ─────────────────────────────────────────────────────────────
df = fetch_kline(SYMBOL, scale="240", n=1100)
print(f"日线数据: {df['date'].iloc[0].date()} ~ {df['date'].iloc[-1].date()}  共{len(df)}根")

df["K"], df["D"], df["J"] = kdj93(df)
df["ema3"]  = df["close"].ewm(span=3,  adjust=False).mean()
df["ema12"] = df["close"].ewm(span=12, adjust=False).mean()
df["ema26"] = df["close"].ewm(span=26, adjust=False).mean()
df["macd"]  = df["ema12"] - df["ema26"]
df["ma3"]   = df["close"].rolling(3).mean()

karr  = df["K"].values
darr  = df["D"].values
jarr  = df["J"].values
close = df["close"].values
opn   = df["open"].values
ema3  = df["ema3"].values
macd  = df["macd"].values
ma3   = df["ma3"].values
dates = df["date"].values

# ── 买入信号状态机 ────────────────────────────────────────────────────────
buy_signals = []   # list of (signal_idx, cond "A"/"B")

watching_b     = False
c_info_b       = None
last_buy_price = None

for i in range(3, len(df)):
    gd = abs(karr[i] - darr[i])
    gp = abs(karr[i-1] - darr[i-1])
    ks = karr[i] - karr[i-1]
    ds = darr[i] - darr[i-1]

    cond_bc = (jarr[i-1] < jarr[i-2] and jarr[i] > jarr[i-1] and jarr[i-1] < BUY_J_MAX)
    cond_ba = (gd < gp and gd <= BUY_KD_MAX and karr[i] < BUY_K_MAX
               and ks > ds and ks > 0
               and ema3[i] > ema3[i-1]
               and macd[i] <= 0)
    cond_bb = (karr[i-1] < darr[i-1] and karr[i] >= darr[i]
               and karr[i] < BUY_K_MAX and gd <= BUY_KD_MAX
               and ema3[i] > ema3[i-1]
               and macd[i] <= 0)

    if cond_bc:
        watching_b     = True
        last_buy_price = None
        c_info_b       = (i, df.at[i, "date"], close[i])
    elif watching_b and (cond_ba or cond_bb):
        bp = close[i]
        ci, cd, cp = c_info_b
        if last_buy_price is not None and abs(bp - last_buy_price) / last_buy_price < PRICE_GAP:
            continue
        days_gap = i - ci
        skip = (bp < cp) and (days_gap <= DAYS_EXEMPT)
        if not skip:
            label = "A" if cond_ba else "B"
            buy_signals.append((i, label, df.at[i, "date"], bp))
        last_buy_price = bp

# ── 卖出信号状态机 ────────────────────────────────────────────────────────
sell_signals = []  # list of (signal_idx, cond "A"/"B")

watching_s = False

for i in range(3, len(df)):
    gd = abs(karr[i] - darr[i])
    gp = abs(karr[i-1] - darr[i-1])
    ks = karr[i] - karr[i-1]

    cond_sc = (jarr[i-1] > jarr[i-2] and jarr[i] < jarr[i-1] and jarr[i-1] > SELL_J_MIN)
    cond_sa = (karr[i] > darr[i] and gd < gp and gd <= SELL_KD_MAX
               and karr[i] > SELL_K_MIN and ks < 0
               and ma3[i] < ma3[i-1])
    cond_sb = (karr[i-1] >= darr[i-1] and karr[i] < darr[i]
               and karr[i] > SELL_K_MIN
               and ma3[i] < ma3[i-1])

    if cond_sc:
        watching_s = True

    if watching_s:
        for label, flag in (("A", cond_sa), ("B", cond_sb)):
            if flag:
                sell_signals.append((i, label, df.at[i, "date"], close[i]))

# ── 回测撮合 ─────────────────────────────────────────────────────────────
# 信号当日收盘价成交
positions = []   # list of dict
trades    = []   # completed trades

# 把买卖信号合并排序（同日先买后卖）
events = ([(idx, "BUY",  lbl, dt, px) for idx, lbl, dt, px in buy_signals]
        + [(idx, "SELL", lbl, dt, px) for idx, lbl, dt, px in sell_signals])
events.sort(key=lambda x: (x[0], 0 if x[1] == "BUY" else 1))

for sig_idx, etype, label, sig_date, sig_close in events:
    exec_price = sig_close        # 信号当日收盘价
    exec_date  = sig_date

    if etype == "BUY":
        units = AMOUNT / exec_price
        positions.append({
            "entry_date":  exec_date,
            "entry_price": exec_price,
            "units":       units,
            "amount":      AMOUNT,
            "buy_cond":    label,
        })

    elif etype == "SELL" and positions:
        for pos in positions:
            proceeds  = pos["units"] * exec_price
            pnl       = proceeds - pos["amount"]
            ret_pct   = pnl / pos["amount"] * 100
            hold_days = (exec_date - pos["entry_date"]).days
            trades.append({
                "entry_date":  pos["entry_date"],
                "entry_price": pos["entry_price"],
                "exit_date":   exec_date,
                "exit_price":  exec_price,
                "buy_cond":    pos["buy_cond"],
                "sell_cond":   label,
                "hold_days":   hold_days,
                "amount_in":   pos["amount"],
                "proceeds":    proceeds,
                "pnl":         pnl,
                "ret_pct":     ret_pct,
            })
        positions = []   # 全部平仓

# ── 打印结果 ─────────────────────────────────────────────────────────────
print("\n" + "="*72)
print(f"  {'买入日':^12} {'买入价':>8} {'卖出日':^12} {'卖出价':>8} "
      f"{'持仓天':>6} {'买类型':>5} {'卖类型':>5} {'盈亏(元)':>10} {'收益率':>7}")
print("-"*72)
for t in trades:
    sign = "+" if t["pnl"] >= 0 else ""
    print(f"  {str(t['entry_date'])[:10]:^12} {t['entry_price']:>8.0f} "
          f"{str(t['exit_date'])[:10]:^12} {t['exit_price']:>8.0f} "
          f"{t['hold_days']:>6} {'买'+t['buy_cond']:>5} {'卖'+t['sell_cond']:>5} "
          f"{sign}{t['pnl']:>+9.0f} {sign}{t['ret_pct']:>+6.1f}%")

total_in   = sum(t["amount_in"] for t in trades)
total_pnl  = sum(t["pnl"]      for t in trades)
wins       = [t for t in trades if t["pnl"] > 0]
losses     = [t for t in trades if t["pnl"] <= 0]

print("="*72)
print(f"  已完成交易: {len(trades)} 笔  |  胜率: {len(wins)/len(trades)*100:.0f}%  "
      f"|  总投入: {total_in/1e4:.1f}万  |  总盈亏: {total_pnl:+.0f}元")
if wins:
    print(f"  盈利笔: {len(wins)}  均盈: {sum(t['ret_pct'] for t in wins)/len(wins):.1f}%  "
          f"最大盈: {max(t['ret_pct'] for t in wins):.1f}%")
if losses:
    print(f"  亏损笔: {len(losses)}  均亏: {sum(t['ret_pct'] for t in losses)/len(losses):.1f}%  "
          f"最大亏: {min(t['ret_pct'] for t in losses):.1f}%")
avg_hold = sum(t["hold_days"] for t in trades) / len(trades) if trades else 0
print(f"  平均持仓: {avg_hold:.0f} 天")

if positions:
    last_close = close[-1]
    last_date  = df["date"].iloc[-1]
    print(f"\n  未平仓位 {len(positions)} 个（截至 {str(last_date)[:10]} 收盘价 {last_close:.0f}）：")
    total_open_pnl = 0
    for pos in positions:
        upnl = (last_close - pos["entry_price"]) / pos["entry_price"] * pos["amount"]
        total_open_pnl += upnl
        sign = "+" if upnl >= 0 else ""
        print(f"    买{pos['buy_cond']} {str(pos['entry_date'])[:10]} "
              f"@ {pos['entry_price']:.0f}  浮动: {sign}{upnl:.0f}元 "
              f"({sign}{upnl/pos['amount']*100:.1f}%)")
    print(f"  持仓浮盈合计: {total_open_pnl:+.0f} 元")
    print(f"  已实现+浮盈合计: {total_pnl + total_open_pnl:+.0f} 元")

# ── 净值曲线 ─────────────────────────────────────────────────────────────
BG = "#131722"; PANEL = "#1e222d"
fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(14, 8), sharex=False,
    gridspec_kw={"height_ratios": [1.5, 1]})
fig.patch.set_facecolor(BG)
for ax in (ax0, ax1):
    ax.set_facecolor(PANEL)
    for sp in ax.spines.values(): sp.set_color("#363a45")
    ax.tick_params(colors="#787b86", labelsize=8)
    ax.grid(color="#2a2e39", lw=0.4)

# 上图：累计盈亏瀑布
if trades:
    cum_pnl = [0]
    for t in trades:
        cum_pnl.append(cum_pnl[-1] + t["pnl"])
    xs = list(range(len(cum_pnl)))
    ax0.step(xs, cum_pnl, where="post", color="#f44336", lw=1.5)
    ax0.fill_between(xs, 0, cum_pnl, step="post",
                     color="#f44336", alpha=0.15)
    ax0.axhline(0, color="#aaaaaa", lw=0.8, ls="--", alpha=0.6)
    for i, t in enumerate(trades):
        col = "#26a69a" if t["pnl"] > 0 else "#ef5350"
        ax0.bar(i + 1, t["pnl"], bottom=cum_pnl[i],
                color=col, alpha=0.5, width=0.6)
    ax0.set_xticks(range(1, len(trades)+1))
    ax0.set_xticklabels(
        [f"#{i+1}\n{str(t['entry_date'])[:7]}" for i, t in enumerate(trades)],
        fontsize=7, color="#787b86")
    ax0.set_ylabel("累计盈亏（元）", fontsize=9, color="#787b86")
    ax0.set_title(
        f"{NAME} 多头回测  共{len(trades)}笔  总盈亏 {total_pnl:+.0f}元  "
        f"胜率{len(wins)/len(trades)*100:.0f}%",
        color="white", fontsize=11)

# 下图：每笔收益率
if trades:
    colors = ["#26a69a" if t["pnl"] > 0 else "#ef5350" for t in trades]
    ax1.bar(range(1, len(trades)+1), [t["ret_pct"] for t in trades],
            color=colors, alpha=0.7)
    ax1.axhline(0, color="#aaaaaa", lw=0.8, ls="--", alpha=0.6)
    ax1.set_ylabel("单笔收益率（%）", fontsize=9, color="#787b86")
    ax1.set_xticks(range(1, len(trades)+1))
    ax1.set_xticklabels(
        [f"#{i+1}" for i in range(len(trades))],
        fontsize=7, color="#787b86")
    for i, t in enumerate(trades):
        col = "#26a69a" if t["pnl"] > 0 else "#ef5350"
        ax1.text(i+1, t["ret_pct"] + (0.3 if t["ret_pct"] >= 0 else -0.8),
                 f"{t['ret_pct']:+.1f}%", ha="center", fontsize=7, color=col)

plt.tight_layout(h_pad=0.6)
plt.savefig(OUT_IMG, dpi=130, bbox_inches="tight", facecolor=BG)
print(f"\n图已保存: {OUT_IMG}")
