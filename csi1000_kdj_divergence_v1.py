"""
中证1000 (sh000852) KDJ建仓策略 + KDJ底背离叠加图
版本：v1  日期：2026-10-14

策略逻辑：
  C触发 → 进入警示状态
  警示期间所有A或B都是买点，直到下一个C重置警示

过滤条件：
  1. 同一C期间两次买点价差 < 3% → 跳过（避免密集重复买入）
  2. 买入价 < C日收盘价 且 距C日 ≤5根 → 跳过（短期内跌破警示价不买）
     距C日 > 5根时豁免条件2（允许经过较长震荡后的买点）

底背离（仅用于辅助参考，不作为买入信号）：
  双向window=5（前后各5根价格最低点）
  两低点J值均 < 40
  价格创新低 + J值未创新低
  两低点间隔 ≤ 60根
  注意：双向window需等后5根收出才能确认，有约5个交易日滞后，不可实时操作
"""
import os, json, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import requests
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Patch
from matplotlib import font_manager

for font_path in ["/System/Library/Fonts/PingFang.ttc", "/System/Library/Fonts/STHeiti Light.ttc"]:
    if os.path.exists(font_path):
        font_manager.fontManager.addfont(font_path)
        matplotlib.rcParams["font.family"] = font_manager.FontProperties(fname=font_path).get_name()
        break
matplotlib.rcParams["axes.unicode_minus"] = False

HEADERS    = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"}
SYMBOL     = "sh000852"
NAME       = "中证1000"
OUT_IMG    = os.path.join(os.path.dirname(__file__), "csi1000_kdj_divergence.png")

TROUGH_WIN  = 5    # 双向window：i必须是前后各N根里的价格最低点
MAX_GAP     = 60   # 两个低点最大间隔（根）
J_MAX       = 40   # 两个低点的J值都必须 < 40
PRICE_GAP_MIN  = 0.03   # 同C期间两次买点最小价差
DAYS_EXEMPT    = 5      # 距C日超过此值则豁免"低于C价"过滤


def fetch_kline(symbol, scale="240", n=1100):
    r = requests.get(
        "http://money.finance.sina.com.cn/quotes_service/api/json_v2.php/CN_MarketData.getKLineData",
        params={"symbol": symbol, "scale": scale, "ma": "no", "datalen": str(n)},
        headers=HEADERS, timeout=20)
    rows = [{"date": pd.Timestamp(d["day"]),
             "open": float(d["open"]), "high": float(d["high"]),
             "low": float(d["low"]),  "close": float(d["close"]),
             "volume": int(d["volume"])} for d in json.loads(r.text)]
    return pd.DataFrame(rows).sort_values("date").reset_index(drop=True)


def rsi(series, period=14):
    delta = series.diff()
    gain  = delta.clip(lower=0)
    loss  = -delta.clip(upper=0)
    avg_g = gain.ewm(alpha=1/period, adjust=False).mean()
    avg_l = loss.ewm(alpha=1/period, adjust=False).mean()
    return 100 - 100 / (1 + avg_g / avg_l.replace(0, np.nan))


def kdj93(df):
    lo  = df["low"].rolling(9, min_periods=1).min()
    hi  = df["high"].rolling(9, min_periods=1).max()
    rsv = (df["close"] - lo) / (hi - lo).replace(0, np.nan) * 100
    K   = rsv.ewm(alpha=1/3, adjust=False).mean()
    D   = K.ewm(alpha=1/3, adjust=False).mean()
    return K, D, 3*K - 2*D


df = fetch_kline(SYMBOL, scale="240", n=1100)
print(f"日线数据: {df['date'].iloc[0].date()} ~ {df['date'].iloc[-1].date()}  共{len(df)}根")

# 周线 RSI(14)（暂保留，供后续研究用）
wk = fetch_kline(SYMBOL, scale="1200", n=300)
wk["rsi14"] = rsi(wk["close"], 14)
df = pd.merge_asof(df, wk[["date", "rsi14"]].rename(columns={"date": "wk_date"}),
                   left_on="date", right_on="wk_date", direction="backward")
print(f"周线 RSI 匹配完成，当前周RSI={df['rsi14'].iloc[-1]:.1f}")

df["K"], df["D"], df["J"] = kdj93(df)
karr  = df["K"].values
darr  = df["D"].values
jarr  = df["J"].values
close = df["close"].values

# ── 策略状态机：C→警示，A/B→买点 ────────────────────────────────────────
watching        = False
c_info          = None
last_buy_price  = None
c_signals = []
buys      = []

for i in range(2, len(df)):
    gap_today = abs(karr[i] - darr[i])
    gap_prev  = abs(karr[i-1] - darr[i-1])
    k_speed   = karr[i] - karr[i-1]
    d_speed   = darr[i] - darr[i-1]

    cond_a = (gap_today < gap_prev and gap_today <= 5 and karr[i] < 45
              and k_speed > d_speed and k_speed > 0)
    cond_b = (karr[i-1] < darr[i-1] and karr[i] >= darr[i]
              and karr[i] < 45 and gap_today <= 5)
    cond_c = (jarr[i-1] < jarr[i-2] and jarr[i] > jarr[i-1] and jarr[i-1] < 1)

    if cond_c:
        watching       = True
        last_buy_price = None
        c_info = (i, df.at[i,"date"], df.at[i,"close"], karr[i], gap_today)
        c_signals.append(c_info)
    elif watching and (cond_a or cond_b):
        bp = df.at[i,"close"]
        ci, cd, cp, ck, cg = c_info
        # 过滤1：同C期间两次买点价差 < 3%
        if last_buy_price is not None and abs(bp - last_buy_price) / last_buy_price < PRICE_GAP_MIN:
            continue
        label = "A" if cond_a else "B"
        days_gap = i - ci
        # 过滤2：买入价 < C价 且 距C ≤5根（超过5根豁免）
        skip = (bp < cp) and (days_gap <= DAYS_EXEMPT)
        buys.append({
            "c_idx": ci, "c_date": cd, "c_price": cp,
            "b_idx": i,  "b_date": df.at[i,"date"], "b_price": bp,
            "b_cond": label, "days": days_gap,
            "skip": skip,
        })
        last_buy_price = bp

pending_c = c_info if watching else None

# ── KDJ底背离检测（双向window，仅供图表参考）────────────────────────────
confirmed_troughs = []
for i in range(TROUGH_WIN, len(df) - TROUGH_WIN):
    window_p = close[i - TROUGH_WIN: i + TROUGH_WIN + 1]
    if close[i] == min(window_p):
        confirmed_troughs.append(i)

divergences = []
seen_i2     = set()
for k, i2 in enumerate(confirmed_troughs):
    for i1 in reversed(confirmed_troughs[:k]):
        if i2 - i1 > MAX_GAP:
            break
        if (close[i2] < close[i1]
                and jarr[i2] > jarr[i1]
                and jarr[i1] < J_MAX
                and jarr[i2] < J_MAX
                and i2 not in seen_i2):
            divergences.append((i1, i2))
            seen_i2.add(i2)
            break

# ── 回溯窗口（改这里调整展示范围）────────────────────────────────────────
cutoff = df["date"].iloc[-1] - pd.DateOffset(years=3)
bt     = df[df["date"] >= cutoff].copy().reset_index(drop=True)

c1yr   = [c for c in c_signals if c[1] >= cutoff]
b1yr   = [b for b in buys     if b["c_date"] >= cutoff or b["b_date"] >= cutoff]
div1yr = [(i1, i2) for i1, i2 in divergences if df.at[i2, "date"] >= cutoff]

print(f"\nC警示: {len(c1yr)} 个  买入信号（含过滤）: {len(b1yr)} 个  实际买入: {sum(1 for b in b1yr if not b['skip'])} 个")
for b in b1yr:
    tag = "✗跳过" if b["skip"] else "✓买入"
    print(f"  [{tag}] C:{b['c_date'].date()} → 买{b['b_cond']}:{b['b_date'].date()}  "
          f"价={b['b_price']:.0f}  +{b['days']}天")
print(f"KDJ底背离: {len(div1yr)} 个")
for i1, i2 in div1yr:
    d1, p1, j1 = df.at[i1,"date"], close[i1], jarr[i1]
    d2, p2, j2 = df.at[i2,"date"], close[i2], jarr[i2]
    print(f"  前低: {d1.date()} 价={p1:.0f} J={j1:.1f}  →  "
          f"低点: {d2.date()} 价={p2:.0f} J={j2:.1f}  间隔{i2-i1}根")

# ── 绘图 ─────────────────────────────────────────────────────────────────
BG     = "#131722"; PANEL = "#1e222d"
COL_C  = "#ab47bc"
COL_A  = "#26a69a"
COL_B  = "#ff9800"
COL_DV = "#ffd600"
COL_SKIP = "#888899"

fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(16, 10), sharex=True,
    gridspec_kw={"height_ratios": [3, 1.6]})
fig.patch.set_facecolor(BG)
for ax in (ax0, ax1):
    ax.set_facecolor(PANEL)
    for sp in ax.spines.values(): sp.set_color("#363a45")
    ax.tick_params(colors="#787b86", labelsize=8)
    ax.yaxis.label.set_color("#787b86")
    ax.grid(axis="x", color="#2a2e39", lw=0.5, ls="--")
    ax.grid(axis="y", color="#2a2e39", lw=0.4)

dates  = bt["date"]
closes = bt["close"]

ax0.plot(dates, closes, color="#2962ff", lw=1.4, zorder=2)
ax0.fill_between(dates, closes.min()*0.97, closes, alpha=0.07, color="#2962ff", zorder=1)

# C警示
seen_c = set()
for ci in c1yr:
    cd, cp = ci[1], ci[2]
    if cd not in seen_c:
        ax0.axvline(cd, color=COL_C, alpha=0.2, lw=1.0, zorder=3)
        ax0.scatter([cd], [cp], c=COL_C, s=90, zorder=5, marker="*",
                    edgecolors="white", linewidths=0.5)
        ax0.annotate("警示", (cd, cp), textcoords="offset points",
                     xytext=(-18, 6), fontsize=6, color=COL_C)
        seen_c.add(cd)

# 买入信号
for b in b1yr:
    cd, cp = b["c_date"], b["c_price"]
    bd, bp = b["b_date"], b["b_price"]
    bcol   = COL_A if b["b_cond"] == "A" else COL_B
    if b["skip"]:
        ax0.axvline(bd, color=COL_SKIP, alpha=0.15, lw=0.8, zorder=3)
        ax0.scatter([bd], [bp], c=COL_SKIP, s=100, zorder=6, marker="x", linewidths=1.8)
        ax0.annotate(f"✗{b['b_cond']}\n{bp:.0f}", (bd, bp),
                     textcoords="offset points", xytext=(4, 8), fontsize=6.5, color=COL_SKIP,
                     bbox=dict(boxstyle="round,pad=0.2", fc="#1e222d", ec=COL_SKIP, alpha=0.75, lw=0.7))
    else:
        ax0.axvline(bd, color=bcol, alpha=0.25, lw=1.0, zorder=3)
        ax0.scatter([bd], [bp], c=bcol, s=100, zorder=6, marker="^",
                    edgecolors="white", linewidths=0.6)
        ax0.annotate(f"买{b['b_cond']}\n{bp:.0f}", (bd, bp),
                     textcoords="offset points", xytext=(4, 8), fontsize=6.5, color=bcol,
                     bbox=dict(boxstyle="round,pad=0.2", fc="#1e222d", ec=bcol, alpha=0.85, lw=0.7))
    if cd >= cutoff:
        ax0.annotate("", xy=(bd, bp), xytext=(cd, cp),
                     arrowprops=dict(arrowstyle="-",
                                     color=COL_SKIP if b["skip"] else COL_C,
                                     lw=0.8, alpha=0.25, linestyle="dotted"))

# 当前警示中
if pending_c:
    pd_, pp = pending_c[1], pending_c[2]
    if pd_ >= cutoff:
        ax0.axvline(pd_, color=COL_C, alpha=0.5, lw=1.2, ls="--", zorder=4)
        ax0.scatter([pd_], [pp], c=COL_C, s=120, zorder=7, marker="*",
                    edgecolors="white", linewidths=0.7)
        ax0.annotate("⏳警示中", (pd_, pp), textcoords="offset points",
                     xytext=(4, 10), fontsize=7, color=COL_C,
                     bbox=dict(boxstyle="round,pad=0.25", fc="#1e222d",
                               ec=COL_C, alpha=0.9, lw=0.8))

# 底背离
for i1, i2 in div1yr:
    d1, p1 = df.at[i1,"date"], close[i1]
    d2, p2 = df.at[i2,"date"], close[i2]
    ax0.axvline(d2, color=COL_DV, alpha=0.3, lw=1.0, zorder=4)
    ax0.scatter([d2], [p2], c=COL_DV, s=110, zorder=7, marker="D",
                edgecolors="white", linewidths=0.6)
    ax0.annotate("底背离", (d2, p2), textcoords="offset points",
                 xytext=(4, -16), fontsize=6.5, color=COL_DV,
                 bbox=dict(boxstyle="round,pad=0.2", fc="#1e222d", ec=COL_DV, alpha=0.85, lw=0.7))
    if d1 >= cutoff:
        ax0.plot([d1, d2], [p1, p2], color=COL_DV, lw=0.9, alpha=0.5, ls="--", zorder=3)

today = bt["date"].iloc[-1].strftime("%Y-%m-%d")
n_buy  = sum(1 for b in b1yr if not b["skip"])
n_skip = sum(1 for b in b1yr if b["skip"])
ax0.set_title(
    f"{NAME}({SYMBOL})  C★警示→A/B买入 + KDJ底背离◆  截至{today}",
    color="white", fontsize=11, pad=8)
ax0.set_ylabel("点位", fontsize=9, color="#787b86")
ax0.legend(handles=[
    Patch(fc=COL_C,    label=f"C 建仓警示 (★)  共{len(c1yr)}个"),
    Patch(fc=COL_A,    label=f"买入A (▲)  共{sum(1 for b in b1yr if b['b_cond']=='A' and not b['skip'])}次"),
    Patch(fc=COL_B,    label=f"买入B (▲)  共{sum(1 for b in b1yr if b['b_cond']=='B' and not b['skip'])}次"),
    Patch(fc=COL_SKIP, label=f"低于C价跳过 ✗（距C≤{DAYS_EXEMPT}根）  共{n_skip}个"),
    Patch(fc=COL_DV,   label=f"KDJ底背离 (◆)  共{len(div1yr)}个"),
], facecolor=PANEL, labelcolor="white", fontsize=9, loc="upper left")

# KDJ面板
ax1.plot(dates, bt["K"], color="#e0e0e0", lw=0.9, label="K")
ax1.plot(dates, bt["D"], color="#ef5350",  lw=0.9, label="D")
ax1.plot(dates, bt["J"], color="#29b6f6",  lw=0.7, label="J", alpha=0.9)
ax1.axhline(0,  color="#ff9800", ls="--", lw=0.8, alpha=0.7)
ax1.axhline(20, color="#26a69a", ls=":",  lw=0.6, alpha=0.6)

for i1, i2 in div1yr:
    d1, j1 = df.at[i1,"date"], jarr[i1]
    d2, j2 = df.at[i2,"date"], jarr[i2]
    ax1.axvline(d2, color=COL_DV, alpha=0.35, lw=1.0)
    ax1.scatter([d2], [j2], c=COL_DV, s=60, zorder=6, marker="D",
                edgecolors="white", linewidths=0.5)
    if d1 >= cutoff:
        ax1.scatter([d1], [j1], c=COL_DV, s=40, zorder=5, marker="o",
                    edgecolors="white", linewidths=0.4, alpha=0.7)
        ax1.plot([d1, d2], [j1, j2], color=COL_DV, lw=1.2, alpha=0.6, zorder=4)

for ci in c1yr:
    ax1.axvline(ci[1], color=COL_C, alpha=0.15, lw=0.8)
for b in b1yr:
    bcol = COL_SKIP if b["skip"] else (COL_A if b["b_cond"] == "A" else COL_B)
    ax1.axvline(b["b_date"], color=bcol, alpha=0.2, lw=0.8)

ax1.set_ylabel("KDJ(9,3,3)", fontsize=9, color="#787b86")
ax1.legend(facecolor=PANEL, labelcolor="white", fontsize=8, ncol=4, loc="upper right")

ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
plt.setp(ax1.xaxis.get_majorticklabels(), rotation=25, color="#787b86", fontsize=8)

plt.tight_layout(h_pad=0.4)
plt.savefig(OUT_IMG, dpi=130, bbox_inches="tight", facecolor=BG)
print(f"\n图已保存: {OUT_IMG}")
