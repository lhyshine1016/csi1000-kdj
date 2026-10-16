"""
中证1000 周线 KDJ 卖出警示独立图
条件：J[i-1] > J[i-2] 且 J[i] < J[i-1] 且 J[i-1] > WK_SELL_J
      即：J 超过阈值后确认回落，标在第 i 周收盘日
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
from matplotlib import font_manager

for fp in ["/System/Library/Fonts/PingFang.ttc", "/System/Library/Fonts/STHeiti Light.ttc"]:
    if os.path.exists(fp):
        font_manager.fontManager.addfont(fp)
        matplotlib.rcParams["font.family"] = font_manager.FontProperties(fname=fp).get_name()
        break
matplotlib.rcParams["axes.unicode_minus"] = False

HEADERS = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"}
SYMBOL  = "sh000852"
OUT_IMG = os.path.join(os.path.dirname(__file__), "weekly_sell.png")

# ── 可调参数 ────────────────────────────────────────────────────────────────
WK_SELL_J  = 70    # 卖C：J 超过此值后极值回落触发警示
WK_SELL_K  = 45    # 卖A/B：K > 此值才有效（过滤低位）
WK_KD_MAX  = 5     # 卖A：|KD| ≤ 此值

# ── 颜色 ────────────────────────────────────────────────────────────────────
COL_SC = "#ef5350"   # 卖C警示（红）★
COL_SA = "#f06292"   # 卖A KD高位收窄（玫红）▼
COL_SB = "#ff9800"   # 卖B KD死叉（橙）✕
COL_BG = "#131722"

# ── 数据 ────────────────────────────────────────────────────────────────────
def fetch(symbol, scale, n=400):
    r = requests.get(
        "http://money.finance.sina.com.cn/quotes_service/api/json_v2.php/CN_MarketData.getKLineData",
        params={"symbol": symbol, "scale": scale, "ma": "no", "datalen": str(n)},
        headers=HEADERS, timeout=20)
    rows = [{"date": pd.Timestamp(d["day"]),
             "open": float(d["open"]), "high": float(d["high"]),
             "low":  float(d["low"]),  "close": float(d["close"])}
            for d in json.loads(r.text)]
    return pd.DataFrame(rows).sort_values("date").reset_index(drop=True)

def kdj93(df):
    lo  = df["low"].rolling(9).min()
    hi  = df["high"].rolling(9).max()
    rsv = (df["close"] - lo) / (hi - lo).replace(0, np.nan) * 100
    K   = rsv.ewm(alpha=1/3, adjust=False).mean()
    D   = K.ewm(alpha=1/3, adjust=False).mean()
    return K, D, 3*K - 2*D

wk = fetch(SYMBOL, "1200", 400)
wk["K"], wk["D"], wk["J"] = kdj93(wk)
jarr = wk["J"].values

# ── 卖出状态机（对称买入 C→A/B）────────────────────────────────────────────
karr = wk["K"].values
darr = wk["D"].values

watching_s      = False
sc_info         = None   # (idx, date, close)
last_sell_price = None
sc_signals      = []     # 卖C警示
s_buys          = []     # 卖A/B（实际卖出点）

for i in range(2, len(wk)):
    gd = abs(karr[i] - darr[i])
    gp = abs(karr[i-1] - darr[i-1])
    ks = karr[i] - karr[i-1]
    ds = darr[i] - darr[i-1]

    # 卖C：J > WK_SELL_J 后极值回落
    cond_sc = (jarr[i-1] > jarr[i-2]
               and jarr[i] < jarr[i-1]
               and jarr[i-1] > WK_SELL_J)

    # 卖A：KD从高位收窄（买A的反向）
    #   K 仍在 D 上方（未死叉）、K在下降、KD收窄且趋近
    cond_sa = (karr[i] > darr[i]          # K仍在D上方
               and gd < gp                # KD差值收窄
               and gd <= WK_KD_MAX        # 已足够接近
               and karr[i] > WK_SELL_K   # K在高位
               and ks < 0)               # K在下降

    # 卖B：KD死叉 + K > 阈值
    cond_sb = (karr[i-1] >= darr[i-1]
               and karr[i] < darr[i]
               and karr[i] > WK_SELL_K)

    # 先判C
    same_week_as_sc = False
    if cond_sc:
        watching_s      = True
        last_sell_price = None
        sc_info = (i, wk.at[i, "date"], wk.at[i, "close"], jarr[i-1])
        sc_signals.append(sc_info)
        same_week_as_sc = True

    # 再判A/B（独立记录，同周可以各自触发）
    if watching_s and sc_info is not None:
        sp = wk.at[i, "close"]
        ci, cd, cp, pj = sc_info
        for label, flag in (("A", cond_sa), ("B", cond_sb)):
            if flag:
                s_buys.append({
                    "c_date": cd, "c_price": cp, "peak_j": pj,
                    "s_date": wk.at[i, "date"], "s_price": sp,
                    "s_cond": label,
                    "k": karr[i], "d": darr[i],
                    "weeks": i - ci,
                })
        if cond_sa or cond_sb:
            last_sell_price = sp

# ── 汇报 ─────────────────────────────────────────────────────────────────────
cutoff = pd.Timestamp("2024-01-01")
sc_vis = [s for s in sc_signals if s[1] >= cutoff]
sb_vis = [s for s in s_buys   if s["s_date"] >= cutoff]

print(f"卖C警示（J>{WK_SELL_J}极值回落）: {len(sc_signals)}个 / 2024年以来: {len(sc_vis)}个")
print(f"{'日期':<14} {'收盘':>7} {'峰值J':>8}")
print("-" * 35)
for s in sc_vis:
    print(f"{str(s[1].date()):<14} {s[2]:>7.0f} {s[3]:>8.1f}")

print(f"\n卖A/B信号（K>{WK_SELL_K}）: {len(s_buys)}个 / 2024年以来: {len(sb_vis)}个")
print(f"{'日期':<14} {'类型':>5} {'收盘':>7} {'K':>7} {'D':>7} {'对应C'}")
print("-" * 60)
for s in sb_vis:
    print(f"{str(s['s_date'].date()):<14} 卖{s['s_cond']:>2}  {s['s_price']:>7.0f}"
          f" {s['k']:>7.1f} {s['d']:>7.1f}  ←C:{s['c_date'].date()}")

# ── 绘图 ─────────────────────────────────────────────────────────────────────
wk_vis = wk[wk["date"] >= cutoff].copy()

fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(16, 9),
                                gridspec_kw={"height_ratios": [3, 1]},
                                facecolor=COL_BG)
for ax in (ax0, ax1):
    ax.set_facecolor(COL_BG)
    ax.tick_params(colors="#aaaaaa", labelsize=8)
    for spine in ax.spines.values():
        spine.set_edgecolor("#333344")
    ax.grid(axis="y", color="#222233", lw=0.5)
    ax.grid(axis="x", color="#222233", lw=0.4, ls="--")

# 上图：价格线
close_vis = wk_vis["close"].values
ax0.plot(wk_vis["date"], close_vis, color="#90caf9", lw=1.5, zorder=2, label="中证1000周线")
ax0.fill_between(wk_vis["date"], close_vis, close_vis.min() * 0.97,
                 color="#90caf9", alpha=0.06, zorder=1)
ax0.set_ylabel("指数", color="#aaaaaa", fontsize=9)
ax0.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
ax0.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
plt.setp(ax0.get_xticklabels(), rotation=30, ha="right", fontsize=7)

# 卖C警示（红色★，只标竖线+点，不堆叠文字）
for s in sc_vis:
    sd, sp = s[1], s[2]
    ax0.axvline(sd, color=COL_SC, alpha=0.15, lw=1.0, ls=":", zorder=2)
    ax0.scatter([sd], [sp], c=COL_SC, s=75, zorder=5, marker="*",
                edgecolors="white", linewidths=0.4)

# 卖A（玫红▼，KD高位收窄，错位标注）
SA_Y = [-58, 70, -82, 88]
sa_vis = [s for s in sb_vis if s["s_cond"] == "A"]
for idx, s in enumerate(sa_vis):
    sd, sp = s["s_date"], s["s_price"]
    yo = SA_Y[idx % len(SA_Y)]
    ax0.axvline(sd, color=COL_SA, alpha=0.22, lw=1.2, zorder=3)
    ax0.scatter([sd], [sp], c=COL_SA, s=120, zorder=6, marker="v",
                edgecolors="white", linewidths=0.6)
    ax0.annotate(f"卖A  {sd.strftime('%m-%d')}\n{sp:.0f}",
                 xy=(sd, sp), xytext=(6, yo), textcoords="offset points",
                 fontsize=6.5, color=COL_SA,
                 bbox=dict(boxstyle="round,pad=0.25", fc="#1e222d", ec=COL_SA, alpha=0.90, lw=0.7),
                 arrowprops=dict(arrowstyle="-", color=COL_SA, lw=0.8, alpha=0.5, linestyle="dashed"))

# 卖B（橙色✕，KD死叉，错位标注）
SB_Y = [62, -65, 82, -85]
sb_only = [s for s in sb_vis if s["s_cond"] == "B"]
for idx, s in enumerate(sb_only):
    sd, sp = s["s_date"], s["s_price"]
    yo = SB_Y[idx % len(SB_Y)]
    ax0.axvline(sd, color=COL_SB, alpha=0.22, lw=1.2, zorder=3)
    ax0.scatter([sd], [sp], c=COL_SB, s=110, zorder=6, marker="X",
                edgecolors="white", linewidths=0.5)
    ax0.annotate(f"卖B  {sd.strftime('%m-%d')}\n{sp:.0f}",
                 xy=(sd, sp), xytext=(6, yo), textcoords="offset points",
                 fontsize=6.5, color=COL_SB,
                 bbox=dict(boxstyle="round,pad=0.25", fc="#1e222d", ec=COL_SB, alpha=0.90, lw=0.7),
                 arrowprops=dict(arrowstyle="-", color=COL_SB, lw=0.8, alpha=0.5, linestyle="dashed"))

# C→卖出连线
for s in sb_vis:
    cd, cp = s["c_date"], s["c_price"]
    sd, sp = s["s_date"], s["s_price"]
    if cd >= cutoff:
        ax0.annotate("", xy=(sd, sp), xytext=(cd, cp),
                     arrowprops=dict(arrowstyle="-", color=COL_SC,
                                     lw=0.7, alpha=0.15, linestyle="dotted"))

from matplotlib.lines import Line2D
n_sa = len(sa_vis)
n_sb = len(sb_only)
ax0.legend(handles=[
    Line2D([0],[0], color="#90caf9", lw=1.5, label="中证1000周线"),
    Line2D([0],[0], marker="*", color="none", markerfacecolor=COL_SC,
           markersize=10, label=f"卖C警示 ★  J>{WK_SELL_J}极值回落  {len(sc_vis)}个"),
    Line2D([0],[0], marker="v", color="none", markerfacecolor=COL_SA,
           markersize=9,  label=f"卖A ▼  KD高位收窄  {n_sa}个"),
    Line2D([0],[0], marker="X", color="none", markerfacecolor=COL_SB,
           markersize=9,  label=f"卖B ✕  KD死叉K>{WK_SELL_K}  {n_sb}个"),
], loc="upper left", framealpha=0.3, facecolor="#1e222d",
   edgecolor="#444455", fontsize=8, labelcolor="#cccccc")

today_str = pd.Timestamp.today().strftime("%Y-%m-%d")
ax0.set_title(
    f"中证1000 周线 KDJ 卖出  |  截至 {today_str}  |  "
    f"卖C:J>{WK_SELL_J}极值回落  卖A:KD高位收窄  卖B:死叉K>{WK_SELL_K}",
    color="#dddddd", fontsize=9, pad=8)

# 下图：KDJ
wk_kdj = wk[wk["date"] >= cutoff]
ax1.plot(wk_kdj["date"], wk_kdj["K"], color="#26a69a", lw=1.1, label="K")
ax1.plot(wk_kdj["date"], wk_kdj["D"], color="#ef5350", lw=1.1, label="D")
ax1.plot(wk_kdj["date"], wk_kdj["J"], color="#ffca28", lw=0.9, alpha=0.85, label="J")
ax1.axhline(WK_SELL_J, color=COL_SC, lw=0.9, ls="--", alpha=0.6, label=f"J={WK_SELL_J}卖C阈值")
ax1.axhline(WK_SELL_K, color=COL_SB, lw=0.7, ls="--", alpha=0.45, label=f"K={WK_SELL_K}死叉阈值")
ax1.axhline(100, color=COL_SC, lw=0.5, ls=":", alpha=0.3, label="J=100")
ax1.set_ylim(-30, 150)
ax1.set_ylabel("KDJ", color="#aaaaaa", fontsize=8)
ax1.legend(loc="upper right", framealpha=0.3, facecolor="#1e222d",
           edgecolor="#444455", fontsize=7, labelcolor="#cccccc", ncol=5)
ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
plt.setp(ax1.get_xticklabels(), rotation=30, ha="right", fontsize=7)

# KDJ图：卖C竖线
for s in sc_vis:
    ax1.axvline(s[1], color=COL_SC, alpha=0.18, lw=1.0, ls=":", zorder=3)
    ax1.scatter([s[1]], [s[3]], c=COL_SC, s=30, zorder=5,
                marker="*", edgecolors="none", alpha=0.7)

# KDJ图：卖A▼
for s in sa_vis:
    ax1.axvline(s["s_date"], color=COL_SA, alpha=0.2, lw=1.0, ls=":", zorder=3)
    ax1.scatter([s["s_date"]], [s["k"]], c=COL_SA, s=30, zorder=5,
                marker="v", edgecolors="none", alpha=0.8)

# KDJ图：卖B✕
for s in sb_only:
    ax1.axvline(s["s_date"], color=COL_SB, alpha=0.2, lw=1.0, ls=":", zorder=3)
    ax1.scatter([s["s_date"]], [s["k"]], c=COL_SB, s=30, zorder=5,
                marker="X", edgecolors="none", alpha=0.8)

fig.tight_layout()
plt.savefig(OUT_IMG, dpi=150, bbox_inches="tight", facecolor=COL_BG)
print(f"\n图已保存: {OUT_IMG}")
