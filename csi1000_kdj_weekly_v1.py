"""
中证1000 周线 KDJ C→A/B 独立图
参数（确认版）：
  C条件: J[i-1] < 10（周线超卖阈值）
  A/B:   K < 45, |K-D| ≤ 5
  价差过滤: 3%
  低于C价豁免: 2 周
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

HEADERS  = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"}
SYMBOL   = "sh000852"
OUT_IMG  = os.path.join(os.path.dirname(__file__), "weekly_kdj.png")

# ── 可调参数 ────────────────────────────────────────────────────────────────
J_THRESH      = 10    # C条件: J[i-1] < J_THRESH
K_MAX         = 60    # A/B条件: K < K_MAX
KD_MAX        = 5     # A/B条件: |K-D| ≤ KD_MAX
PRICE_GAP_MIN = 0.03  # 同C期间连续买点最小价差
WEEKS_EXEMPT  = 2     # 低于C价豁免周数

# ── 颜色 ────────────────────────────────────────────────────────────────────
COL_C  = "#ff6f00"   # 周线C警示（深橙）
COL_A  = "#ffca28"   # 周线买A（金黄）
COL_B  = "#ff8f00"   # 周线买B（琥珀）
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
    lo = df["low"].rolling(9).min()
    hi = df["high"].rolling(9).max()
    rsv = (df["close"] - lo) / (hi - lo).replace(0, np.nan) * 100
    K = rsv.ewm(alpha=1/3, adjust=False).mean()
    D = K.ewm(alpha=1/3, adjust=False).mean()
    J = 3*K - 2*D
    return K, D, J

wk = fetch(SYMBOL, "1200", 400)
wk["K"], wk["D"], wk["J"] = kdj93(wk)
karr = wk["K"].values
darr = wk["D"].values
jarr = wk["J"].values

# ── 状态机 ──────────────────────────────────────────────────────────────────
watching       = False
c_info         = None
last_buy_price = None
c_signals      = []
buys           = []

for i in range(2, len(wk)):
    gd = abs(karr[i] - darr[i])
    gp = abs(karr[i-1] - darr[i-1])
    ks = karr[i] - karr[i-1]
    ds = darr[i] - darr[i-1]

    cond_a = (gd < gp and gd <= KD_MAX and karr[i] < K_MAX and ks > ds and ks > 0)
    cond_b = (karr[i-1] < darr[i-1] and karr[i] >= darr[i] and karr[i] < K_MAX and gd <= KD_MAX)
    cond_c = (jarr[i-1] < jarr[i-2] and jarr[i] > jarr[i-1] and jarr[i-1] < J_THRESH)

    # 先判C（重置警示周期）
    same_week_as_c = False
    if cond_c:
        watching       = True
        last_buy_price = None
        c_info = (i, wk.at[i, "date"], wk.at[i, "close"])
        c_signals.append(c_info)
        same_week_as_c = True

    # 再判A/B（同一周触发C不阻止当周A/B；同周则跳过C价比较）
    if watching and c_info is not None and (cond_a or cond_b):
        bp = wk.at[i, "close"]
        ci, cd, cp = c_info
        if last_buy_price is None or abs(bp - last_buy_price) / last_buy_price >= PRICE_GAP_MIN:
            weeks_gap = i - ci
            skip = False if same_week_as_c else (bp < cp) and (weeks_gap <= WEEKS_EXEMPT)
            label = "A" if cond_a else "B"
            buys.append({"c_idx": ci, "c_date": cd, "c_price": cp,
                         "b_idx": i,  "b_date": wk.at[i, "date"], "b_price": bp,
                         "b_cond": label, "weeks": weeks_gap, "skip": skip})
            last_buy_price = bp

pending_c = c_info if watching else None

# ── 汇报 ────────────────────────────────────────────────────────────────────
print(f"周线C警示: {len(c_signals)} 个")
for ci in c_signals:
    print(f"  {ci[1].date()}  收盘={ci[2]:.0f}  J={jarr[ci[0]-1]:.2f}")
print(f"\n周线买点（含过滤）: {len(buys)} 个")
for b in buys:
    flag = "✗过滤" if b["skip"] else "✓"
    print(f"  {flag} C:{b['c_date'].date()} → 买{b['b_cond']}:{b['b_date'].date()}  价={b['b_price']:.0f}  +{b['weeks']}周")

# ── 绘图 ────────────────────────────────────────────────────────────────────
cutoff = pd.Timestamp("2024-01-01")   # 周线图显示范围（近约3年）
wk_vis = wk[wk["date"] >= cutoff].copy()
dates  = wk_vis["date"].values

fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(16, 9),
                                gridspec_kw={"height_ratios": [3, 1]},
                                facecolor=COL_BG)
for ax in (ax0, ax1):
    ax.set_facecolor(COL_BG)
    ax.tick_params(colors="#aaaaaa", labelsize=8)
    for spine in ax.spines.values():
        spine.set_edgecolor("#333344")

# 上图：K线 + 信号
close_vis = wk_vis["close"].values
ax0.plot(wk_vis["date"], close_vis, color="#90caf9", lw=1.5, zorder=2, label="中证1000周线")
ax0.fill_between(wk_vis["date"], close_vis, close_vis.min() * 0.97,
                 color="#90caf9", alpha=0.06, zorder=1)
ax0.set_ylabel("指数", color="#aaaaaa", fontsize=9)
ax0.grid(axis="y", color="#222233", lw=0.5, zorder=0)
ax0.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
ax0.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
plt.setp(ax0.get_xticklabels(), rotation=30, ha="right", fontsize=7)

# C警示（只在cutoff之后的才画详细标注，之前的只画竖线）
seen_c = set()
for ci in c_signals:
    cd, cp, cidx = ci[1], ci[2], ci[0]
    if cd in seen_c or cd < cutoff:
        continue
    ax0.axvline(cd, color=COL_C, alpha=0.25, lw=1.2, ls=":", zorder=3)
    ax0.scatter([cd], [cp], c=COL_C, s=80, zorder=5, marker="h",
                edgecolors="white", linewidths=0.5)
    seen_c.add(cd)

# 买点（非过滤，错位引出）
vis_buys = [b for b in buys if not b["skip"] and b["b_date"] >= cutoff]
Y_OFFSETS = [60, -65, 85, -85]

for idx, b in enumerate(vis_buys):
    cd, cp = b["c_date"], b["c_price"]
    bd, bp = b["b_date"], b["b_price"]
    bcol = COL_A if b["b_cond"] == "A" else COL_B
    yo   = Y_OFFSETS[idx % len(Y_OFFSETS)]

    ax0.axvline(bd, color=bcol, alpha=0.22, lw=1.2, zorder=3)
    ax0.scatter([bd], [bp], c=bcol, s=130, zorder=6, marker="p",
                edgecolors="white", linewidths=0.7)
    ax0.annotate(
        f"周买{b['b_cond']}  {bd.strftime('%m-%d')}\n{bp:.0f}",
        xy=(bd, bp), xytext=(7, yo), textcoords="offset points",
        fontsize=7, color=bcol,
        bbox=dict(boxstyle="round,pad=0.3", fc="#1e222d", ec=bcol, alpha=0.92, lw=0.8),
        arrowprops=dict(arrowstyle="-", color=bcol, lw=0.9, alpha=0.6, linestyle="dashed"))

    # C→买入虚线
    if cd >= cutoff:
        ax0.annotate("", xy=(bd, bp), xytext=(cd, cp),
                     arrowprops=dict(arrowstyle="-", color=COL_C,
                                     lw=0.8, alpha=0.2, linestyle="dotted"))

# 过滤买点（小×，不标注）
for b in buys:
    if b["skip"] and b["b_date"] >= cutoff:
        ax0.scatter([b["b_date"]], [b["b_price"]], c="#666677", s=60,
                    zorder=5, marker="x", linewidths=1.5)

# 当前警示
if pending_c and pending_c[1] >= cutoff:
    wpd, wpp = pending_c[1], pending_c[2]
    ax0.axvline(wpd, color=COL_C, alpha=0.6, lw=1.5, ls="--", zorder=4)
    ax0.scatter([wpd], [wpp], c=COL_C, s=130, zorder=7, marker="h",
                edgecolors="white", linewidths=0.8)
    ax0.annotate("⏳警示中", xy=(wpd, wpp), xytext=(8, 14),
                 textcoords="offset points", fontsize=8, color=COL_C,
                 bbox=dict(boxstyle="round,pad=0.3", fc="#1e222d", ec=COL_C, alpha=0.95, lw=0.9),
                 arrowprops=dict(arrowstyle="-", color=COL_C, lw=0.8, alpha=0.6, linestyle="dashed"))

# 图例
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
legend_elems = [
    Line2D([0],[0], color="#90caf9", lw=1.5, label="中证1000周线"),
    Line2D([0],[0], marker="h", color="none", markerfacecolor=COL_C,
           markersize=8, label=f"C警示（J<{J_THRESH}）"),
    Line2D([0],[0], marker="p", color="none", markerfacecolor=COL_A,
           markersize=8, label="买A（KD收窄）"),
    Line2D([0],[0], marker="p", color="none", markerfacecolor=COL_B,
           markersize=8, label="买B（KD金叉）"),
    Line2D([0],[0], marker="x", color="#666677", markersize=7,
           markeredgewidth=1.5, label="买点已过滤"),
]
ax0.legend(handles=legend_elems, loc="upper left", framealpha=0.3,
           facecolor="#1e222d", edgecolor="#444455", fontsize=7.5, labelcolor="#cccccc")

today_str = pd.Timestamp.today().strftime("%Y-%m-%d")
ax0.set_title(f"中证1000 周线 KDJ C→A/B 策略  |  截至 {today_str}  |  "
              f"C阈值J<{J_THRESH}  K<{K_MAX}  |KD|≤{KD_MAX}  价差≥{int(PRICE_GAP_MIN*100)}%  豁免{WEEKS_EXEMPT}周",
              color="#dddddd", fontsize=9, pad=8)

# 下图：KDJ指标
wk_kdj = wk[wk["date"] >= cutoff]
ax1.plot(wk_kdj["date"], wk_kdj["K"], color="#26a69a", lw=1.1, label="K")
ax1.plot(wk_kdj["date"], wk_kdj["D"], color="#ef5350", lw=1.1, label="D")
ax1.plot(wk_kdj["date"], wk_kdj["J"], color="#ffca28", lw=0.9, alpha=0.8, label="J")
ax1.axhline(J_THRESH, color=COL_C, lw=0.8, ls="--", alpha=0.5, label=f"J={J_THRESH}阈值")
ax1.axhline(0, color="#ef5350", lw=0.8, ls="--", alpha=0.55, label="J=0阈值")
ax1.axhline(K_MAX, color="#90caf9", lw=0.6, ls="--", alpha=0.3, label=f"K={K_MAX}")
ax1.set_ylim(-30, 130)
ax1.set_ylabel("KDJ", color="#aaaaaa", fontsize=8)
ax1.grid(axis="y", color="#222233", lw=0.5)
ax1.legend(loc="upper right", framealpha=0.3, facecolor="#1e222d",
           edgecolor="#444455", fontsize=7, labelcolor="#cccccc", ncol=5)
ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
plt.setp(ax1.get_xticklabels(), rotation=30, ha="right", fontsize=7)

# C警示在KDJ图上也标竖线
for ci in c_signals:
    if ci[1] >= cutoff:
        ax1.axvline(ci[1], color=COL_C, alpha=0.2, lw=1.0, ls=":", zorder=3)

fig.tight_layout(rect=[0, 0, 1, 1])
plt.savefig(OUT_IMG, dpi=150, bbox_inches="tight", facecolor=COL_BG)
print(f"\n图已保存: {OUT_IMG}")
