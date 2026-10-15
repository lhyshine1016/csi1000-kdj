"""
中证1000 建仓策略 + KDJ底背离（实时可用版）
底背离检测逻辑：
  低点在第 i-1 天，在第 i 天确认（价格开始回升后才触发）
  仅用历史数据，无前视偏差
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
OUT_IMG    = os.path.join(os.path.dirname(__file__), "strategy_divergence.png")

TROUGH_WIN  = 5    # 双向window：i必须是前后各N根里的价格最低点
MAX_GAP     = 60   # 两个低点最大间隔（根）
J_MAX       = 40   # 两个低点的J值都必须 < 40（超卖/中性区才有效）


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

# 周线数据
wk = fetch_kline(SYMBOL, scale="1200", n=300)
wk["rsi14"] = rsi(wk["close"], 14)
df = pd.merge_asof(df, wk[["date", "rsi14"]].rename(columns={"date": "wk_date"}),
                   left_on="date", right_on="wk_date", direction="backward")
print(f"周线 RSI 匹配完成，当前周RSI={df['rsi14'].iloc[-1]:.1f}")

PRICE_GAP_MIN = 0.03   # 同C期间两次买点最小价差
DAYS_EXEMPT   = 5      # 日线：距C日超过此值豁免"低于C价"过滤
WK_J_THRESH   = 10     # 周线C条件：J极小值 < 10
WK_K_MAX      = 60     # 周线A/B条件：K < 60
WK_WEEKS_EXEMPT = 2    # 周线：距C周超过此值豁免"低于C价"过滤

# ── 周线 KDJ C→A/B 状态机（先判C再判A/B，同周不比C价）────────────────────
wk["K"], wk["D"], wk["J"] = kdj93(wk)
wk_karr = wk["K"].values
wk_darr = wk["D"].values
wk_jarr = wk["J"].values

wk_watching       = False
wk_c_info         = None
wk_last_buy_price = None
wk_buys           = []
wk_c_signals      = []

for i in range(2, len(wk)):
    gd  = abs(wk_karr[i] - wk_darr[i])
    gp  = abs(wk_karr[i-1] - wk_darr[i-1])
    ks  = wk_karr[i] - wk_karr[i-1]
    ds  = wk_darr[i] - wk_darr[i-1]

    cond_a = (gd < gp and gd <= 5 and wk_karr[i] < WK_K_MAX and ks > ds and ks > 0)
    cond_b = (wk_karr[i-1] < wk_darr[i-1] and wk_karr[i] >= wk_darr[i]
              and wk_karr[i] < WK_K_MAX and gd <= 5)
    cond_c = (wk_jarr[i-1] < wk_jarr[i-2] and wk_jarr[i] > wk_jarr[i-1]
              and wk_jarr[i-1] < WK_J_THRESH)

    # 先判C
    same_week_as_c = False
    if cond_c:
        wk_watching       = True
        wk_last_buy_price = None
        wk_c_info = (i, wk.at[i,"date"], wk.at[i,"close"])
        wk_c_signals.append(wk_c_info)
        same_week_as_c = True

    # 再判A/B（同周触发C不阻止当周买点；同周跳过C价比较）
    if wk_watching and wk_c_info is not None and (cond_a or cond_b):
        bp = wk.at[i,"close"]
        ci, cd, cp = wk_c_info
        if wk_last_buy_price is None or abs(bp - wk_last_buy_price) / wk_last_buy_price >= PRICE_GAP_MIN:
            weeks_gap = i - ci
            skip = False if same_week_as_c else (bp < cp) and (weeks_gap <= WK_WEEKS_EXEMPT)
            label = "A" if cond_a else "B"
            wk_buys.append({
                "c_date": cd, "c_price": cp,
                "b_date": wk.at[i,"date"], "b_price": bp,
                "b_cond": label, "weeks": weeks_gap, "skip": skip,
            })
            wk_last_buy_price = bp

wk_pending_c = wk_c_info if wk_watching else None

df["K"], df["D"], df["J"] = kdj93(df)
karr  = df["K"].values
darr  = df["D"].values
jarr  = df["J"].values
close = df["close"].values

# ── 策略 C→A/B（警示后所有A/B都算买点）────────────────────────────────
# C触发 → 进入警示状态；期间所有A或B都是买点，直到下一个C才重置警示
watching        = False
c_info          = None
last_buy_price  = None   # 当前C期间最近一次买入价，用于3%过滤
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
        last_buy_price = None   # 新C重置
        c_info = (i, df.at[i,"date"], df.at[i,"close"], karr[i], gap_today)
        c_signals.append(c_info)
    elif watching and (cond_a or cond_b):
        bp = df.at[i,"close"]
        ci, cd, cp, ck, cg = c_info
        # 过滤1：与上次买点价差 < 3% 则跳过
        if last_buy_price is not None and abs(bp - last_buy_price) / last_buy_price < PRICE_GAP_MIN:
            continue
        label = "A" if cond_a else "B"
        days_gap = i - ci
        skip = (bp < cp) and (days_gap <= DAYS_EXEMPT)
        buys.append({
            "c_idx": ci, "c_date": cd, "c_price": cp,
            "b_idx": i,  "b_date": df.at[i,"date"], "b_price": bp,
            "b_cond": label, "days": days_gap,
            "skip": skip,
        })
        last_buy_price = bp
        # 不重置 watching —— 继续等更多A/B

pending_c = c_info if watching else None

# ── KDJ底背离检测（实时可用，无前视偏差）────────────────────────────────
# 在第 i 天，确认第 i-1 天是低点：
#   条件1：close[i] > close[i-1]               今天价格开始回升
#   条件2：close[i-1] == min(过去TROUGH_WIN根)  昨天是近期最低
#   条件3：J[i-1] < J_THRESH                   昨天J在低位
# 信号在第 i 天触发（已知昨天是低点），不依赖未来数据

confirmed_troughs = []   # trough_idx

# 双向 window：i 是前后各TROUGH_WIN根里的价格最低点
# 注：历史图表用；实时监测须在 i+TROUGH_WIN 天后才知道
for i in range(TROUGH_WIN, len(df) - TROUGH_WIN):
    window_p = close[i - TROUGH_WIN: i + TROUGH_WIN + 1]
    if close[i] == min(window_p):
        confirmed_troughs.append(i)

# 在已确认低点中找底背离对
divergences  = []    # (i1, i2) — 两个低点的下标
seen_i2      = set()

for k, i2 in enumerate(confirmed_troughs):
    for i1 in reversed(confirmed_troughs[:k]):
        if i2 - i1 > MAX_GAP:
            break
        if (close[i2] < close[i1]        # 价格更低（创新低）
                and jarr[i2] > jarr[i1]  # J更高（底背离）
                and jarr[i1] < J_MAX     # 前低J在超卖/中性区（<40）
                and jarr[i2] < J_MAX     # 后低J也在超卖/中性区（<40）
                and i2 not in seen_i2):
            divergences.append((i1, i2))
            seen_i2.add(i2)
            break

# ── 近1年过滤 ────────────────────────────────────────────────────────────
cutoff = pd.Timestamp("2024-09-01")
bt     = df[df["date"] >= cutoff].copy().reset_index(drop=True)

c1yr    = [c for c in c_signals if c[1] >= cutoff]
b1yr    = [b for b in buys     if b["c_date"] >= cutoff or b["b_date"] >= cutoff]
div1yr  = [(i1, i2) for i1, i2 in divergences if df.at[i2, "date"] >= cutoff]
wk_c1yr = [c for c in wk_c_signals if c[1] >= cutoff]
wk_b1yr = [b for b in wk_buys     if b["b_date"] >= cutoff]

print(f"\n日线 C警示: {len(c1yr)} 个  买入信号: {len(b1yr)} 个")
for b in b1yr:
    tag = "✗" if b["skip"] else "✓"
    print(f"  [{tag}] C:{b['c_date'].date()} → 买{b['b_cond']}:{b['b_date'].date()}  价={b['b_price']:.0f}  +{b['days']}天")
print(f"近3年KDJ底背离: {len(div1yr)} 个")
for i1, i2 in div1yr:
    d1, p1, j1 = df.at[i1,"date"], close[i1], jarr[i1]
    d2, p2, j2 = df.at[i2,"date"], close[i2], jarr[i2]
    print(f"  前低: {d1.date()} 价={p1:.0f} J={j1:.1f}  →  低点: {d2.date()} 价={p2:.0f} J={j2:.1f}  间隔{i2-i1}根")
print(f"\n周线 C警示: {len(wk_c1yr)} 个  买入信号: {len(wk_b1yr)} 个")
for b in wk_b1yr:
    tag = "✗" if b["skip"] else "✓"
    print(f"  [{tag}] C:{b['c_date'].date()} → 买{b['b_cond']}:{b['b_date'].date()}  价={b['b_price']:.0f}  +{b['weeks']}周")

# ── 绘图 ─────────────────────────────────────────────────────────────────
BG     = "#131722"; PANEL = "#1e222d"
COL_C  = "#ab47bc"
COL_A  = "#26a69a"
COL_B  = "#ff9800"
COL_OR = "#555566"
COL_DV = "#00e5ff"

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

# ── 辅助：错位标注（带虚线引出） ─────────────────────────────────────────
def stagger_annotate(ax, pts, col, marker, label_fn, size=100,
                     y_offsets=(55, -60, 80, -80), x_off=6):
    """pts: list of (date, price). 按日期顺序交替上下错位，虚线引出。"""
    for idx, (d, p) in enumerate(pts):
        yo = y_offsets[idx % len(y_offsets)]
        xo = x_off if yo > 0 else x_off
        ax.axvline(d, color=col, alpha=0.2, lw=1.0, zorder=3)
        ax.scatter([d], [p], c=col, s=size, zorder=6, marker=marker,
                   edgecolors="white", linewidths=0.6)
        ax.annotate(label_fn(d, p),
                    xy=(d, p), xytext=(xo, yo),
                    textcoords="offset points",
                    fontsize=6.5, color=col,
                    bbox=dict(boxstyle="round,pad=0.25", fc="#1e222d",
                              ec=col, alpha=0.88, lw=0.7),
                    arrowprops=dict(arrowstyle="-", color=col,
                                    lw=0.8, alpha=0.55, linestyle="dashed"))

# C警示（去重，只画竖线+星，不加文字标注，避免堆叠）
COL_SKIP = "#888899"
seen_c = set()
for ci in c1yr:
    cd, cp = ci[1], ci[2]
    if cd not in seen_c:
        ax0.axvline(cd, color=COL_C, alpha=0.18, lw=1.0, zorder=3)
        ax0.scatter([cd], [cp], c=COL_C, s=70, zorder=5, marker="*",
                    edgecolors="white", linewidths=0.4)
        seen_c.add(cd)

# 日线买入（只画非skip点，交替错位）
day_buys_vis = [(b["b_date"], b["b_price"], b["b_cond"], b["c_date"], b["c_price"])
                for b in b1yr if not b["skip"]]

for idx, (bd, bp, cond, cd, cp) in enumerate(day_buys_vis):
    bcol = COL_A if cond == "A" else COL_B
    yo   = [55, -62, 78, -78][idx % 4]
    ax0.axvline(bd, color=bcol, alpha=0.2, lw=1.0, zorder=3)
    ax0.scatter([bd], [bp], c=bcol, s=100, zorder=6, marker="^",
                edgecolors="white", linewidths=0.6)
    ax0.annotate(f"买{cond}  {bd.strftime('%m-%d')}\n{bp:.0f}",
                 xy=(bd, bp), xytext=(6, yo), textcoords="offset points",
                 fontsize=6.5, color=bcol,
                 bbox=dict(boxstyle="round,pad=0.25", fc="#1e222d", ec=bcol, alpha=0.88, lw=0.7),
                 arrowprops=dict(arrowstyle="-", color=bcol, lw=0.8, alpha=0.5, linestyle="dashed"))
    # C→买入虚线
    if cd >= cutoff:
        ax0.annotate("", xy=(bd, bp), xytext=(cd, cp),
                     arrowprops=dict(arrowstyle="-", color=COL_C,
                                     lw=0.7, alpha=0.2, linestyle="dotted"))

# 当前警示
if pending_c:
    pd_, pp = pending_c[1], pending_c[2]
    if pd_ >= cutoff:
        ax0.axvline(pd_, color=COL_C, alpha=0.5, lw=1.2, ls="--", zorder=4)
        ax0.scatter([pd_], [pp], c=COL_C, s=120, zorder=7, marker="*",
                    edgecolors="white", linewidths=0.7)
        ax0.annotate("⏳警示中", xy=(pd_, pp), xytext=(6, 14),
                     textcoords="offset points", fontsize=7, color=COL_C,
                     bbox=dict(boxstyle="round,pad=0.25", fc="#1e222d", ec=COL_C, alpha=0.9, lw=0.8),
                     arrowprops=dict(arrowstyle="-", color=COL_C, lw=0.7, alpha=0.5, linestyle="dashed"))

# 周线 C→A/B 信号（暖橙色系，六边形警示 + 五边形买点）
COL_WC = "#ff6f00"
COL_WA = "#ffca28"
COL_WB = "#ff8f00"

WC_Y = [55, -60, 75, -80]    # 周线C警示错位
WB_Y = [-65, 58, -85, 75]    # 周线买点错位（和C、日线买点错开）

# 周线C警示（全部标出，含没有买点的）
for idx, ci in enumerate(wk_c1yr):
    cd, cp = ci[1], ci[2]
    yo = WC_Y[idx % len(WC_Y)]
    ax0.axvline(cd, color=COL_WC, alpha=0.18, lw=1.2, ls=":", zorder=2)
    ax0.scatter([cd], [cp], c=COL_WC, s=80, zorder=5, marker="h",
                edgecolors="white", linewidths=0.5)
    ax0.annotate(f"周警  {cd.strftime('%m-%d')}\n{cp:.0f}",
                 xy=(cd, cp), xytext=(6, yo), textcoords="offset points",
                 fontsize=6.5, color=COL_WC,
                 bbox=dict(boxstyle="round,pad=0.25", fc="#1e222d", ec=COL_WC, alpha=0.88, lw=0.7),
                 arrowprops=dict(arrowstyle="-", color=COL_WC, lw=0.8, alpha=0.5, linestyle="dashed"))

# 周线买点（非过滤，错位引出）
wk_vis = [b for b in wk_b1yr if not b["skip"]]
for idx, b in enumerate(wk_vis):
    cd, cp = b["c_date"], b["c_price"]
    bd, bp = b["b_date"], b["b_price"]
    wcol = COL_WA if b["b_cond"] == "A" else COL_WB
    yo   = WB_Y[idx % len(WB_Y)]
    ax0.axvline(bd, color=wcol, alpha=0.2, lw=1.0, ls=":", zorder=3)
    ax0.scatter([bd], [bp], c=wcol, s=130, zorder=6, marker="p",
                edgecolors="white", linewidths=0.6)
    ax0.annotate(f"周买{b['b_cond']}  {bd.strftime('%m-%d')}\n{bp:.0f}",
                 xy=(bd, bp), xytext=(6, yo), textcoords="offset points",
                 fontsize=6.5, color=wcol,
                 bbox=dict(boxstyle="round,pad=0.25", fc="#1e222d", ec=wcol, alpha=0.88, lw=0.7),
                 arrowprops=dict(arrowstyle="-", color=wcol, lw=0.8, alpha=0.5, linestyle="dashed"))
    # C→买入连线
    if cd >= cutoff:
        ax0.annotate("", xy=(bd, bp), xytext=(cd, cp),
                     arrowprops=dict(arrowstyle="-", color=COL_WC,
                                     lw=0.7, alpha=0.18, linestyle="dotted"))

# 周线当前警示
if wk_pending_c and wk_pending_c[1] >= cutoff:
    wpd, wpp = wk_pending_c[1], wk_pending_c[2]
    ax0.axvline(wpd, color=COL_WC, alpha=0.5, lw=1.5, ls="--", zorder=4)
    ax0.scatter([wpd], [wpp], c=COL_WC, s=110, zorder=7, marker="h",
                edgecolors="white", linewidths=0.8)
    ax0.annotate("⏳周警", xy=(wpd, wpp), xytext=(6, -22),
                 textcoords="offset points", fontsize=7, color=COL_WC,
                 bbox=dict(boxstyle="round,pad=0.25", fc="#1e222d", ec=COL_WC, alpha=0.92, lw=0.9),
                 arrowprops=dict(arrowstyle="-", color=COL_WC, lw=0.8, alpha=0.6, linestyle="dashed"))

# 底背离：菱形标在低点（i2），连线到前低（i1）
for i1, i2 in div1yr:
    d1, p1, j1 = df.at[i1,"date"], close[i1], jarr[i1]
    d2, p2, j2 = df.at[i2,"date"], close[i2], jarr[i2]
    ax0.axvline(d2, color=COL_DV, alpha=0.3, lw=1.0, zorder=4)
    ax0.scatter([d2], [p2], c=COL_DV, s=110, zorder=7, marker="D",
                edgecolors="white", linewidths=0.6)
    ax0.annotate("底背离", (d2, p2), textcoords="offset points",
                 xytext=(4, -16), fontsize=6.5, color=COL_DV,
                 bbox=dict(boxstyle="round,pad=0.2", fc="#1e222d",
                           ec=COL_DV, alpha=0.85, lw=0.7))
    if d1 >= cutoff:
        ax0.plot([d1, d2], [p1, p2], color=COL_DV, lw=0.9, alpha=0.5, ls="--", zorder=3)

today = bt["date"].iloc[-1].strftime("%Y-%m-%d")
ax0.set_title(
    f"{NAME}({SYMBOL})  日线C★→A/B  +  周线C⬡→A/B  +  底背离◆  截至{today}",
    color="white", fontsize=11, pad=8)
ax0.set_ylabel("点位", fontsize=9, color="#787b86")
n_buy  = sum(1 for b in b1yr if not b["skip"])
n_skip = sum(1 for b in b1yr if b["skip"])
wk_n_buy  = sum(1 for b in wk_b1yr if not b["skip"])
wk_n_skip = sum(1 for b in wk_b1yr if b["skip"])
ax0.legend(handles=[
    Patch(fc=COL_C,    label=f"日线C警示(★)  {len(c1yr)}个"),
    Patch(fc=COL_A,    label=f"日线买A(▲)  {sum(1 for b in b1yr if b['b_cond']=='A' and not b['skip'])}次"),
    Patch(fc=COL_B,    label=f"日线买B(▲)  {sum(1 for b in b1yr if b['b_cond']=='B' and not b['skip'])}次"),
    Patch(fc=COL_SKIP, label=f"日线跳过✗  {n_skip}个"),
    Patch(fc=COL_WC,   label=f"周线C警示(⬡)  {len(wk_c1yr)}个"),
    Patch(fc=COL_WA,   label=f"周线买A/B(⬠)  {wk_n_buy}次"),
    Patch(fc=COL_DV,   label=f"底背离(◆)  {len(div1yr)}个"),
], facecolor=PANEL, labelcolor="white", fontsize=9, loc="upper left", ncol=2)

# KDJ面板
ax1.plot(dates, bt["K"], color="#e0e0e0", lw=0.9, label="K")
ax1.plot(dates, bt["D"], color="#ef5350",  lw=0.9, label="D")
ax1.plot(dates, bt["J"], color="#29b6f6",  lw=0.7, label="J", alpha=0.9)
ax1.axhline(0,        color="#ff9800", ls="--", lw=0.8, alpha=0.7)
ax1.axhline(20,       color="#26a69a", ls=":",  lw=0.6, alpha=0.6)
ax1.axhline(20, color=COL_DV, ls=":", lw=0.6, alpha=0.5)

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
