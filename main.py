import base64
import json
import logging
import math
import os
import random
import statistics
import threading
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import gspread
from fastapi import FastAPI, HTTPException


# ----------------------------
# Helpers
# ----------------------------

def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def safe_float(value: Any, default: Optional[float] = None) -> Optional[float]:
    if value is None:
        return default
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    if text == "":
        return default
    try:
        return float(text)
    except Exception:
        return default


def safe_int(value: Any, default: Optional[int] = None) -> Optional[int]:
    f = safe_float(value, None)
    if f is None:
        return default
    try:
        return int(f)
    except Exception:
        return default


def boolish(value: Any) -> bool:
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def env_str(name: str, default: Optional[str] = None) -> str:
    value = os.getenv(name, default)
    if value is None or value.strip() == "":
        raise ValueError(f"Missing required env var: {name}")
    return value.strip()


def env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    return int(value) if value and value.strip() else int(default)


def env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    return float(value) if value and value.strip() else float(default)


def env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return default
    return boolish(value)


def clip(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def col_to_a1(n: int) -> str:
    s = ""
    while n > 0:
        n, r = divmod(n - 1, 26)
        s = chr(65 + r) + s
    return s


def _safe_log10(x: float) -> float:
    return math.log10(max(x, 1e-9))


def sample_wide_range(min_value: float, max_value: float, anchor_value: float, tail_prob: float = 0.15) -> float:
    min_value = max(1e-9, float(min_value))
    max_value = max(min_value, float(max_value))
    anchor_value = clip(float(anchor_value), min_value, max_value)

    # Preserve some true full-range exploration.
    if random.random() < tail_prob:
        return random.uniform(min_value, max_value)

    # When the range is very wide, explore in log space so extremes do not dominate.
    ratio = max_value / min_value if min_value > 0 else 1.0
    if ratio >= 20.0:
        lo = _safe_log10(min_value)
        hi = _safe_log10(max_value)
        mid = _safe_log10(anchor_value)
        return 10 ** random.triangular(lo, hi, mid)

    return random.triangular(min_value, max_value, anchor_value)


def mutate_wide_param(current: float, min_value: float, max_value: float, strength: float, anchor_value: float) -> float:
    min_value = max(1e-9, float(min_value))
    max_value = max(min_value, float(max_value))
    current = clip(float(current), min_value, max_value)
    anchor_value = clip(float(anchor_value), min_value, max_value)

    # Occasionally do a fresh jump anywhere in the full range.
    if random.random() < 0.18:
        return sample_wide_range(min_value, max_value, anchor_value)

    ratio = max_value / min_value if min_value > 0 else 1.0
    if ratio >= 20.0:
        lo = _safe_log10(min_value)
        hi = _safe_log10(max_value)
        cur = _safe_log10(current)
        step = max(0.05, strength * 0.35)
        return 10 ** clip(cur + random.gauss(0.0, step), lo, hi)

    step = max((max_value - min_value) * 0.01, strength * (max_value - min_value) * 0.12)
    return clip(current + random.gauss(0.0, step), min_value, max_value)


def clamp_exit_params(
    arm_pct: float,
    trail_drop_pct: float,
    stop_loss_pct: float,
    cfg: "Config",
) -> Tuple[float, float, float]:
    # Guarantee trail_drop_pct <= 0.5 * arm_pct while also respecting the
    # configured trail minimum.
    arm_floor = max(cfg.arm_pct_min, cfg.trail_drop_pct_min * 2.0)

    if cfg.arm_pct_max < arm_floor:
        raise ValueError(
            "Invalid config: ARM_PCT_MAX must be >= max(ARM_PCT_MIN, 2 * TRAIL_DROP_PCT_MIN)"
        )

    arm = clip(float(arm_pct), arm_floor, cfg.arm_pct_max)

    trail_cap = min(cfg.trail_drop_pct_max, arm * 0.5)
    trail = clip(float(trail_drop_pct), cfg.trail_drop_pct_min, trail_cap)

    stop = clip(float(stop_loss_pct), cfg.stop_loss_pct_min, cfg.stop_loss_pct_max)
    return arm, trail, stop


# ----------------------------
# Config
# ----------------------------

FEATURE_COLUMNS = [
    "chg_1d_pct",
    "chg_5d_pct",
    "chg_20d_pct",
    "dist_sma50_pct",
    "dist_sma200_pct",
    "trend_200",
    "atr_pct",
    "pos_20d",
    "pos_52w",
    "vol_z_20d",
    "sma50_slope_5d_pct",
    "sma200_slope_5d_pct",
    "daily_range_pct",
    "m5_range_pct",
    "spread_bps",
    "spread_vs_atr",
    "quote_age_s",
]

PRICE_COLUMNS = ["last_trade_price", "close", "bid", "ask"]


@dataclass
class Config:
    sheet_id: str
    screener_tab: str = "Screener"
    history_tab: str = "history"
    signals_tab: str = "Signals"
    leaderboard_tab: str = "StrategyLeaderboard"
    best_strategy_tab: str = "BestStrategy"
    open_trades_tab: str = "OpenTrades"
    closed_trades_tab: str = "ClosedTrades"

    signal_poll_seconds: int = 60
    retrain_minutes: int = 60

    population_size: int = 125
    generations: int = 40
    elite_count: int = 10
    immigrant_fraction: float = 0.20
    leaderboard_size: int = 100
    mutation_rate: float = 0.18
    mutation_strength: float = 0.45

    threshold_min: float = 20.0
    threshold_max: float = 75.0

    arm_pct_min: float = 0.15
    arm_pct_max: float = 0.80
    trail_drop_pct_min: float = 0.05
    trail_drop_pct_max: float = 0.40
    stop_loss_pct_min: float = 0.05
    stop_loss_pct_max: float = 0.40

    min_trades: int = 15
    validation_split: float = 0.25
    random_seed: int = 42

    min_closed_trades_for_timeout: int = 6
    min_live_win_rate_pct: float = 50.0
    strategy_timeout_hours: int = 24
    timeout_state_file: str = "/tmp/strategy_timeouts.json"

    state_file: str = "/tmp/best_strategy.json"
    log_level: str = "INFO"

    def __post_init__(self) -> None:
        if self.elite_count >= self.population_size:
            self.elite_count = max(1, self.population_size // 4)

        self.arm_pct_min = max(0.01, self.arm_pct_min)
        self.arm_pct_max = max(self.arm_pct_min, self.arm_pct_max)

        self.trail_drop_pct_min = max(0.01, self.trail_drop_pct_min)
        self.trail_drop_pct_max = max(self.trail_drop_pct_min, self.trail_drop_pct_max)

        self.stop_loss_pct_min = max(0.01, self.stop_loss_pct_min)
        self.stop_loss_pct_max = max(self.stop_loss_pct_min, self.stop_loss_pct_max)

        # Ensure every valid arm_pct can support at least the minimum trail_drop_pct
        self.arm_pct_min = max(self.arm_pct_min, self.trail_drop_pct_min * 2.0)

        if self.arm_pct_max < self.arm_pct_min:
            raise ValueError(
                "Invalid config: ARM_PCT_MAX must be >= max(ARM_PCT_MIN, 2 * TRAIL_DROP_PCT_MIN)"
            )

        self.immigrant_fraction = clip(self.immigrant_fraction, 0.0, 0.80)
        self.leaderboard_size = max(10, self.leaderboard_size)
        self.min_live_win_rate_pct = clip(self.min_live_win_rate_pct, 0.0, 100.0)
        self.min_closed_trades_for_timeout = max(1, self.min_closed_trades_for_timeout)
        self.strategy_timeout_hours = max(1, self.strategy_timeout_hours)


def load_config() -> Config:
    return Config(
        sheet_id=env_str("SHEET_ID"),
        screener_tab=os.getenv("SCREENER_TAB", "Screener"),
        history_tab=os.getenv("HISTORY_TAB", "history"),
        signals_tab=os.getenv("SIGNALS_TAB", "Signals"),
        leaderboard_tab=os.getenv("LEADERBOARD_TAB", "StrategyLeaderboard"),
        best_strategy_tab=os.getenv("BEST_STRATEGY_TAB", "BestStrategy"),
        open_trades_tab=os.getenv("OPEN_TRADES_TAB", "OpenTrades"),
        closed_trades_tab=os.getenv("CLOSED_TRADES_TAB", "ClosedTrades"),
        signal_poll_seconds=env_int("SIGNAL_POLL_SECONDS", 60),
        retrain_minutes=env_int("RETRAIN_MINUTES", 60),
        population_size=env_int("POPULATION_SIZE", 125),
        generations=env_int("GENERATIONS", 40),
        elite_count=env_int("ELITE_COUNT", 10),
        immigrant_fraction=env_float("IMMIGRANT_FRACTION", 0.20),
        leaderboard_size=env_int("LEADERBOARD_SIZE", 100),
        mutation_rate=env_float("MUTATION_RATE", 0.18),
        mutation_strength=env_float("MUTATION_STRENGTH", 0.45),
        threshold_min=env_float("THRESHOLD_MIN", 20.0),
        threshold_max=env_float("THRESHOLD_MAX", 75.0),
        arm_pct_min=env_float("ARM_PCT_MIN", 0.15),
        arm_pct_max=env_float("ARM_PCT_MAX", 0.80),
        trail_drop_pct_min=env_float("TRAIL_DROP_PCT_MIN", 0.05),
        trail_drop_pct_max=env_float("TRAIL_DROP_PCT_MAX", 0.40),
        stop_loss_pct_min=env_float("STOP_LOSS_PCT_MIN", 0.05),
        stop_loss_pct_max=env_float("STOP_LOSS_PCT_MAX", 0.40),
        min_trades=env_int("MIN_TRADES", 15),
        validation_split=env_float("VALIDATION_SPLIT", 0.25),
        random_seed=env_int("RANDOM_SEED", 42),
        min_closed_trades_for_timeout=env_int("MIN_CLOSED_TRADES_FOR_TIMEOUT", 6),
        min_live_win_rate_pct=env_float("MIN_LIVE_WIN_RATE_PCT", 50.0),
        strategy_timeout_hours=env_int("STRATEGY_TIMEOUT_HOURS", 24),
        timeout_state_file=os.getenv("TIMEOUT_STATE_FILE", "/tmp/strategy_timeouts.json"),
        state_file=os.getenv("STATE_FILE", "/tmp/best_strategy.json"),
        log_level=os.getenv("LOG_LEVEL", "INFO").upper(),
    )


# ----------------------------
# Google Sheets
# ----------------------------

def get_gspread_client() -> gspread.Client:
    b64 = os.getenv("GOOGLE_SERVICE_ACCOUNT_JSON_B64")
    if b64 and b64.strip():
        info = json.loads(base64.b64decode(b64).decode("utf-8"))
        return gspread.service_account_from_dict(info)

    raw = os.getenv("GOOGLE_SERVICE_ACCOUNT_JSON")
    if raw and raw.strip():
        info = json.loads(raw)
        return gspread.service_account_from_dict(info)

    path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    if path and path.strip():
        return gspread.service_account(filename=path)

    raise ValueError(
        "Missing Google credentials. Set GOOGLE_SERVICE_ACCOUNT_JSON_B64 or GOOGLE_SERVICE_ACCOUNT_JSON "
        "or GOOGLE_APPLICATION_CREDENTIALS."
    )


def get_or_create_ws(ss: gspread.Spreadsheet, title: str, rows: int = 2000, cols: int = 40) -> gspread.Worksheet:
    try:
        return ss.worksheet(title)
    except gspread.WorksheetNotFound:
        return ss.add_worksheet(title=title, rows=str(rows), cols=str(cols))


def write_table(ws: gspread.Worksheet, headers: List[str], rows: List[List[Any]]) -> None:
    ws.clear()
    values = [headers] + rows if rows else [headers]
    last_col = col_to_a1(len(headers))
    ws.update(range_name=f"A1:{last_col}{len(values)}", values=values)


def write_key_values(ws: gspread.Worksheet, pairs: List[Tuple[str, Any]]) -> None:
    ws.clear()
    rows = [["field", "value"]]
    for k, v in pairs:
        rows.append([k, json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list)) else str(v)])
    ws.update(range_name=f"A1:B{len(rows)}", values=rows)


# ----------------------------
# Data structures
# ----------------------------

@dataclass
class FeatureStats:
    mean: float
    stdev: float


@dataclass
class Sample:
    symbol: str
    asof_utc: str
    features: Dict[str, float]
    future_prices: List[float]
    spread_cost_pct: float
    current_price: float


@dataclass
class StrategyGenome:
    strategy_id: str
    weights: Dict[str, float]
    bias: float
    threshold: float
    arm_pct: float = 0.30
    trail_drop_pct: float = 0.15
    stop_loss_pct: float = 0.15


@dataclass
class StrategyMetrics:
    trades: int = 0
    wins: int = 0
    losses: int = 0
    win_rate: float = 0.0
    total_pnl_pct: float = 0.0
    avg_pnl_pct: float = 0.0
    avg_win_pct: float = 0.0
    avg_loss_pct: float = 0.0
    profit_factor: float = 0.0
    long_trades: int = 0
    short_trades: int = 0
    tp_exits: int = 0
    stop_exits: int = 0
    open_at_end_exits: int = 0
    armed_trades: int = 0
    tp_rate: float = 0.0
    stop_rate: float = 0.0
    open_at_end_rate: float = 0.0
    armed_rate: float = 0.0
    validation_trades: int = 0
    validation_win_rate: float = 0.0
    validation_pnl_pct: float = 0.0
    validation_tp_rate: float = 0.0
    validation_stop_rate: float = 0.0
    validation_open_at_end_rate: float = 0.0
    fitness: float = -1e9


@dataclass
class LeaderboardEntry:
    rank: int
    genome: StrategyGenome
    metrics: StrategyMetrics


@dataclass
class BestStrategyBundle:
    genome: StrategyGenome
    metrics: StrategyMetrics
    feature_stats: Dict[str, FeatureStats]
    trained_at: str
    leaderboard: List[LeaderboardEntry] = field(default_factory=list)


@dataclass
class ExitResult:
    pnl_pct: float
    peak_pnl_pct: float
    armed: bool
    reason: str
    bars_held: int


@dataclass
class TradeStats:
    strategy_id: str
    open_trades: int = 0
    closed_trades: int = 0
    closed_wins: int = 0
    closed_losses: int = 0
    closed_win_rate: float = 0.0


# ----------------------------
# Sheet readers
# ----------------------------

def normalize_row(row: List[Any], width: int) -> List[str]:
    cells = ["" if v is None else str(v) for v in row]
    if len(cells) < width:
        cells.extend([""] * (width - len(cells)))
    return cells[:width]


def read_screener_and_history(gc: gspread.Client, cfg: Config) -> Tuple[List[str], List[Dict[str, str]], List[List[Dict[str, str]]]]:
    ss = gc.open_by_key(cfg.sheet_id)

    ws_screener = ss.worksheet(cfg.screener_tab)
    screener_values = ws_screener.get_all_values()
    if not screener_values:
        raise RuntimeError(f"{cfg.screener_tab} is empty")

    headers = normalize_row(screener_values[0], len(screener_values[0]))
    screener_rows: List[Dict[str, str]] = []
    for row in screener_values[1:]:
        norm = normalize_row(row, len(headers))
        if not any(cell.strip() for cell in norm):
            continue
        screener_rows.append(dict(zip(headers, norm)))

    ws_history = ss.worksheet(cfg.history_tab)
    history_values = ws_history.get_all_values()

    blocks: List[List[Dict[str, str]]] = []
    current_block: List[Dict[str, str]] = []
    for row in history_values:
        norm = normalize_row(row, len(headers))
        if not any(cell.strip() for cell in norm):
            if current_block:
                blocks.append(current_block)
                current_block = []
            continue
        current_block.append(dict(zip(headers, norm)))
    if current_block:
        blocks.append(current_block)

    return headers, screener_rows, blocks


def read_sheet_rows(ss: gspread.Spreadsheet, title: str) -> List[Dict[str, str]]:
    try:
        ws = ss.worksheet(title)
    except gspread.WorksheetNotFound:
        return []

    values = ws.get_all_values()
    if not values:
        return []

    headers = normalize_row(values[0], len(values[0]))
    rows: List[Dict[str, str]] = []
    for row in values[1:]:
        norm = normalize_row(row, len(headers))
        if not any(cell.strip() for cell in norm):
            continue
        rows.append(dict(zip(headers, norm)))
    return rows


def read_trade_state(gc: gspread.Client, cfg: Config) -> Tuple[List[Dict[str, str]], List[Dict[str, str]]]:
    ss = gc.open_by_key(cfg.sheet_id)
    open_rows = read_sheet_rows(ss, cfg.open_trades_tab)
    closed_rows = read_sheet_rows(ss, cfg.closed_trades_tab)
    return open_rows, closed_rows


def aggregate_trade_stats(
    open_rows: List[Dict[str, str]],
    closed_rows: List[Dict[str, str]],
) -> Dict[str, TradeStats]:
    stats: Dict[str, TradeStats] = {}

    def get_stat(strategy_id: str) -> TradeStats:
        return stats.setdefault(strategy_id, TradeStats(strategy_id=strategy_id))

    for row in open_rows:
        strategy_id = str(row.get("StrategyID", "")).strip()
        if not strategy_id:
            continue
        get_stat(strategy_id).open_trades += 1

    for row in closed_rows:
        strategy_id = str(row.get("StrategyID", "")).strip()
        if not strategy_id:
            continue

        stat = get_stat(strategy_id)
        stat.closed_trades += 1

        realized_pl = safe_float(row.get("RealizedPL"), None)
        price_pct = safe_float(row.get("PricePct"), None)

        is_win = False
        if realized_pl is not None:
            is_win = realized_pl > 0
        elif price_pct is not None:
            is_win = price_pct > 0

        if is_win:
            stat.closed_wins += 1
        else:
            stat.closed_losses += 1

    for stat in stats.values():
        if stat.closed_trades > 0:
            stat.closed_win_rate = stat.closed_wins / stat.closed_trades * 100.0

    return stats


def load_timeout_state(path: str) -> Dict[str, Any]:
    p = Path(path)
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}


def save_timeout_state(path: str, payload: Dict[str, Any]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")


def refresh_timeout_state(
    current_state: Dict[str, Any],
    trade_stats: Dict[str, TradeStats],
    cfg: Config,
    now: Optional[datetime] = None,
) -> Dict[str, Any]:
    now = now or datetime.now(timezone.utc)
    refreshed: Dict[str, Any] = {}

    for strategy_id, info in current_state.items():
        try:
            until = datetime.fromisoformat(str(info.get("until", "")).replace("Z", "+00:00"))
        except Exception:
            continue
        if until > now:
            refreshed[strategy_id] = info

    for strategy_id, stat in trade_stats.items():
        if stat.closed_trades < cfg.min_closed_trades_for_timeout:
            continue
        if stat.closed_win_rate >= cfg.min_live_win_rate_pct:
            continue

        refreshed[strategy_id] = {
            "until": (now + timedelta(hours=cfg.strategy_timeout_hours)).isoformat(timespec="seconds"),
            "closed_trades": stat.closed_trades,
            "closed_win_rate": round(stat.closed_win_rate, 2),
            "open_trades": stat.open_trades,
            "triggered_at": now.isoformat(timespec="seconds"),
        }

    return refreshed


def is_strategy_timed_out(
    strategy_id: str,
    timeout_state: Dict[str, Any],
    now: Optional[datetime] = None,
) -> bool:
    now = now or datetime.now(timezone.utc)
    info = timeout_state.get(strategy_id)
    if not info:
        return False
    try:
        until = datetime.fromisoformat(str(info.get("until", "")).replace("Z", "+00:00"))
    except Exception:
        return False
    return until > now


# ----------------------------
# History prep
# ----------------------------

def pick_price(row: Dict[str, str]) -> Optional[float]:
    for key in PRICE_COLUMNS:
        val = safe_float(row.get(key), None)
        if val is not None and val > 0:
            return val
    return None


def compute_feature_stats(blocks: List[List[Dict[str, str]]]) -> Dict[str, FeatureStats]:
    values: Dict[str, List[float]] = {f: [] for f in FEATURE_COLUMNS}
    for block in blocks:
        for row in block:
            for feature in FEATURE_COLUMNS:
                val = safe_float(row.get(feature), None)
                if val is not None and math.isfinite(val):
                    values[feature].append(val)

    stats: Dict[str, FeatureStats] = {}
    for feature, vals in values.items():
        if len(vals) >= 2:
            sd = statistics.pstdev(vals)
            stats[feature] = FeatureStats(mean=statistics.mean(vals), stdev=sd if sd > 1e-9 else 1.0)
        elif len(vals) == 1:
            stats[feature] = FeatureStats(mean=vals[0], stdev=1.0)
        else:
            stats[feature] = FeatureStats(mean=0.0, stdev=1.0)
    return stats


def normalize_features(row: Dict[str, str], feature_stats: Dict[str, FeatureStats]) -> Dict[str, float]:
    normalized: Dict[str, float] = {}
    for feature in FEATURE_COLUMNS:
        stats = feature_stats[feature]
        raw = safe_float(row.get(feature), None)
        if raw is None or not math.isfinite(raw):
            normalized[feature] = 0.0
            continue
        z = (raw - stats.mean) / stats.stdev
        normalized[feature] = clip(z, -3.0, 3.0)
    return normalized


def build_samples(blocks: List[List[Dict[str, str]]], feature_stats: Dict[str, FeatureStats]) -> List[Sample]:
    series: Dict[str, List[Dict[str, str]]] = {}
    for block in blocks:
        for row in block:
            symbol = str(row.get("symbol", "")).strip()
            asof = str(row.get("asof_utc", "")).strip()
            if not symbol or not asof:
                continue
            series.setdefault(symbol, []).append(row)

    for symbol in list(series.keys()):
        series[symbol].sort(key=lambda r: r.get("asof_utc", ""))

    samples: List[Sample] = []
    for symbol, rows in series.items():
        if len(rows) <= 2:
            continue
        for i in range(len(rows) - 1):
            current = rows[i]
            entry_price = pick_price(current)
            if entry_price is None or entry_price <= 0:
                continue

            future_prices: List[float] = []
            for future_row in rows[i + 1 :]:
                future_price = pick_price(future_row)
                if future_price is None or future_price <= 0:
                    continue
                future_prices.append(future_price)

            if not future_prices:
                continue

            spread_pct = safe_float(current.get("spread_pct"), 0.0) or 0.0
            spread_cost_pct = max(0.0, spread_pct) * 2.0
            samples.append(
                Sample(
                    symbol=symbol,
                    asof_utc=str(current.get("asof_utc", "")),
                    features=normalize_features(current, feature_stats),
                    future_prices=future_prices,
                    spread_cost_pct=spread_cost_pct,
                    current_price=entry_price,
                )
            )

    samples.sort(key=lambda s: s.asof_utc)
    return samples


# ----------------------------
# Strategy scoring / evaluation
# ----------------------------

def score_strategy(genome: StrategyGenome, features: Dict[str, float]) -> float:
    raw = genome.bias
    for feature in FEATURE_COLUMNS:
        raw += genome.weights.get(feature, 0.0) * features.get(feature, 0.0)
    raw /= math.sqrt(max(1, len(FEATURE_COLUMNS)))
    return clip(math.tanh(raw) * 100.0, -100.0, 100.0)


def simulate_exit(sample: Sample, genome: StrategyGenome, direction: float) -> ExitResult:
    peak_pnl = -1e9
    armed = False
    last_pnl = 0.0

    for step, future_price in enumerate(sample.future_prices, start=1):
        pnl_pct = direction * (((future_price / sample.current_price) - 1.0) * 100.0)
        last_pnl = pnl_pct
        peak_pnl = max(peak_pnl, pnl_pct)

        if peak_pnl >= genome.arm_pct:
            armed = True

        # Conservative ordering on snapshot data: if a sampled price is already through stop,
        # treat it as a stop-loss exit.
        if pnl_pct <= -genome.stop_loss_pct:
            return ExitResult(
                pnl_pct=pnl_pct,
                peak_pnl_pct=max(peak_pnl, pnl_pct),
                armed=armed,
                reason="STOP_LOSS",
                bars_held=step,
            )

        if armed and (peak_pnl - pnl_pct) >= genome.trail_drop_pct:
            return ExitResult(
                pnl_pct=pnl_pct,
                peak_pnl_pct=peak_pnl,
                armed=armed,
                reason="TRAIL_TP",
                bars_held=step,
            )

    if peak_pnl == -1e9:
        peak_pnl = last_pnl

    return ExitResult(
        pnl_pct=last_pnl,
        peak_pnl_pct=peak_pnl,
        armed=armed,
        reason="OPEN_AT_END",
        bars_held=len(sample.future_prices),
    )


def simulate(genome: StrategyGenome, samples: Iterable[Sample]) -> StrategyMetrics:
    metrics = StrategyMetrics()
    gross_profit = 0.0
    gross_loss = 0.0

    for sample in samples:
        score = score_strategy(genome, sample.features)
        if abs(score) < genome.threshold:
            continue

        metrics.trades += 1
        direction = 1.0 if score > 0 else -1.0
        if direction > 0:
            metrics.long_trades += 1
        else:
            metrics.short_trades += 1

        exit_result = simulate_exit(sample, genome, direction)
        pnl_pct = exit_result.pnl_pct - sample.spread_cost_pct
        metrics.total_pnl_pct += pnl_pct

        if exit_result.armed:
            metrics.armed_trades += 1

        if exit_result.reason == "TRAIL_TP" and pnl_pct > 0:
            metrics.tp_exits += 1
        elif exit_result.reason == "STOP_LOSS":
            metrics.stop_exits += 1
        else:
            metrics.open_at_end_exits += 1

        if pnl_pct > 0:
            metrics.wins += 1
            gross_profit += pnl_pct
        else:
            metrics.losses += 1
            gross_loss += abs(pnl_pct)

    if metrics.trades > 0:
        metrics.win_rate = metrics.wins / metrics.trades * 100.0
        metrics.avg_pnl_pct = metrics.total_pnl_pct / metrics.trades
        metrics.tp_rate = metrics.tp_exits / metrics.trades * 100.0
        metrics.stop_rate = metrics.stop_exits / metrics.trades * 100.0
        metrics.open_at_end_rate = metrics.open_at_end_exits / metrics.trades * 100.0
        metrics.armed_rate = metrics.armed_trades / metrics.trades * 100.0
    if metrics.wins > 0:
        metrics.avg_win_pct = gross_profit / metrics.wins
    if metrics.losses > 0:
        metrics.avg_loss_pct = gross_loss / metrics.losses
    if gross_loss > 0:
        metrics.profit_factor = gross_profit / gross_loss
    elif gross_profit > 0:
        metrics.profit_factor = gross_profit

    return metrics


def combine_fitness(train_metrics: StrategyMetrics, val_metrics: StrategyMetrics, min_trades: int) -> float:
    train_trade_factor = min(1.0, train_metrics.trades / max(1, min_trades))
    val_trade_factor = min(1.0, val_metrics.trades / max(1, max(5, min_trades // 2)))

    train_realized_rate = max(0.0, 100.0 - train_metrics.open_at_end_rate)
    val_realized_rate = max(0.0, 100.0 - val_metrics.open_at_end_rate)

    train_score = (
        train_metrics.total_pnl_pct * 1.0
        + train_metrics.win_rate * 0.45
        + train_metrics.avg_pnl_pct * 32.0
        + min(train_metrics.profit_factor, 5.0) * 3.5
        + train_metrics.tp_rate * 0.90
        + train_realized_rate * 0.70
        + train_metrics.armed_rate * 0.08
        - train_metrics.stop_rate * 1.10
        - train_metrics.open_at_end_rate * 1.80
    ) * train_trade_factor

    val_score = (
        val_metrics.total_pnl_pct * 1.25
        + val_metrics.win_rate * 0.55
        + val_metrics.avg_pnl_pct * 40.0
        + min(val_metrics.profit_factor, 5.0) * 4.5
        + val_metrics.tp_rate * 1.20
        + val_realized_rate * 1.00
        + val_metrics.armed_rate * 0.08
        - val_metrics.stop_rate * 1.50
        - val_metrics.open_at_end_rate * 2.60
    ) * val_trade_factor

    shortage_penalty = max(0, min_trades - train_metrics.trades) * 2.0
    negative_val_penalty = abs(min(0.0, val_metrics.total_pnl_pct)) * 8.0

    open_end_penalty = 0.0
    if train_metrics.open_at_end_rate > 50.0:
        open_end_penalty += (train_metrics.open_at_end_rate - 50.0) * 2.5
    if val_metrics.open_at_end_rate > 40.0:
        open_end_penalty += (val_metrics.open_at_end_rate - 40.0) * 4.0

    if train_metrics.open_at_end_rate >= 80.0:
        open_end_penalty += 150.0
    if val_metrics.open_at_end_rate >= 70.0:
        open_end_penalty += 250.0

    if train_metrics.open_at_end_rate >= 95.0:
        open_end_penalty += 250.0
    if val_metrics.open_at_end_rate >= 90.0:
        open_end_penalty += 400.0

    if train_metrics.trades > 0 and train_metrics.open_at_end_rate >= 99.9:
        open_end_penalty += 500.0
    if val_metrics.trades > 0 and val_metrics.open_at_end_rate >= 99.9:
        open_end_penalty += 800.0

    return train_score * 0.45 + val_score * 0.55 - shortage_penalty - negative_val_penalty - open_end_penalty


def random_genome(cfg: Config) -> StrategyGenome:
    arm_pct = sample_wide_range(
        cfg.arm_pct_min,
        cfg.arm_pct_max,
        max(0.30, cfg.trail_drop_pct_min * 2.0),
        tail_prob=0.15,
    )

    trail_cap = min(cfg.trail_drop_pct_max, arm_pct * 0.5)
    trail_drop_pct = sample_wide_range(
        cfg.trail_drop_pct_min,
        trail_cap,
        min(0.15, trail_cap),
        tail_prob=0.15,
    )

    stop_loss_pct = sample_wide_range(
        cfg.stop_loss_pct_min,
        cfg.stop_loss_pct_max,
        0.15,
        tail_prob=0.15,
    )

    arm_pct, trail_drop_pct, stop_loss_pct = clamp_exit_params(
        arm_pct, trail_drop_pct, stop_loss_pct, cfg
    )

    return StrategyGenome(
        strategy_id=str(uuid.uuid4())[:8],
        weights={feature: random.uniform(-2.0, 2.0) for feature in FEATURE_COLUMNS},
        bias=random.uniform(-1.5, 1.5),
        threshold=sample_wide_range(cfg.threshold_min, cfg.threshold_max, 50.0, tail_prob=0.12),
        arm_pct=arm_pct,
        trail_drop_pct=trail_drop_pct,
        stop_loss_pct=stop_loss_pct,
    )


def crossover(a: StrategyGenome, b: StrategyGenome, cfg: Config) -> StrategyGenome:
    child_weights: Dict[str, float] = {}
    for feature in FEATURE_COLUMNS:
        if random.random() < 0.5:
            child_weights[feature] = a.weights[feature]
        else:
            child_weights[feature] = b.weights[feature]
        if random.random() < 0.25:
            child_weights[feature] = (a.weights[feature] + b.weights[feature]) / 2.0

    arm_pct = random.choice([a.arm_pct, b.arm_pct, (a.arm_pct + b.arm_pct) / 2.0])
    trail_drop_pct = random.choice([
        a.trail_drop_pct,
        b.trail_drop_pct,
        (a.trail_drop_pct + b.trail_drop_pct) / 2.0,
    ])
    stop_loss_pct = random.choice([
        a.stop_loss_pct,
        b.stop_loss_pct,
        (a.stop_loss_pct + b.stop_loss_pct) / 2.0,
    ])

    arm_pct, trail_drop_pct, stop_loss_pct = clamp_exit_params(
        arm_pct, trail_drop_pct, stop_loss_pct, cfg
    )

    return StrategyGenome(
        strategy_id=str(uuid.uuid4())[:8],
        weights=child_weights,
        bias=(a.bias + b.bias) / 2.0 if random.random() < 0.5 else random.choice([a.bias, b.bias]),
        threshold=random.choice([a.threshold, b.threshold, (a.threshold + b.threshold) / 2.0]),
        arm_pct=arm_pct,
        trail_drop_pct=trail_drop_pct,
        stop_loss_pct=stop_loss_pct,
    )


def mutate(genome: StrategyGenome, cfg: Config) -> StrategyGenome:
    child = StrategyGenome(
        strategy_id=str(uuid.uuid4())[:8],
        weights=dict(genome.weights),
        bias=genome.bias,
        threshold=genome.threshold,
        arm_pct=genome.arm_pct,
        trail_drop_pct=genome.trail_drop_pct,
        stop_loss_pct=genome.stop_loss_pct,
    )

    for feature in FEATURE_COLUMNS:
        if random.random() < cfg.mutation_rate:
            child.weights[feature] += random.gauss(0.0, cfg.mutation_strength)
            child.weights[feature] = clip(child.weights[feature], -4.0, 4.0)

    if random.random() < cfg.mutation_rate:
        child.bias = clip(child.bias + random.gauss(0.0, cfg.mutation_strength), -4.0, 4.0)

    if random.random() < cfg.mutation_rate:
        child.threshold = mutate_wide_param(
            child.threshold,
            cfg.threshold_min,
            cfg.threshold_max,
            cfg.mutation_strength,
            50.0,
        )

    if random.random() < cfg.mutation_rate:
        child.arm_pct = mutate_wide_param(
            child.arm_pct,
            cfg.arm_pct_min,
            cfg.arm_pct_max,
            cfg.mutation_strength,
            max(0.30, cfg.trail_drop_pct_min * 2.0),
        )

    if random.random() < cfg.mutation_rate:
        trail_cap = min(cfg.trail_drop_pct_max, child.arm_pct * 0.5)
        child.trail_drop_pct = mutate_wide_param(
            child.trail_drop_pct,
            cfg.trail_drop_pct_min,
            trail_cap,
            cfg.mutation_strength,
            min(0.15, trail_cap),
        )

    if random.random() < cfg.mutation_rate:
        child.stop_loss_pct = mutate_wide_param(
            child.stop_loss_pct,
            cfg.stop_loss_pct_min,
            cfg.stop_loss_pct_max,
            cfg.mutation_strength,
            0.15,
        )

    child.arm_pct, child.trail_drop_pct, child.stop_loss_pct = clamp_exit_params(
        child.arm_pct,
        child.trail_drop_pct,
        child.stop_loss_pct,
        cfg,
    )

    return child


def clone_genome(genome: StrategyGenome, preserve_id: bool = True) -> StrategyGenome:
    return StrategyGenome(
        strategy_id=genome.strategy_id if preserve_id else str(uuid.uuid4())[:8],
        weights=dict(genome.weights),
        bias=genome.bias,
        threshold=genome.threshold,
        arm_pct=genome.arm_pct,
        trail_drop_pct=genome.trail_drop_pct,
        stop_loss_pct=genome.stop_loss_pct,
    )


def genome_signature(genome: StrategyGenome, decimals: int = 6) -> str:
    payload = {
        "bias": round(genome.bias, decimals),
        "threshold": round(genome.threshold, decimals),
        "arm_pct": round(genome.arm_pct, decimals),
        "trail_drop_pct": round(genome.trail_drop_pct, decimals),
        "stop_loss_pct": round(genome.stop_loss_pct, decimals),
        "weights": {
            feature: round(genome.weights.get(feature, 0.0), decimals)
            for feature in FEATURE_COLUMNS
        },
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def dedupe_population(population: List[StrategyGenome]) -> List[StrategyGenome]:
    seen: set[str] = set()
    unique: List[StrategyGenome] = []
    for genome in population:
        sig = genome_signature(genome)
        if sig in seen:
            continue
        seen.add(sig)
        unique.append(genome)
    return unique


def dedupe_scored(
    scored: List[Tuple[StrategyGenome, StrategyMetrics]]
) -> List[Tuple[StrategyGenome, StrategyMetrics]]:
    best_by_sig: Dict[str, Tuple[StrategyGenome, StrategyMetrics]] = {}
    for genome, metrics in scored:
        sig = genome_signature(genome)
        current = best_by_sig.get(sig)
        if current is None or metrics.fitness > current[1].fitness:
            best_by_sig[sig] = (genome, metrics)

    unique = list(best_by_sig.values())
    unique.sort(key=lambda item: item[1].fitness, reverse=True)
    return unique


def score_genome(
    genome: StrategyGenome,
    train_samples: List[Sample],
    val_samples: List[Sample],
    min_trades: int,
) -> StrategyMetrics:
    train_metrics = simulate(genome, train_samples)
    val_metrics = simulate(genome, val_samples)
    train_metrics.validation_trades = val_metrics.trades
    train_metrics.validation_win_rate = val_metrics.win_rate
    train_metrics.validation_pnl_pct = val_metrics.total_pnl_pct
    train_metrics.validation_tp_rate = val_metrics.tp_rate
    train_metrics.validation_stop_rate = val_metrics.stop_rate
    train_metrics.validation_open_at_end_rate = val_metrics.open_at_end_rate
    train_metrics.fitness = combine_fitness(train_metrics, val_metrics, min_trades)
    return train_metrics


def tournament_select(scored: List[Tuple[StrategyGenome, StrategyMetrics]], k: int = 4) -> StrategyGenome:
    pool = random.sample(scored, k=min(k, len(scored)))
    pool.sort(key=lambda item: item[1].fitness, reverse=True)
    return pool[0][0]


def evolve_strategies(
    samples: List[Sample],
    feature_stats: Dict[str, FeatureStats],
    cfg: Config,
    log: logging.Logger,
    incumbent_bundle: Optional[BestStrategyBundle] = None,
    timed_out_strategy_ids: Optional[set[str]] = None,
) -> BestStrategyBundle:
    timed_out_strategy_ids = timed_out_strategy_ids or set()

    if len(samples) < max(20, cfg.min_trades + 5):
        raise RuntimeError("Not enough history to evolve strategies yet.")

    split_idx = max(1, int(len(samples) * (1.0 - cfg.validation_split)))
    train_samples = samples[:split_idx]
    val_samples = samples[split_idx:] if split_idx < len(samples) else samples[-max(5, len(samples)//4):]

    population: List[StrategyGenome] = []

    if incumbent_bundle is not None and incumbent_bundle.genome.strategy_id not in timed_out_strategy_ids:
        # Persist the current champion into the next retrain.
        population.append(clone_genome(incumbent_bundle.genome, preserve_id=True))

        # Seed with prior top genomes so separate cycles build on prior progress.
        prior_elites = incumbent_bundle.leaderboard or [
            LeaderboardEntry(rank=1, genome=incumbent_bundle.genome, metrics=incumbent_bundle.metrics)
        ]

        max_seeded = min(max(cfg.elite_count * 3, 12), max(1, cfg.population_size // 3))
        for entry in prior_elites[:max_seeded]:
            if entry.genome.strategy_id in timed_out_strategy_ids:
                continue
            population.append(clone_genome(entry.genome, preserve_id=False))

        # Add mutated descendants near the current high score to keep searching around it.
        seed_sources = [g for g in population[: min(len(population), max(cfg.elite_count, 6))]]
        max_descendants = min(max(cfg.elite_count * 2, 12), max(1, cfg.population_size // 3))
        descendants_added = 0
        for seed in seed_sources:
            if descendants_added >= max_descendants:
                break
            if len(population) >= cfg.population_size:
                break
            population.append(mutate(clone_genome(seed, preserve_id=False), cfg))
            descendants_added += 1

    population = dedupe_population(population)
    population_signatures = {genome_signature(genome) for genome in population}

    while len(population) < cfg.population_size:
        candidate = random_genome(cfg)
        sig = genome_signature(candidate)
        if sig in population_signatures:
            continue
        population.append(candidate)
        population_signatures.add(sig)

    population = population[: cfg.population_size]

    top_scored: List[Tuple[StrategyGenome, StrategyMetrics]] = []
    immigrant_count = max(1, int(round(cfg.population_size * cfg.immigrant_fraction))) if cfg.population_size > 2 else 0

    for generation in range(1, cfg.generations + 1):
        scored: List[Tuple[StrategyGenome, StrategyMetrics]] = []
        for genome in population:
            if genome.strategy_id in timed_out_strategy_ids:
                continue
            metrics = score_genome(genome, train_samples, val_samples, cfg.min_trades)
            scored.append((genome, metrics))

        scored = dedupe_scored(scored)
        if not scored:
            raise RuntimeError("All candidate strategies were filtered out.")

        best_genome, best_metrics = scored[0]
        log.info(
            "generation=%d best_fitness=%.2f trades=%d win_rate=%.1f total_pnl=%.3f threshold=%.1f arm=%.3f trail=%.3f stop=%.3f open_end_rate=%.1f unique_candidates=%d",
            generation,
            best_metrics.fitness,
            best_metrics.trades,
            best_metrics.win_rate,
            best_metrics.total_pnl_pct,
            best_genome.threshold,
            best_genome.arm_pct,
            best_genome.trail_drop_pct,
            best_genome.stop_loss_pct,
            best_metrics.open_at_end_rate,
            len(scored),
        )
        top_scored = scored[: max(cfg.leaderboard_size, cfg.elite_count)]

        elites = [clone_genome(g, preserve_id=False) for g, _ in scored[: cfg.elite_count]]
        next_population: List[StrategyGenome] = dedupe_population(elites)
        next_population_signatures = {genome_signature(genome) for genome in next_population}

        breed_target = max(cfg.elite_count, cfg.population_size - immigrant_count)
        while len(next_population) < breed_target:
            parent_a = tournament_select(scored)
            parent_b = tournament_select(scored)
            child = mutate(crossover(parent_a, parent_b, cfg), cfg)
            sig = genome_signature(child)
            if sig in next_population_signatures:
                continue
            next_population.append(child)
            next_population_signatures.add(sig)

        while len(next_population) < cfg.population_size:
            candidate = random_genome(cfg)
            sig = genome_signature(candidate)
            if sig in next_population_signatures:
                continue
            next_population.append(candidate)
            next_population_signatures.add(sig)

        population = next_population[: cfg.population_size]

    top_scored = dedupe_scored(top_scored)
    best_genome, best_metrics = top_scored[0]

    leaderboard: List[LeaderboardEntry] = []
    for rank, (genome, metrics) in enumerate(top_scored[: cfg.leaderboard_size], start=1):
        leaderboard.append(LeaderboardEntry(rank=rank, genome=genome, metrics=metrics))

    bundle = BestStrategyBundle(
        genome=best_genome,
        metrics=best_metrics,
        feature_stats=feature_stats,
        trained_at=utc_now_iso(),
        leaderboard=leaderboard,
    )
    return bundle

# ----------------------------
# Persistence
# ----------------------------

def save_bundle(bundle: BestStrategyBundle, path: str) -> None:
    serializable = {
        "genome": asdict(bundle.genome),
        "metrics": asdict(bundle.metrics),
        "feature_stats": {k: asdict(v) for k, v in bundle.feature_stats.items()},
        "trained_at": bundle.trained_at,
        "leaderboard": [
            {
                "rank": entry.rank,
                "genome": asdict(entry.genome),
                "metrics": asdict(entry.metrics),
            }
            for entry in bundle.leaderboard
        ],
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(serializable, indent=2), encoding="utf-8")


def _genome_from_dict(data: Dict[str, Any]) -> StrategyGenome:
    clean = dict(data)
    clean.pop("hold_steps", None)

    arm_pct = float(clean.get("arm_pct", 0.30))
    trail_drop_pct = float(clean.get("trail_drop_pct", 0.15))
    stop_loss_pct = float(clean.get("stop_loss_pct", 0.15))

    # Clamp old saved genomes so they cannot keep invalid trail values.
    trail_drop_pct = min(trail_drop_pct, arm_pct * 0.5)

    return StrategyGenome(
        strategy_id=str(clean.get("strategy_id") or str(uuid.uuid4())[:8]),
        weights=dict(clean.get("weights") or {}),
        bias=float(clean.get("bias", 0.0)),
        threshold=float(clean.get("threshold", 50.0)),
        arm_pct=arm_pct,
        trail_drop_pct=trail_drop_pct,
        stop_loss_pct=stop_loss_pct,
    )


def _metrics_from_dict(data: Dict[str, Any]) -> StrategyMetrics:
    clean = dict(data)
    if "timeout_exits" in clean and "open_at_end_exits" not in clean:
        clean["open_at_end_exits"] = clean.pop("timeout_exits")
    else:
        clean.pop("timeout_exits", None)
    return StrategyMetrics(**clean)


def load_bundle(path: str) -> Optional[BestStrategyBundle]:
    p = Path(path)
    if not p.exists():
        return None
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        return BestStrategyBundle(
            genome=_genome_from_dict(data["genome"]),
            metrics=_metrics_from_dict(data["metrics"]),
            feature_stats={k: FeatureStats(**v) for k, v in data["feature_stats"].items()},
            trained_at=data["trained_at"],
            leaderboard=[
                LeaderboardEntry(
                    rank=entry["rank"],
                    genome=_genome_from_dict(entry["genome"]),
                    metrics=_metrics_from_dict(entry["metrics"]),
                )
                for entry in data.get("leaderboard", [])
            ],
        )
    except Exception:
        return None


# ----------------------------
# Live scoring
# ----------------------------

def build_signal_rows(bundle: BestStrategyBundle, screener_rows: List[Dict[str, str]]) -> List[List[Any]]:
    rows: List[List[Any]] = []
    for row in screener_rows:
        symbol = str(row.get("symbol", "")).strip()
        if not symbol:
            continue

        features = normalize_features(row, bundle.feature_stats)
        score = score_strategy(bundle.genome, features)
        side = "FLAT"
        entry_signal = "FALSE"
        if score >= bundle.genome.threshold:
            side = "LONG"
            entry_signal = "TRUE"
        elif score <= -bundle.genome.threshold:
            side = "SHORT"
            entry_signal = "TRUE"

        contributions = []
        for feature in FEATURE_COLUMNS:
            contrib = bundle.genome.weights.get(feature, 0.0) * features.get(feature, 0.0)
            contributions.append((abs(contrib), feature, contrib))
        contributions.sort(reverse=True)
        top_drivers = ", ".join(f"{name}:{contrib:+.2f}" for _, name, contrib in contributions[:3])

        rows.append([
            row.get("asof_utc", utc_now_iso()),
            symbol,
            round(score, 2),
            side,
            entry_signal,
            round(bundle.genome.threshold, 2),
            round(bundle.genome.arm_pct, 4),
            round(bundle.genome.trail_drop_pct, 4),
            round(bundle.genome.stop_loss_pct, 4),
            row.get("last_trade_price", ""),
            row.get("bid", ""),
            row.get("ask", ""),
            row.get("spread_bps", ""),
            row.get("atr_pct", ""),
            row.get("pos_20d", ""),
            row.get("pos_52w", ""),
            row.get("trend_200", ""),
            row.get("chg_1d_pct", ""),
            row.get("chg_5d_pct", ""),
            row.get("dist_sma50_pct", ""),
            row.get("dist_sma200_pct", ""),
            top_drivers,
            bundle.genome.strategy_id,
            bundle.trained_at,
        ])

    rows.sort(key=lambda r: abs(float(r[2])), reverse=True)
    return rows


def build_flat_signal_rows(
    screener_rows: List[Dict[str, str]],
    strategy_id: str = "TIMEOUT",
    trained_at: str = "",
) -> List[List[Any]]:
    rows: List[List[Any]] = []
    for row in screener_rows:
        symbol = str(row.get("symbol", "")).strip()
        if not symbol:
            continue

        rows.append([
            row.get("asof_utc", utc_now_iso()),
            symbol,
            0.0,
            "FLAT",
            "FALSE",
            "",
            "",
            "",
            "",
            row.get("last_trade_price", ""),
            row.get("bid", ""),
            row.get("ask", ""),
            row.get("spread_bps", ""),
            row.get("atr_pct", ""),
            row.get("pos_20d", ""),
            row.get("pos_52w", ""),
            row.get("trend_200", ""),
            row.get("chg_1d_pct", ""),
            row.get("chg_5d_pct", ""),
            row.get("dist_sma50_pct", ""),
            row.get("dist_sma200_pct", ""),
            "timed_out",
            strategy_id,
            trained_at,
        ])

    return rows


SIGNAL_HEADERS = [
    "asof_utc",
    "symbol",
    "score",
    "side",
    "entry_signal",
    "threshold",
    "arm_pct",
    "trail_drop_pct",
    "stop_loss_pct",
    "last_trade_price",
    "bid",
    "ask",
    "spread_bps",
    "atr_pct",
    "pos_20d",
    "pos_52w",
    "trend_200",
    "chg_1d_pct",
    "chg_5d_pct",
    "dist_sma50_pct",
    "dist_sma200_pct",
    "top_drivers",
    "strategy_id",
    "trained_at",
]


# ----------------------------
# Service
# ----------------------------

class StrategyService:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.log = logging.getLogger("strategy-service")
        self.gc = get_gspread_client()
        self.best_bundle: Optional[BestStrategyBundle] = load_bundle(cfg.state_file)
        self.last_train_at: Optional[str] = self.best_bundle.trained_at if self.best_bundle else None
        self.last_signal_at: Optional[str] = None
        self.last_error: Optional[str] = None
        self.last_generation_summary: Optional[Dict[str, Any]] = None
        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        self.thread: Optional[threading.Thread] = None

    def _sheet(self) -> gspread.Spreadsheet:
        return self.gc.open_by_key(self.cfg.sheet_id)

    def train_once(self) -> Dict[str, Any]:
        headers, screener_rows, history_blocks = read_screener_and_history(self.gc, self.cfg)
        if not headers:
            raise RuntimeError("Missing screener headers")
        feature_stats = compute_feature_stats(history_blocks)
        samples = build_samples(history_blocks, feature_stats)
        open_trade_rows, closed_trade_rows = read_trade_state(self.gc, self.cfg)
        trade_stats = aggregate_trade_stats(open_trade_rows, closed_trade_rows)

        timeout_state = load_timeout_state(self.cfg.timeout_state_file)
        timeout_state = refresh_timeout_state(timeout_state, trade_stats, self.cfg)
        save_timeout_state(self.cfg.timeout_state_file, timeout_state)

        timed_out_strategy_ids = {
            strategy_id
            for strategy_id in timeout_state.keys()
            if is_strategy_timed_out(strategy_id, timeout_state)
        }

        with self.lock:
            incumbent_bundle = self.best_bundle

        bundle = evolve_strategies(
            samples,
            feature_stats,
            self.cfg,
            self.log,
            incumbent_bundle=incumbent_bundle,
            timed_out_strategy_ids=timed_out_strategy_ids,
        )
        save_bundle(bundle, self.cfg.state_file)

        ss = self._sheet()
        leaderboard_ws = get_or_create_ws(ss, self.cfg.leaderboard_tab, rows=max(1000, self.cfg.leaderboard_size + 10), cols=40)
        best_ws = get_or_create_ws(ss, self.cfg.best_strategy_tab, rows=200, cols=10)

        leaderboard_headers = [
            "rank",
            "strategy_id",
            "fitness",
            "trades",
            "win_rate",
            "total_pnl_pct",
            "avg_pnl_pct",
            "profit_factor",
            "tp_exits",
            "tp_rate",
            "stop_exits",
            "stop_rate",
            "open_at_end_exits",
            "open_at_end_rate",
            "armed_rate",
            "validation_trades",
            "validation_win_rate",
            "validation_pnl_pct",
            "validation_tp_rate",
            "validation_stop_rate",
            "validation_open_at_end_rate",
            "live_open_trades",
            "live_closed_trades",
            "live_closed_win_rate",
            "timed_out_until",
            "threshold",
            "arm_pct",
            "trail_drop_pct",
            "stop_loss_pct",
            "trained_at",
            "weights_json",
            "signature",
        ]

        rows = []
        for entry in bundle.leaderboard or [LeaderboardEntry(rank=1, genome=bundle.genome, metrics=bundle.metrics)]:
            live = trade_stats.get(entry.genome.strategy_id, TradeStats(strategy_id=entry.genome.strategy_id))
            timeout_info = timeout_state.get(entry.genome.strategy_id, {})
            rows.append([
                entry.rank,
                entry.genome.strategy_id,
                round(entry.metrics.fitness, 4),
                entry.metrics.trades,
                round(entry.metrics.win_rate, 2),
                round(entry.metrics.total_pnl_pct, 4),
                round(entry.metrics.avg_pnl_pct, 4),
                round(entry.metrics.profit_factor, 4),
                entry.metrics.tp_exits,
                round(entry.metrics.tp_rate, 2),
                entry.metrics.stop_exits,
                round(entry.metrics.stop_rate, 2),
                entry.metrics.open_at_end_exits,
                round(entry.metrics.open_at_end_rate, 2),
                round(entry.metrics.armed_rate, 2),
                entry.metrics.validation_trades,
                round(entry.metrics.validation_win_rate, 2),
                round(entry.metrics.validation_pnl_pct, 4),
                round(entry.metrics.validation_tp_rate, 2),
                round(entry.metrics.validation_stop_rate, 2),
                round(entry.metrics.validation_open_at_end_rate, 2),
                live.open_trades,
                live.closed_trades,
                round(live.closed_win_rate, 2),
                timeout_info.get("until", ""),
                round(entry.genome.threshold, 2),
                round(entry.genome.arm_pct, 4),
                round(entry.genome.trail_drop_pct, 4),
                round(entry.genome.stop_loss_pct, 4),
                bundle.trained_at,
                json.dumps(entry.genome.weights, separators=(",", ":")),
                genome_signature(entry.genome),
            ])
        write_table(leaderboard_ws, leaderboard_headers, rows)

        best_live = trade_stats.get(bundle.genome.strategy_id, TradeStats(strategy_id=bundle.genome.strategy_id))
        best_timeout_info = timeout_state.get(bundle.genome.strategy_id, {})
        write_key_values(best_ws, [
            ("strategy_id", bundle.genome.strategy_id),
            ("trained_at", bundle.trained_at),
            ("fitness", round(bundle.metrics.fitness, 4)),
            ("trades", bundle.metrics.trades),
            ("win_rate", round(bundle.metrics.win_rate, 2)),
            ("total_pnl_pct", round(bundle.metrics.total_pnl_pct, 4)),
            ("avg_pnl_pct", round(bundle.metrics.avg_pnl_pct, 4)),
            ("profit_factor", round(bundle.metrics.profit_factor, 4)),
            ("tp_exits", bundle.metrics.tp_exits),
            ("tp_rate", round(bundle.metrics.tp_rate, 2)),
            ("stop_exits", bundle.metrics.stop_exits),
            ("stop_rate", round(bundle.metrics.stop_rate, 2)),
            ("open_at_end_exits", bundle.metrics.open_at_end_exits),
            ("open_at_end_rate", round(bundle.metrics.open_at_end_rate, 2)),
            ("armed_rate", round(bundle.metrics.armed_rate, 2)),
            ("validation_trades", bundle.metrics.validation_trades),
            ("validation_win_rate", round(bundle.metrics.validation_win_rate, 2)),
            ("validation_pnl_pct", round(bundle.metrics.validation_pnl_pct, 4)),
            ("validation_tp_rate", round(bundle.metrics.validation_tp_rate, 2)),
            ("validation_stop_rate", round(bundle.metrics.validation_stop_rate, 2)),
            ("validation_open_at_end_rate", round(bundle.metrics.validation_open_at_end_rate, 2)),
            ("live_open_trades", best_live.open_trades),
            ("live_closed_trades", best_live.closed_trades),
            ("live_closed_win_rate", round(best_live.closed_win_rate, 2)),
            ("timed_out_until", best_timeout_info.get("until", "")),
            ("threshold", round(bundle.genome.threshold, 2)),
            ("arm_pct", round(bundle.genome.arm_pct, 4)),
            ("trail_drop_pct", round(bundle.genome.trail_drop_pct, 4)),
            ("stop_loss_pct", round(bundle.genome.stop_loss_pct, 4)),
            ("signature", genome_signature(bundle.genome)),
            ("weights", bundle.genome.weights),
        ])

        with self.lock:
            self.best_bundle = bundle
            self.last_train_at = bundle.trained_at
            self.last_generation_summary = {
                "samples": len(samples),
                "history_blocks": len(history_blocks),
                "screener_rows": len(screener_rows),
                "open_trade_rows": len(open_trade_rows),
                "closed_trade_rows": len(closed_trade_rows),
                "timed_out_strategies": len(timed_out_strategy_ids),
                "strategy_id": bundle.genome.strategy_id,
                "fitness": bundle.metrics.fitness,
                "threshold": bundle.genome.threshold,
                "arm_pct": bundle.genome.arm_pct,
                "trail_drop_pct": bundle.genome.trail_drop_pct,
                "stop_loss_pct": bundle.genome.stop_loss_pct,
                "leaderboard_size": self.cfg.leaderboard_size,
                "immigrant_fraction": self.cfg.immigrant_fraction,
                "immigrants_per_generation": max(1, int(round(self.cfg.population_size * self.cfg.immigrant_fraction))) if self.cfg.population_size > 2 else 0,
                "strategies_tested_per_train": self.cfg.population_size * self.cfg.generations,
                "seeded_from_incumbent": incumbent_bundle is not None,
            }

        return {
            "ok": True,
            "trained_at": bundle.trained_at,
            "strategy_id": bundle.genome.strategy_id,
            "fitness": bundle.metrics.fitness,
            "trades": bundle.metrics.trades,
            "win_rate": bundle.metrics.win_rate,
            "tp_rate": bundle.metrics.tp_rate,
            "stop_rate": bundle.metrics.stop_rate,
            "open_at_end_rate": bundle.metrics.open_at_end_rate,
            "total_pnl_pct": bundle.metrics.total_pnl_pct,
            "live_closed_trades": best_live.closed_trades,
            "live_closed_win_rate": best_live.closed_win_rate,
        }

    def score_once(self) -> Dict[str, Any]:
        with self.lock:
            bundle = self.best_bundle
        if bundle is None:
            raise RuntimeError("No trained strategy is available yet.")

        headers, screener_rows, history_blocks = read_screener_and_history(self.gc, self.cfg)
        _ = headers, history_blocks

        timeout_state = load_timeout_state(self.cfg.timeout_state_file)

        active_genome = bundle.genome
        active_metrics = bundle.metrics

        if is_strategy_timed_out(active_genome.strategy_id, timeout_state):
            fallback: Optional[LeaderboardEntry] = None
            for entry in bundle.leaderboard:
                if not is_strategy_timed_out(entry.genome.strategy_id, timeout_state):
                    fallback = entry
                    break

            if fallback is not None:
                active_genome = fallback.genome
                active_metrics = fallback.metrics
            else:
                rows = build_flat_signal_rows(
                    screener_rows,
                    strategy_id=active_genome.strategy_id,
                    trained_at=bundle.trained_at,
                )
                ss = self._sheet()
                signals_ws = get_or_create_ws(ss, self.cfg.signals_tab, rows=max(500, len(rows) + 5), cols=len(SIGNAL_HEADERS) + 5)
                write_table(signals_ws, SIGNAL_HEADERS, rows)

                with self.lock:
                    self.last_signal_at = utc_now_iso()

                return {
                    "ok": True,
                    "signals_written": len(rows),
                    "last_signal_at": self.last_signal_at,
                    "strategy_id": active_genome.strategy_id,
                    "timed_out": True,
                    "fallback_strategy_id": None,
                }

        active_bundle = BestStrategyBundle(
            genome=active_genome,
            metrics=active_metrics,
            feature_stats=bundle.feature_stats,
            trained_at=bundle.trained_at,
            leaderboard=bundle.leaderboard,
        )

        rows = build_signal_rows(active_bundle, screener_rows)
        ss = self._sheet()
        signals_ws = get_or_create_ws(ss, self.cfg.signals_tab, rows=max(500, len(rows) + 5), cols=len(SIGNAL_HEADERS) + 5)
        write_table(signals_ws, SIGNAL_HEADERS, rows)

        with self.lock:
            self.last_signal_at = utc_now_iso()

        return {
            "ok": True,
            "signals_written": len(rows),
            "last_signal_at": self.last_signal_at,
            "strategy_id": active_genome.strategy_id,
            "arm_pct": active_genome.arm_pct,
            "trail_drop_pct": active_genome.trail_drop_pct,
            "stop_loss_pct": active_genome.stop_loss_pct,
            "timed_out": active_genome.strategy_id != bundle.genome.strategy_id,
            "fallback_strategy_id": active_genome.strategy_id if active_genome.strategy_id != bundle.genome.strategy_id else None,
        }

    def cycle_once(self) -> Dict[str, Any]:
        now = datetime.now(timezone.utc)
        should_train = False
        with self.lock:
            if self.best_bundle is None:
                should_train = True
            elif self.last_train_at:
                last = datetime.fromisoformat(self.last_train_at.replace("Z", "+00:00"))
                should_train = (now - last).total_seconds() >= self.cfg.retrain_minutes * 60
            else:
                should_train = True

        result: Dict[str, Any] = {}
        if should_train:
            self.log.info("Starting evolution cycle")
            result["train"] = self.train_once()
        result["signals"] = self.score_once()
        return result

    def worker(self) -> None:
        self.log.info("Background worker started")
        while not self.stop_event.is_set():
            cycle_started = time.time()
            try:
                self.cycle_once()
                self.last_error = None
            except Exception as exc:
                self.last_error = str(exc)
                self.log.exception("Worker cycle failed: %s", exc)
            elapsed = time.time() - cycle_started
            sleep_for = max(5.0, self.cfg.signal_poll_seconds - elapsed)
            self.stop_event.wait(sleep_for)
        self.log.info("Background worker stopped")

    def start(self) -> None:
        if self.thread and self.thread.is_alive():
            return
        self.thread = threading.Thread(target=self.worker, daemon=True, name="strategy-worker")
        self.thread.start()

    def stop(self) -> None:
        self.stop_event.set()
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=10)

    def status(self) -> Dict[str, Any]:
        timeout_state = load_timeout_state(self.cfg.timeout_state_file)
        active_timeouts = {
            strategy_id: info
            for strategy_id, info in timeout_state.items()
            if is_strategy_timed_out(strategy_id, timeout_state)
        }
        with self.lock:
            bundle = self.best_bundle
            return {
                "running": self.thread.is_alive() if self.thread else False,
                "last_train_at": self.last_train_at,
                "last_signal_at": self.last_signal_at,
                "last_error": self.last_error,
                "best_strategy": asdict(bundle.genome) if bundle else None,
                "best_metrics": asdict(bundle.metrics) if bundle else None,
                "summary": self.last_generation_summary,
                "active_timeouts": active_timeouts,
            }


# ----------------------------
# FastAPI app
# ----------------------------

cfg = load_config()

logging.basicConfig(
    level=getattr(logging, cfg.log_level, logging.INFO),
    format="%(asctime)s | %(levelname)s | %(message)s",
)

random.seed(cfg.random_seed)
service = StrategyService(cfg)


@asynccontextmanager
async def lifespan(app: FastAPI):
    service.start()
    try:
        yield
    finally:
        service.stop()


app = FastAPI(
    title="FX Strategy Generator",
    version="2.3.0",
    lifespan=lifespan,
)


@app.get("/")
def root() -> Dict[str, Any]:
    return {
        "service": "fx-strategy-generator",
        "status": service.status(),
    }


@app.get("/health")
def health() -> Dict[str, Any]:
    status = service.status()
    ok = status["running"] and status["last_error"] is None
    return {"ok": ok, "status": status}


@app.get("/status")
def status() -> Dict[str, Any]:
    return service.status()


@app.post("/train")
def train_now() -> Dict[str, Any]:
    try:
        result = service.train_once()
        return result
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/score")
def score_now() -> Dict[str, Any]:
    try:
        return service.score_once()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/cycle")
def cycle_now() -> Dict[str, Any]:
    try:
        return service.cycle_once()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))