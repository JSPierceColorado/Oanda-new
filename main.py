
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
from datetime import datetime, timezone
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


def chunked(items: List[str], n: int) -> List[List[str]]:
    return [items[i:i + n] for i in range(0, len(items), n)]


def col_to_a1(n: int) -> str:
    s = ""
    while n > 0:
        n, r = divmod(n - 1, 26)
        s = chr(65 + r) + s
    return s


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

    signal_poll_seconds: int = 60
    retrain_minutes: int = 60

    population_size: int = 80
    generations: int = 30
    elite_count: int = 10
    mutation_rate: float = 0.18
    mutation_strength: float = 0.45
    hold_min_steps: int = 1
    hold_max_steps: int = 8
    threshold_min: float = 20.0
    threshold_max: float = 75.0
    min_trades: int = 15
    validation_split: float = 0.25
    random_seed: int = 42

    state_file: str = "/tmp/best_strategy.json"
    log_level: str = "INFO"

    def __post_init__(self) -> None:
        if self.elite_count >= self.population_size:
            self.elite_count = max(1, self.population_size // 4)


def load_config() -> Config:
    return Config(
        sheet_id=env_str("SHEET_ID"),
        screener_tab=os.getenv("SCREENER_TAB", "Screener"),
        history_tab=os.getenv("HISTORY_TAB", "history"),
        signals_tab=os.getenv("SIGNALS_TAB", "Signals"),
        leaderboard_tab=os.getenv("LEADERBOARD_TAB", "StrategyLeaderboard"),
        best_strategy_tab=os.getenv("BEST_STRATEGY_TAB", "BestStrategy"),
        signal_poll_seconds=env_int("SIGNAL_POLL_SECONDS", 60),
        retrain_minutes=env_int("RETRAIN_MINUTES", 60),
        population_size=env_int("POPULATION_SIZE", 80),
        generations=env_int("GENERATIONS", 30),
        elite_count=env_int("ELITE_COUNT", 10),
        mutation_rate=env_float("MUTATION_RATE", 0.18),
        mutation_strength=env_float("MUTATION_STRENGTH", 0.45),
        hold_min_steps=env_int("HOLD_MIN_STEPS", 1),
        hold_max_steps=env_int("HOLD_MAX_STEPS", 8),
        threshold_min=env_float("THRESHOLD_MIN", 20.0),
        threshold_max=env_float("THRESHOLD_MAX", 75.0),
        min_trades=env_int("MIN_TRADES", 15),
        validation_split=env_float("VALIDATION_SPLIT", 0.25),
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
    future_returns: Dict[int, float]
    spread_cost_pct: float
    current_price: float


@dataclass
class StrategyGenome:
    strategy_id: str
    weights: Dict[str, float]
    bias: float
    threshold: float
    hold_steps: int


@dataclass
class StrategyMetrics:
    trades: int = 0
    wins: int = 0
    losses: int = 0
    win_rate: float = 0.0
    total_pnl_pct: float = 0.0
    avg_pnl_pct: float = 0.0
    profit_factor: float = 0.0
    long_trades: int = 0
    short_trades: int = 0
    validation_trades: int = 0
    validation_win_rate: float = 0.0
    validation_pnl_pct: float = 0.0
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


def build_samples(blocks: List[List[Dict[str, str]]], feature_stats: Dict[str, FeatureStats], hold_max_steps: int) -> List[Sample]:
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
        if len(rows) <= hold_max_steps:
            continue
        for i in range(len(rows) - hold_max_steps):
            current = rows[i]
            entry_price = pick_price(current)
            if entry_price is None or entry_price <= 0:
                continue

            future_returns: Dict[int, float] = {}
            for hold in range(1, hold_max_steps + 1):
                future_price = pick_price(rows[i + hold])
                if future_price is None or future_price <= 0:
                    continue
                future_returns[hold] = ((future_price / entry_price) - 1.0) * 100.0

            if not future_returns:
                continue

            spread_pct = safe_float(current.get("spread_pct"), 0.0) or 0.0
            spread_cost_pct = max(0.0, spread_pct) * 2.0
            samples.append(
                Sample(
                    symbol=symbol,
                    asof_utc=str(current.get("asof_utc", "")),
                    features=normalize_features(current, feature_stats),
                    future_returns=future_returns,
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


def simulate(genome: StrategyGenome, samples: Iterable[Sample]) -> StrategyMetrics:
    metrics = StrategyMetrics()
    gross_profit = 0.0
    gross_loss = 0.0

    for sample in samples:
        forward = sample.future_returns.get(genome.hold_steps)
        if forward is None:
            continue

        score = score_strategy(genome, sample.features)
        if abs(score) < genome.threshold:
            continue

        metrics.trades += 1
        direction = 1.0 if score > 0 else -1.0
        if direction > 0:
            metrics.long_trades += 1
        else:
            metrics.short_trades += 1

        pnl_pct = direction * forward - sample.spread_cost_pct
        metrics.total_pnl_pct += pnl_pct

        if pnl_pct > 0:
            metrics.wins += 1
            gross_profit += pnl_pct
        else:
            metrics.losses += 1
            gross_loss += abs(pnl_pct)

    if metrics.trades > 0:
        metrics.win_rate = metrics.wins / metrics.trades * 100.0
        metrics.avg_pnl_pct = metrics.total_pnl_pct / metrics.trades
    if gross_loss > 0:
        metrics.profit_factor = gross_profit / gross_loss
    elif gross_profit > 0:
        metrics.profit_factor = gross_profit

    return metrics


def combine_fitness(train_metrics: StrategyMetrics, val_metrics: StrategyMetrics, min_trades: int) -> float:
    trade_factor = min(1.0, train_metrics.trades / max(1, min_trades))
    val_trade_factor = min(1.0, val_metrics.trades / max(1, max(5, min_trades // 2)))

    train_score = (
        train_metrics.total_pnl_pct
        + train_metrics.win_rate * 0.45
        + train_metrics.avg_pnl_pct * 25.0
        + min(train_metrics.profit_factor, 5.0) * 2.0
    ) * trade_factor

    val_score = (
        val_metrics.total_pnl_pct
        + val_metrics.win_rate * 0.55
        + val_metrics.avg_pnl_pct * 30.0
        + min(val_metrics.profit_factor, 5.0) * 2.0
    ) * val_trade_factor

    shortage_penalty = max(0, min_trades - train_metrics.trades) * 2.0
    return train_score * 0.65 + val_score * 0.35 - shortage_penalty


def random_genome(cfg: Config) -> StrategyGenome:
    return StrategyGenome(
        strategy_id=str(uuid.uuid4())[:8],
        weights={feature: random.uniform(-2.0, 2.0) for feature in FEATURE_COLUMNS},
        bias=random.uniform(-1.5, 1.5),
        threshold=random.uniform(cfg.threshold_min, cfg.threshold_max),
        hold_steps=random.randint(cfg.hold_min_steps, cfg.hold_max_steps),
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

    return StrategyGenome(
        strategy_id=str(uuid.uuid4())[:8],
        weights=child_weights,
        bias=(a.bias + b.bias) / 2.0 if random.random() < 0.5 else random.choice([a.bias, b.bias]),
        threshold=random.choice([a.threshold, b.threshold, (a.threshold + b.threshold) / 2.0]),
        hold_steps=random.choice([a.hold_steps, b.hold_steps]),
    )


def mutate(genome: StrategyGenome, cfg: Config) -> StrategyGenome:
    child = StrategyGenome(
        strategy_id=str(uuid.uuid4())[:8],
        weights=dict(genome.weights),
        bias=genome.bias,
        threshold=genome.threshold,
        hold_steps=genome.hold_steps,
    )

    for feature in FEATURE_COLUMNS:
        if random.random() < cfg.mutation_rate:
            child.weights[feature] += random.gauss(0.0, cfg.mutation_strength)
            child.weights[feature] = clip(child.weights[feature], -4.0, 4.0)

    if random.random() < cfg.mutation_rate:
        child.bias = clip(child.bias + random.gauss(0.0, cfg.mutation_strength), -4.0, 4.0)

    if random.random() < cfg.mutation_rate:
        child.threshold = clip(
            child.threshold + random.gauss(0.0, cfg.mutation_strength * 15.0),
            cfg.threshold_min,
            cfg.threshold_max,
        )

    if random.random() < cfg.mutation_rate:
        child.hold_steps = clip(
            child.hold_steps + random.choice([-2, -1, 1, 2]),
            cfg.hold_min_steps,
            cfg.hold_max_steps,
        )
        child.hold_steps = int(child.hold_steps)

    return child


def tournament_select(scored: List[Tuple[StrategyGenome, StrategyMetrics]], k: int = 4) -> StrategyGenome:
    pool = random.sample(scored, k=min(k, len(scored)))
    pool.sort(key=lambda item: item[1].fitness, reverse=True)
    return pool[0][0]


def evolve_strategies(samples: List[Sample], feature_stats: Dict[str, FeatureStats], cfg: Config, log: logging.Logger) -> BestStrategyBundle:
    if len(samples) < max(20, cfg.min_trades + cfg.hold_max_steps):
        raise RuntimeError("Not enough history to evolve strategies yet.")

    split_idx = max(1, int(len(samples) * (1.0 - cfg.validation_split)))
    train_samples = samples[:split_idx]
    val_samples = samples[split_idx:] if split_idx < len(samples) else samples[-max(5, len(samples)//4):]

    population = [random_genome(cfg) for _ in range(cfg.population_size)]
    top_scored: List[Tuple[StrategyGenome, StrategyMetrics]] = []

    for generation in range(1, cfg.generations + 1):
        scored: List[Tuple[StrategyGenome, StrategyMetrics]] = []
        for genome in population:
            train_metrics = simulate(genome, train_samples)
            val_metrics = simulate(genome, val_samples)
            train_metrics.validation_trades = val_metrics.trades
            train_metrics.validation_win_rate = val_metrics.win_rate
            train_metrics.validation_pnl_pct = val_metrics.total_pnl_pct
            train_metrics.fitness = combine_fitness(train_metrics, val_metrics, cfg.min_trades)
            scored.append((genome, train_metrics))

        scored.sort(key=lambda item: item[1].fitness, reverse=True)
        best_genome, best_metrics = scored[0]
        log.info(
            "generation=%d best_fitness=%.2f trades=%d win_rate=%.1f total_pnl=%.2f hold=%d threshold=%.1f",
            generation,
            best_metrics.fitness,
            best_metrics.trades,
            best_metrics.win_rate,
            best_metrics.total_pnl_pct,
            best_genome.hold_steps,
            best_genome.threshold,
        )
        top_scored = scored[: max(10, cfg.elite_count)]

        next_population: List[StrategyGenome] = [g for g, _ in scored[: cfg.elite_count]]
        while len(next_population) < cfg.population_size:
            parent_a = tournament_select(scored)
            parent_b = tournament_select(scored)
            child = crossover(parent_a, parent_b, cfg)
            child = mutate(child, cfg)
            next_population.append(child)
        population = next_population

    top_scored.sort(key=lambda item: item[1].fitness, reverse=True)
    best_genome, best_metrics = top_scored[0]

    leaderboard: List[LeaderboardEntry] = []
    for rank, (genome, metrics) in enumerate(top_scored[:10], start=1):
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


def load_bundle(path: str) -> Optional[BestStrategyBundle]:
    p = Path(path)
    if not p.exists():
        return None
    data = json.loads(p.read_text(encoding="utf-8"))
    return BestStrategyBundle(
        genome=StrategyGenome(**data["genome"]),
        metrics=StrategyMetrics(**data["metrics"]),
        feature_stats={k: FeatureStats(**v) for k, v in data["feature_stats"].items()},
        trained_at=data["trained_at"],
        leaderboard=[
            LeaderboardEntry(
                rank=entry["rank"],
                genome=StrategyGenome(**entry["genome"]),
                metrics=StrategyMetrics(**entry["metrics"]),
            )
            for entry in data.get("leaderboard", [])
        ],
    )


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
            bundle.genome.hold_steps,
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


SIGNAL_HEADERS = [
    "asof_utc",
    "symbol",
    "score",
    "side",
    "entry_signal",
    "threshold",
    "hold_steps",
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
        samples = build_samples(history_blocks, feature_stats, self.cfg.hold_max_steps)
        bundle = evolve_strategies(samples, feature_stats, self.cfg, self.log)
        save_bundle(bundle, self.cfg.state_file)

        ss = self._sheet()
        leaderboard_ws = get_or_create_ws(ss, self.cfg.leaderboard_tab, rows=1000, cols=30)
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
            "validation_trades",
            "validation_win_rate",
            "validation_pnl_pct",
            "hold_steps",
            "threshold",
            "trained_at",
            "weights_json",
        ]

        rows = []
        for entry in bundle.leaderboard or [LeaderboardEntry(rank=1, genome=bundle.genome, metrics=bundle.metrics)]:
            rows.append([
                entry.rank,
                entry.genome.strategy_id,
                round(entry.metrics.fitness, 4),
                entry.metrics.trades,
                round(entry.metrics.win_rate, 2),
                round(entry.metrics.total_pnl_pct, 4),
                round(entry.metrics.avg_pnl_pct, 4),
                round(entry.metrics.profit_factor, 4),
                entry.metrics.validation_trades,
                round(entry.metrics.validation_win_rate, 2),
                round(entry.metrics.validation_pnl_pct, 4),
                entry.genome.hold_steps,
                round(entry.genome.threshold, 2),
                bundle.trained_at,
                json.dumps(entry.genome.weights, separators=(",", ":")),
            ])
        write_table(leaderboard_ws, leaderboard_headers, rows)

        write_key_values(best_ws, [
            ("strategy_id", bundle.genome.strategy_id),
            ("trained_at", bundle.trained_at),
            ("fitness", round(bundle.metrics.fitness, 4)),
            ("trades", bundle.metrics.trades),
            ("win_rate", round(bundle.metrics.win_rate, 2)),
            ("total_pnl_pct", round(bundle.metrics.total_pnl_pct, 4)),
            ("avg_pnl_pct", round(bundle.metrics.avg_pnl_pct, 4)),
            ("profit_factor", round(bundle.metrics.profit_factor, 4)),
            ("validation_trades", bundle.metrics.validation_trades),
            ("validation_win_rate", round(bundle.metrics.validation_win_rate, 2)),
            ("validation_pnl_pct", round(bundle.metrics.validation_pnl_pct, 4)),
            ("hold_steps", bundle.genome.hold_steps),
            ("threshold", round(bundle.genome.threshold, 2)),
            ("weights", bundle.genome.weights),
        ])

        with self.lock:
            self.best_bundle = bundle
            self.last_train_at = bundle.trained_at
            self.last_generation_summary = {
                "samples": len(samples),
                "history_blocks": len(history_blocks),
                "screener_rows": len(screener_rows),
                "strategy_id": bundle.genome.strategy_id,
                "fitness": bundle.metrics.fitness,
            }

        return {
            "ok": True,
            "trained_at": bundle.trained_at,
            "strategy_id": bundle.genome.strategy_id,
            "fitness": bundle.metrics.fitness,
            "trades": bundle.metrics.trades,
            "win_rate": bundle.metrics.win_rate,
            "total_pnl_pct": bundle.metrics.total_pnl_pct,
        }

    def score_once(self) -> Dict[str, Any]:
        with self.lock:
            bundle = self.best_bundle
        if bundle is None:
            raise RuntimeError("No trained strategy is available yet.")

        headers, screener_rows, history_blocks = read_screener_and_history(self.gc, self.cfg)
        _ = headers, history_blocks

        rows = build_signal_rows(bundle, screener_rows)
        ss = self._sheet()
        signals_ws = get_or_create_ws(ss, self.cfg.signals_tab, rows=max(500, len(rows) + 5), cols=len(SIGNAL_HEADERS) + 5)
        write_table(signals_ws, SIGNAL_HEADERS, rows)

        with self.lock:
            self.last_signal_at = utc_now_iso()

        return {
            "ok": True,
            "signals_written": len(rows),
            "last_signal_at": self.last_signal_at,
            "strategy_id": bundle.genome.strategy_id,
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
    version="1.0.0",
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
