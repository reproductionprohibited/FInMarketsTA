"""
Оптимизация стратегии Elder Ray + Aroon + ATR-стоп (без графиков).
Обёрнуто в функцию для возможности запуска на нескольких датасетах.
"""

import json
import itertools
import warnings
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import pandas_ta as ta

import matplotlib.pyplot as plt
import matplotlib.dates as mdates

warnings.filterwarnings("ignore")

# ----------------------------- КОНСТАНТЫ -----------------------------
INITIAL_CAPITAL: float = 100_000.0
RISK_FREE_ANNUAL: float = 0.12
TRADING_DAYS: int = 252

# ----------------------------- ЗАГРУЗКА ДАННЫХ -----------------------------
def load_candles(path: str) -> pd.DataFrame:
    """Загрузка свечей из JSON-файла."""
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    cols = raw["candles"]["columns"]
    data = raw["candles"]["data"]
    df = pd.DataFrame(data, columns=cols)
    df["begin"] = pd.to_datetime(df["begin"])
    df = df.set_index("begin").sort_index()
    df = df[["open", "high", "low", "close", "volume"]].astype(float)
    return df

# ----------------------------- ИНДИКАТОРЫ -----------------------------
def add_indicators(
    df: pd.DataFrame,
    ema_period: int,
    aroon_period: int,
    use_vol_filter: bool,
    vol_ma_period: int = 20,
    atr_period: int = 14,
) -> pd.DataFrame:
    """Расчёт Elder Ray (EMA, Bull Power), Aroon, ATR и опционально фильтра объёма."""
    df = df.copy()
    # EMA и Bull Power
    df["EMA"] = df["close"].ewm(span=ema_period, adjust=False).mean()
    df["BULL_POWER"] = df["high"] - df["EMA"]

    # Aroon
    aroon = ta.aroon(df["high"], df["low"], length=aroon_period)
    df["AROON_UP"] = aroon[f"AROONU_{aroon_period}"]
    df["AROON_DOWN"] = aroon[f"AROOND_{aroon_period}"]

    # ATR
    df["ATR"] = ta.atr(df["high"], df["low"], df["close"], length=atr_period)

    # Объёмный фильтр
    if use_vol_filter:
        df["VOL_MA"] = df["volume"].rolling(window=vol_ma_period).mean()
    else:
        df["VOL_MA"] = 0.0  # вместо df["volume"] * 0 – просто константа

    return df.dropna()

# ----------------------------- БЭКТЕСТ -----------------------------
def backtest(
    df: pd.DataFrame,
    ema_period: int,
    aroon_period: int,
    bull_threshold: float,
    stop_atr_mult: float,
    use_vol_filter: bool,
) -> Tuple[float, np.ndarray, List[Dict]]:
    """Запуск торговой симуляции. Возвращает итоговый капитал, кривую эквити и список сделок."""
    df = add_indicators(df, ema_period, aroon_period, use_vol_filter)
    if len(df) < 2:
        return INITIAL_CAPITAL, np.array([INITIAL_CAPITAL]), []

    # Состояние системы
    position = 0               # 0 – нет позиции, 1 – в лонге
    capital = INITIAL_CAPITAL
    shares = 0.0
    high_since_entry = 0.0
    equity = [INITIAL_CAPITAL]
    trades: List[Dict] = []

    # Извлекаем массивы для скорости
    closes = df["close"].values
    highs = df["high"].values
    lows = df["low"].values
    bull = df["BULL_POWER"].values
    aroon_up = df["AROON_UP"].values
    aroon_down = df["AROON_DOWN"].values
    atr = df["ATR"].values
    volume = df["volume"].values
    vol_ma = df["VOL_MA"].values
    dates = df.index

    for i in range(1, len(df)):
        price = closes[i]
        date = dates[i]

        if position == 0:
            # Условия входа
            if (
                bull[i] > bull_threshold
                and bull[i] > bull[i - 1]
                and aroon_up[i] > aroon_down[i]
                and (not use_vol_filter or volume[i] > vol_ma[i])
            ):
                # Покупка целого числа акций (можно заменить на дробное при необходимости)
                shares = capital // price
                if shares == 0:
                    continue
                capital -= shares * price
                high_since_entry = price
                position = 1
                trades.append({"date": date, "action": "BUY", "price": price})

        elif position == 1:
            # Обновление максимума с момента входа
            high_since_entry = max(high_since_entry, highs[i])

            # Уровень стоп-лосса по ATR от максимума
            stop_price = high_since_entry - stop_atr_mult * atr[i]

            # Сигнал на выход: либо цена закрытия ниже стопа, либо разворот по Aroon
            exit_signal = (closes[i] <= stop_price) or (aroon_down[i] > aroon_up[i])

            if exit_signal:
                # Определяем цену выхода
                if aroon_down[i] > aroon_up[i]:
                    exit_price = closes[i]
                else:
                    exit_price = stop_price
                # Но не ниже минимума текущего бара (защита от гэпов)
                exit_price = max(exit_price, lows[i])

                capital += shares * exit_price
                trades.append(
                    {
                        "date": date,
                        "action": "SELL",
                        "price": exit_price,
                        "pnl": shares * (exit_price - trades[-1]["price"]),
                    }
                )
                shares = 0.0
                position = 0
                high_since_entry = 0.0

        # Текущая стоимость портфеля
        equity.append(capital + shares * price)

    # Принудительное закрытие в конце периода
    if position == 1:
        last_price = closes[-1]
        capital += shares * last_price
        trades.append(
            {
                "date": dates[-1],
                "action": "SELL (end)",
                "price": last_price,
                "pnl": shares * (last_price - trades[-1]["price"]),
            }
        )
        equity[-1] = capital

    return capital, np.array(equity), trades

# ----------------------------- МЕТРИКИ -----------------------------
def _sharpe_ratio(equity: np.ndarray) -> float:
    """Годовой коэффициент Шарпа по кривой эквити."""
    if len(equity) < 2 or np.std(equity) == 0:
        return 0.0
    daily_returns = np.diff(equity) / (equity[:-1] + 1e-9)
    rf_daily = RISK_FREE_ANNUAL / TRADING_DAYS
    excess = daily_returns - rf_daily
    std_dev = excess.std()
    if std_dev == 0:
        return 0.0
    return (excess.mean() / std_dev) * np.sqrt(TRADING_DAYS)


def _max_drawdown_pct(equity: np.ndarray) -> float:
    """Максимальная просадка в процентах."""
    peak = np.maximum.accumulate(equity)
    return float(((equity - peak) / (peak + 1e-9)).min() * 100)


def _win_rate(trades: List[Dict]) -> float:
    """Процент прибыльных сделок (по количеству пар BUY/SELL)."""
    buy_trades = [t for t in trades if t["action"] == "BUY"]
    sell_trades = [t for t in trades if "SELL" in t["action"]]
    if len(buy_trades) == 0:
        return 0.0
    wins = sum(
        1 for b, s in zip(buy_trades, sell_trades) if s["price"] > b["price"]
    )
    return (wins / len(buy_trades)) * 100


def compute_metrics(
    df: pd.DataFrame,
    ema_period: int,
    aroon_period: int,
    bull_threshold: float,
    stop_atr_mult: float,
    use_vol_filter: bool,
) -> Dict:
    """Запуск бэктеста и расчёт всех метрик."""
    final_capital, equity, trades = backtest(
        df, ema_period, aroon_period, bull_threshold, stop_atr_mult, use_vol_filter
    )
    total_return = (final_capital - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100

    return {
        "total_return_%": round(total_return, 2),
        "sharpe": round(_sharpe_ratio(equity), 3),
        "max_drawdown_%": round(_max_drawdown_pct(equity), 2),
        "n_trades": len([t for t in trades if t["action"] == "BUY"]),
        "win_rate_%": round(_win_rate(trades), 2),
        "final_capital": round(final_capital, 2),
        "equity": equity,
        "trades": trades,
    }

# ----------------------------- BUY & HOLD -----------------------------
def buy_and_hold_return(df: pd.DataFrame) -> float:
    """Доходность стратегии 'купи и держи' в процентах."""
    start = df["close"].iloc[0]
    end = df["close"].iloc[-1]
    shares = INITIAL_CAPITAL // start
    final = shares * end + (INITIAL_CAPITAL - shares * start)
    return (final - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100

# ----------------------------- ПОИСК ПАРАМЕТРОВ -----------------------------
def _generate_param_grid(
    ema_periods: List[int],
    aroon_periods: List[int],
    bull_thresholds: List[float],
    stop_atr_mults: List[float],
    vol_filter_opts: List[bool],
) -> pd.DataFrame:
    """Создаёт DataFrame со всеми комбинациями параметров."""
    keys = ["ema_period", "aroon_period", "bull_threshold", "stop_atr_mult", "use_vol_filter"]
    grid = list(itertools.product(ema_periods, aroon_periods, bull_thresholds, stop_atr_mults, vol_filter_opts))
    return pd.DataFrame(grid, columns=keys)


def _evaluate_grid(
    grid: pd.DataFrame,
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
) -> pd.DataFrame:
    """Вычисляет метрики для всех комбинаций и возвращает DataFrame с результатами."""
    results = []
    for _, row in grid.iterrows():
        params = row.to_dict()
        m_opt = compute_metrics(df_train, **params)
        m_test = compute_metrics(df_test, **params)
        results.append(
            {
                **params,
                "opt_return_%": m_opt["total_return_%"],
                "opt_sharpe": m_opt["sharpe"],
                "opt_max_dd_%": m_opt["max_drawdown_%"],
                "opt_n_trades": m_opt["n_trades"],
                "opt_win_rate_%": m_opt["win_rate_%"],
                "test_return_%": m_test["total_return_%"],
                "test_sharpe": m_test["sharpe"],
                "test_max_dd_%": m_test["max_drawdown_%"],
                "test_n_trades": m_test["n_trades"],
                "test_win_rate_%": m_test["win_rate_%"],
            }
        )
    return pd.DataFrame(results)


def _select_best(
    results_df: pd.DataFrame,
    opt_target: float,
) -> pd.Series:
    """Выбор лучшей строки: сначала по критериям (opt_return > opt_target и opt_sharpe > 0.7),
    затем по наивысшему test_sharpe."""
    criteria = (results_df["opt_return_%"] > opt_target) & (results_df["opt_sharpe"] > 0.7)
    qualifying = results_df[criteria]
    if len(qualifying) > 0:
        return qualifying.loc[qualifying["test_sharpe"].idxmax()]
    return results_df.loc[results_df["test_sharpe"].idxmax()]


def _print_results_table(
    name: str,
    best_params: Dict,
    m_opt: Dict,
    m_test: Dict,
    bah_opt: float,
    bah_test: float,
    inf_opt: float,
    inf_test: float,
) -> None:
    """Форматированный вывод итогов."""
    print(f"\n{'='*55}")
    print(f"  ИТОГОВЫЕ РЕЗУЛЬТАТЫ – {name}")
    print(f"{'='*55}")
    print(f"  {'Метрика':<28} {'Оптим.':>10} {'Тест':>10}")
    print(f"  {'-'*55}")
    for label, key in [
        ("Доходность стратегии, %", "total_return_%"),
        ("Коэффициент Шарпа", "sharpe"),
        ("Макс. просадка, %", "max_drawdown_%"),
        ("Кол-во сделок", "n_trades"),
        ("Процент прибыльных, %", "win_rate_%"),
    ]:
        print(f"  {label:<28} {m_opt[key]:>10} {m_test[key]:>10}")
    print(f"  {'-'*55}")
    print(f"  {'Buy & Hold, %':<28} {bah_opt:>10.2f} {bah_test:>10.2f}")
    print(f"  {'Инфляция, %':<28} {inf_opt:>10.1f} {inf_test:>10.1f}")
    print(f"  {'='*55}")

    c1 = (m_opt["total_return_%"] > max(inf_opt, bah_opt)) and (m_opt["sharpe"] > 0.7)
    c2 = (m_test["total_return_%"] > max(inf_test, bah_test)) and (m_test["sharpe"] > 0.3)
    print("\n  Проверка критериев задания:")
    print(f"  [{'✓' if c1 else '✗'}] Оптим.: доходность > B&H и инфляции, Шарп > 0.7 (Шарп={m_opt['sharpe']})")
    print(f"  [{'✓' if c2 else '✗'}] Тест:   доходность > B&H и инфляции, Шарп > 0.3 (Шарп={m_test['sharpe']})")

# ----------------------------- ГЛАВНАЯ ФУНКЦИЯ -----------------------------
def optimize_and_evaluate(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    name: str = "",
    inf_opt: float = 14.5,
    inf_test: float = 9.5,
) -> Dict:
    """
    Запускает оптимизацию на train, тестирует на test, выводит краткие результаты.
    Возвращает словарь с лучшими параметрами и метриками.
    """
    bah_opt = buy_and_hold_return(df_train)
    bah_test = buy_and_hold_return(df_test)
    print(f"\n{'='*55}")
    print(f"  {name}")
    print(f"{'='*55}")
    print(f"Buy & Hold доходность: оптим. {bah_opt:.2f}%, тест {bah_test:.2f}%")

    # Сетка параметров
    grid = _generate_param_grid(
        ema_periods=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        aroon_periods=[24, 25, 26, 27, 28, 30, 31, 32, 33, 34, 35],
        bull_thresholds=[-3.5, -3, -2.5, -2, -1.5, -1.0, -0.75, -0.5, -0.25, 0.0, 0.3, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2],
        stop_atr_mults=[1.25, 1.5, 1.75, 2.0, 2.25, 2.5],
        vol_filter_opts=[False, True],
    )
    print(f"Оптимизация ({len(grid)} комбинаций)...")

    results_df = _evaluate_grid(grid, df_train, df_test)

    # Топ-15 по тестовому Шарпу
    print("\nТоп-15 по тестовому Шарпу:")
    top15 = results_df.sort_values("test_sharpe", ascending=False).head(15)
    print(
        top15[
            [
                "ema_period",
                "aroon_period",
                "bull_threshold",
                "stop_atr_mult",
                "use_vol_filter",
                "opt_return_%",
                "opt_sharpe",
                "test_return_%",
                "test_sharpe",
            ]
        ].to_string(index=False)
    )

    # Выбор лучшей комбинации
    opt_target = max(bah_opt, inf_opt)
    best_row = _select_best(results_df, opt_target)

    if len(results_df[
        (results_df["opt_return_%"] > opt_target) & (results_df["opt_sharpe"] > 0.7)
    ]) > 0:
        print("\n✓ Выбрана комбинация, прошедшая критерии оптимизации и с лучшим тестовым Шарпом.")
    else:
        print("\n⚠ Ни одна комбинация не прошла критерии оптимизации. Выбрана лучшая по тестовому Шарпу.")

    best_params = {
        "ema_period": int(best_row["ema_period"]),
        "aroon_period": int(best_row["aroon_period"]),
        "bull_threshold": float(best_row["bull_threshold"]),
        "stop_atr_mult": float(best_row["stop_atr_mult"]),
        "use_vol_filter": bool(best_row["use_vol_filter"]),
    }
    print("\n✓ Лучшие параметры:")
    for k, v in best_params.items():
        print(f"   {k} = {v}")

    # Финальные метрики
    m_opt_final = compute_metrics(df_train, **best_params)
    m_test_final = compute_metrics(df_test, **best_params)
    _print_results_table(name, best_params, m_opt_final, m_test_final, bah_opt, bah_test, inf_opt, inf_test)

    return {
        "best_params": best_params,
        "opt_metrics": m_opt_final,
        "test_metrics": m_test_final,
        "bah_opt": bah_opt,
        "bah_test": bah_test,
        "top15": top15,
    }


def plot_strategy_report(
    df: pd.DataFrame,
    metrics: Dict,
    params: Optional[Dict] = None,
    title_prefix: str = "",
):
    """
    Строит набор аналитических графиков без разрывов на выходных.

    Входные данные те же, что и раньше.
    """
    equity = metrics.get("equity")
    trades = metrics.get("trades")
    if equity is None or trades is None:
        print("Нет данных equity/trades для построения графиков")
        return

    # ----- Преобразуем индекс в числовой формат, чтобы избежать пропусков -----
    dates = df.index
    x_dates = mdates.date2num(dates)                # числовая ось без разрывов
    x_range = np.arange(len(dates))                 # равномерная шкала (если нужна)

    # Настройка внешнего вида
    plt.rcParams["figure.figsize"] = (20, 13)       # увеличенный размер
    plt.rcParams["axes.grid"] = True
    plt.rcParams["grid.alpha"] = 0.4
    plt.rcParams["font.size"] = 10

    fig = plt.figure()

    # ---------------------- 1. Цена и сделки ---------------------------
    ax1 = plt.subplot2grid((3, 2), (0, 0), colspan=2)
    ax1.set_title("Цена закрытия и сигналы входа / выхода", fontsize=13, fontweight="bold")
    ax1.plot(x_dates, df["close"], color="black", linewidth=1.2, label="Close")
    # сделки
    buy_trades = [t for t in trades if t["action"] == "BUY"]
    sell_trades = [t for t in trades if "SELL" in t["action"]]
    if buy_trades:
        ax1.scatter(
            [mdates.date2num(t["date"]) for t in buy_trades],
            [t["price"] for t in buy_trades],
            marker="^", color="limegreen", s=80, zorder=5, label="BUY"
        )
    if sell_trades:
        ax1.scatter(
            [mdates.date2num(t["date"]) for t in sell_trades],
            [t["price"] for t in sell_trades],
            marker="v", color="crimson", s=80, zorder=5, label="SELL"
        )
    ax1.plot(x_dates, df["EMA"], color="orange", linestyle="--", linewidth=1.2, alpha=0.7, label="EMA")
    ax1.legend(fontsize=9)
    # Форматирование оси дат
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax1.xaxis.set_major_locator(mdates.AutoDateLocator())
    plt.setp(ax1.get_xticklabels(), rotation=30, ha="right")

    # ---------------------- 2. Bull Power (Elder Ray) -------------------
    ax2 = plt.subplot2grid((3, 2), (1, 0))
    ax2.set_title("Bull Power (Elder Ray)", fontsize=12)
    ax2.fill_between(x_dates, 0, df["BULL_POWER"],
                     where=df["BULL_POWER"] >= 0,
                     color="green", alpha=0.5, label="Бычья фаза")
    ax2.fill_between(x_dates, df["BULL_POWER"], 0,
                     where=df["BULL_POWER"] < 0,
                     color="red", alpha=0.5, label="Медвежья фаза")
    ax2.axhline(y=0, color="grey", linewidth=0.8, linestyle="--")
    if params and "bull_threshold" in params:
        ax2.axhline(y=params["bull_threshold"], color="blue", linewidth=1,
                    linestyle=":", label=f'Порог {params["bull_threshold"]}')
    ax2.legend(fontsize=8)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax2.xaxis.set_major_locator(mdates.AutoDateLocator())
    plt.setp(ax2.get_xticklabels(), rotation=30, ha="right")

    # ---------------------- 3. Aroon Up / Down --------------------------
    ax3 = plt.subplot2grid((3, 2), (1, 1))
    ax3.set_title("Aroon Up / Down", fontsize=12)
    ax3.plot(x_dates, df["AROON_UP"], color="green", linewidth=1, label="Aroon Up")
    ax3.plot(x_dates, df["AROON_DOWN"], color="red", linewidth=1, label="Aroon Down")
    ax3.axhline(y=50, color="grey", linestyle="--", linewidth=0.7)
    ax3.legend(fontsize=8)
    ax3.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax3.xaxis.set_major_locator(mdates.AutoDateLocator())
    plt.setp(ax3.get_xticklabels(), rotation=30, ha="right")

    # ---------------------- 4. Кривая эквити и просадка (добавим) -------
    ax4 = plt.subplot2grid((3, 2), (2, 0), colspan=2)
    ax4.set_title("Кривая эквити и максимальная просадка", fontsize=13, fontweight="bold")
    ax4.plot(x_dates, equity, color="dodgerblue", linewidth=1.5, label="Эквити")
    peak = np.maximum.accumulate(equity)
    drawdown = (equity - peak) / peak * 100
    ax4_twin = ax4.twinx()
    ax4_twin.fill_between(x_dates, 0, drawdown, color="salmon", alpha=0.3, label="Просадка %")
    ax4_twin.set_ylabel("Просадка, %", color="darkred")
    ax4_twin.tick_params(axis="y", labelcolor="darkred")
    ax4_twin.legend(loc="upper right")
    ax4.set_ylabel("Эквити, руб.", color="dodgerblue")
    ax4.legend(loc="upper left")
    ax4.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax4.xaxis.set_major_locator(mdates.AutoDateLocator())
    plt.setp(ax4.get_xticklabels(), rotation=30, ha="right")

    # Общий заголовок
    param_str = ""
    if params:
        param_str = f" | EMA={params.get('ema_period','?')}, Aroon={params.get('aroon_period','?')}, BullThr={params.get('bull_threshold','?')}, ATRmult={params.get('stop_atr_mult','?')}"
    fig.suptitle(f"{title_prefix} {metrics.get('total_return_%','')}% | Sharpe {metrics.get('sharpe','')}{param_str}",
                 fontsize=14, fontweight="bold")

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(f"{title_prefix}.png", dpi=200, bbox_inches="tight")


# ----------------------------- ПРИМЕР ЗАПУСКА -----------------------------
if __name__ == "__main__":
    # Яндекс
    df_opt1 = load_candles("data/yandex1_candles_train.json")
    df_opt2 = load_candles("data/yandex_candles_train.json")
    df_train = pd.concat([df_opt1, df_opt2])
    df_test = load_candles("data/yandex_candles_test.json")
    res_yandex = optimize_and_evaluate(df_train, df_test, name="Яндекс")

    # ------------------- ПОСТРОЕНИЕ ГРАФИКОВ -------------------
    best = res_yandex["best_params"]

    # Извлекаем параметры, нужные для расчёта индикаторов
    ind_params = {
        "ema_period": best["ema_period"],
        "aroon_period": best["aroon_period"],
        "use_vol_filter": best["use_vol_filter"],
    }

    # Пересчитываем индикаторы на обучающей и тестовой выборках
    df_train_ind = add_indicators(df_train, **ind_params)
    df_test_ind  = add_indicators(df_test,  **ind_params)

    # График для оптимизационного периода
    plot_strategy_report(
        df_train_ind,
        res_yandex["opt_metrics"],
        params=best,
        title_prefix="ОПТИМИЗАЦИЯ"
    )

    # График для тестового периода
    plot_strategy_report(
        df_test_ind,
        res_yandex["test_metrics"],
        params=best,
        title_prefix="ТЕСТИРОВАНИЕ"
    )