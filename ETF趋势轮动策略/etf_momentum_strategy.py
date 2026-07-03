"""
ETF动量轮动策略 - VnPy PortfolioStrategy版本
基于年化收益和判定系数打分的动量因子轮动，支持ATR动态回溯期
原策略来源：聚宽 - https://www.joinquant.com/post/62821
"""

from datetime import datetime, date
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from vnpy.trader.utility import ArrayManager
from vnpy.trader.object import TickData, BarData, TradeData
from vnpy.trader.constant import Direction

from vnpy_portfoliostrategy import StrategyTemplate, StrategyEngine
from vnpy_portfoliostrategy.utility import PortfolioBarGenerator


class ETFMomentumStrategy(StrategyTemplate):
    """
    ETF动量轮动策略
    
    核心逻辑：
    1. 每日9:50（或每日Bar推送时）对所有ETF进行动量打分
    2. 动量因子 = 年化收益率 × R²（判定系数）
    3. 支持ATR动态调整回溯期（20~60天）
    4. 风控：近3日跌幅超过5%则得分归零
    5. 全仓持有得分最高的1只ETF
    """

    author = "ETFMomentum"

    # ========== 策略参数 ==========
    
    # ---- ETF池子 ----
    # 注意：vt_symbol格式为 "代码.交易所"，如 "513100.SSE"
    etf_pool: List[str] = [
        "513100.SSE",   # 纳指ETF
        "513520.SSE",   # 日经ETF
        "513030.SSE",   # 德国ETF
        "518880.SSE",   # 黄金ETF
        "159980.SZSE",  # 有色ETF
        "501018.SSE",   # 南方原油
        "511090.SSE",   # 30年国债ETF
        "512890.SSE",   # 红利低波
        "159915.SZSE",  # 创业板100
    ]
    
    # ---- 动量参数 ----
    m_days: int = 25              # 默认动量参考天数（auto_day=False时使用）
    auto_day: bool = True         # 是否根据ATR动态调整回溯期
    min_days: int = 20            # 最小回溯期（高波动时）
    max_days: int = 60            # 最大回溯期（低波动时）
    annual_days: int = 250        # 年化交易日数
    
    # ---- 风控参数 ----
    drop_threshold: float = 0.95  # 跌幅阈值（5%）
    score_min: float = 0.0        # 得分下限
    score_max: float = 6.0        # 得分上限
    
    # ---- 仓位参数 ----
    target_num: int = 1           # 目标持仓ETF数量
    position_pct: float = 1.0     # 仓位比例（100%）
    price_add: float = 0.01       # 下单价格偏移
    
    # ---- 交易成本 ----
    commission_rate: float = 0.0002  # 佣金费率（双边万二）
    min_commission: float = 1.0      # 最低佣金
    
    # ---- 初始化资金 ----
    initial_capital: float = 1_000_000

    # ========== 策略变量 ==========
    scores: Dict[str, float] = {}           # 各ETF得分
    slopes: Dict[str, float] = {}           # 回归斜率
    rsquared: Dict[str, float] = {}         # R²值
    annual_returns: Dict[str, float] = {}   # 年化收益率
    lookback_days: Dict[str, int] = {}      # 各ETF实际使用的回溯期
    
    selected_assets: List[str] = []         # 选中的ETF列表
    current_target: str = ""                # 当前目标持仓
    last_trade_date: Optional[date] = None  # 上次交易日期
    
    current_capital: float = 0.0
    total_equity: float = 0.0

    parameters = [
        "etf_pool",
        "m_days",
        "auto_day",
        "min_days",
        "max_days",
        "annual_days",
        "drop_threshold",
        "score_min",
        "score_max",
        "target_num",
        "position_pct",
        "price_add",
        "commission_rate",
        "min_commission",
        "initial_capital",
    ]

    variables = [
        "scores",
        "selected_assets",
        "current_target",
        "total_equity",
    ]

    def __init__(
        self,
        strategy_engine: StrategyEngine,
        strategy_name: str,
        vt_symbols: List[str],
        setting: dict
    ) -> None:
        super().__init__(strategy_engine, strategy_name, vt_symbols, setting)
        
        # 如果传入的vt_symbols为空，使用默认池子
        if not vt_symbols:
            vt_symbols = self.etf_pool
        
        # 初始化ArrayManager
        array_size = self.max_days + 50
        self.ams: Dict[str, ArrayManager] = {}
        for vt_symbol in vt_symbols:
            self.ams[vt_symbol] = ArrayManager(size=array_size)
        
        # 初始化数据结构
        self.scores = {}
        self.slopes = {}
        self.rsquared = {}
        self.annual_returns = {}
        self.lookback_days = {}
        self.selected_assets = []
        self.current_target = ""
        self.last_trade_date = None
        
        self.current_capital = float(self.initial_capital)
        self.total_equity = float(self.initial_capital)
        
        # PortfolioBarGenerator
        self.pbg = PortfolioBarGenerator(self.on_bars)
        
        self.write_log(f"ETF动量轮动策略初始化完成")
        self.write_log(f"ETF数量: {len(vt_symbols)}")
        self.write_log(f"目标持仓数: {self.target_num}")
        self.write_log(f"动态回溯期: {'启用' if self.auto_day else '禁用 (固定' + str(self.m_days) + '天)'}")
        if self.auto_day:
            self.write_log(f"  回溯期范围: {self.min_days} ~ {self.max_days}天")

    def on_init(self) -> None:
        """策略初始化"""
        self.write_log(f"策略初始化，初始资金: {self.initial_capital:,.0f}")
        self.load_bars(self.max_days + 50)

    def on_start(self) -> None:
        """策略启动"""
        self.write_log("策略启动")

    def on_stop(self) -> None:
        """策略停止"""
        self.write_log("策略停止")

    def on_tick(self, tick: TickData) -> None:
        """Tick推送"""
        self.pbg.update_tick(tick)

    def update_trade(self, trade: TradeData) -> None:
        """交易更新"""
        super().update_trade(trade)
        
        size = self.get_size(trade.vt_symbol)
        trade_value = trade.price * trade.volume * size
        commission = abs(trade_value) * self.commission_rate
        commission = max(commission, self.min_commission)
        
        if trade.direction == Direction.LONG:
            self.current_capital -= trade_value
        else:
            self.current_capital += trade_value
        
        self.current_capital -= commission

    def calculate_price(self, vt_symbol: str, direction: Direction, reference: float) -> float:
        """计算下单价格"""
        if direction == Direction.LONG:
            return reference + self.price_add
        else:
            return max(reference - self.price_add, 0.01)

    # ========== 核心打分逻辑 ==========

    def calculate_dynamic_lookback(self, am: ArrayManager) -> int:
        """
        基于ATR动态计算回溯期
        
        公式: lookback = min_days + (max_days - min_days) × (1 - min(0.9, short_atr/long_atr))
        当近期波动率高于长期波动率时，回溯期缩短（更敏感）
        当近期波动率低于长期波动率时，回溯期延长（更平滑）
        """
        if not am.inited or len(am.close) < self.max_days + 10:
            return self.m_days
        
        # 计算ATR
        high = np.asarray(am.high[-self.max_days-10:], dtype=float)
        low = np.asarray(am.low[-self.max_days-10:], dtype=float)
        close = np.asarray(am.close[-self.max_days-10:], dtype=float)
        
        if len(high) < self.max_days:
            return self.m_days
        
        # 计算长周期ATR (60天) 和 短周期ATR (20天)
        long_atr = self._calculate_atr(high, low, close, self.max_days)
        short_atr = self._calculate_atr(high, low, close, self.min_days)
        
        if long_atr <= 0:
            return self.m_days
        
        ratio = min(0.9, short_atr / long_atr)
        lookback = int(self.min_days + (self.max_days - self.min_days) * (1 - ratio))
        
        return max(self.min_days, min(self.max_days, lookback))

    def _calculate_atr(self, high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int) -> float:
        """计算ATR（平均真实波幅）"""
        if len(high) < period + 1:
            return 0.0
        
        # 真实波幅
        high_shift = high[1:]
        low_shift = low[1:]
        close_shift = close[:-1]
        
        tr1 = high_shift - low_shift
        tr2 = np.abs(high_shift - close_shift)
        tr3 = np.abs(low_shift - close_shift)
        
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        
        if len(tr) < period:
            return 0.0
        
        # 取最近period天的平均值
        return float(np.mean(tr[-period:]))

    def calculate_momentum_score(self, am: ArrayManager, lookback: int) -> Tuple[float, float, float, float]:
        """
        计算动量得分
        
        得分 = 年化收益率 × R²
        
        年化收益率：对log(价格)做加权线性回归，斜率年化
        R²：判定系数，衡量趋势的线性强度
        
        返回: (得分, 斜率, 年化收益率, R²)
        """
        if not am.inited or len(am.close) < lookback:
            return 0.0, 0.0, 0.0, 0.0
        
        closes = np.asarray(am.close[-lookback:], dtype=float)
        
        if len(closes) < lookback:
            return 0.0, 0.0, 0.0, 0.0
        
        # 取对数
        log_prices = np.log(closes)
        
        # 加权：近期权重更高 (1.0 -> 2.0)
        x = np.arange(len(log_prices)).reshape(-1, 1)
        y = log_prices.reshape(-1, 1)
        weights = np.linspace(1, 2, len(log_prices))
        
        # 加权线性回归
        model = LinearRegression()
        model.fit(x, y, sample_weight=weights)
        
        slope = model.coef_[0][0]
        intercept = model.intercept_[0]
        
        # 年化收益率
        annual_return = np.exp(slope * self.annual_days) - 1
        
        # 计算R²
        y_pred = model.predict(x).flatten()
        ss_res = np.sum(weights * (y.flatten() - y_pred) ** 2)
        ss_tot = np.sum(weights * (y.flatten() - np.mean(y.flatten())) ** 2)
        
        if ss_tot > 0:
            r2 = 1 - ss_res / ss_tot
        else:
            r2 = 0.0
        
        # 得分 = 年化收益率 × R²
        score = annual_return * r2
        
        return score, slope, annual_return, r2

    def check_drop_risk(self, am: ArrayManager, lookback: int) -> bool:
        """
        检查近3日跌幅风险
        
        条件1: 近3日任一天跌幅超过5%
        条件2: 连续3日下跌且累计跌幅超过5%
        条件3: 近4日连续下跌且累计跌幅超过5%（用于处理数据不足的情况）
        """
        if not am.inited or len(am.close) < 6:
            return True  # 数据不足，不触发风控
        
        closes = np.asarray(am.close[-6:], dtype=float)
        
        # 计算每日收益率（前5天的数据）
        # returns[0] = closes[1]/closes[0] - 1, 表示第1个交易日到第2个交易日的收益率
        returns = closes[1:] / closes[:-1] - 1
        
        # 条件1: 近3日（即最近3个交易日）任一天跌幅超过5%
        # 对应 returns[-3], returns[-2], returns[-1]
        if len(returns) >= 3:
            recent_3_returns = returns[-3:]
            if np.min(recent_3_returns) < self.drop_threshold - 1:  # 注意：收益率是比例，阈值为0.95表示跌幅-5%，所以收益率 < -0.05
                return False  # 触发风控，得分归零
        
        # 条件2: 连续3日下跌且累计跌幅超过5%
        # 判断条件：最近3日（closes[-4] -> closes[-1]）连续下跌且总跌幅>5%
        if len(closes) >= 4:
            # 检查是否连续3日下跌
            is_consecutive_down = (closes[-1] < closes[-2] and 
                                  closes[-2] < closes[-3] and 
                                  closes[-3] < closes[-4])
            if is_consecutive_down:
                # 累计跌幅超过5%
                total_drop = closes[-1] / closes[-4] - 1
                if total_drop < self.drop_threshold - 1:  # < -0.05
                    return False
        
        # 条件3: 近4日连续下跌且累计跌幅超过5%（用于处理数据不足的情况）
        # 判断条件：closes[-5] -> closes[-2] 连续4日下跌且累计跌幅>5%
        if len(closes) >= 5:
            is_consecutive_down = (closes[-2] < closes[-3] and 
                                  closes[-3] < closes[-4] and 
                                  closes[-4] < closes[-5])
            if is_consecutive_down:
                total_drop = closes[-2] / closes[-5] - 1
                if total_drop < self.drop_threshold - 1:
                    return False
        
        return True

    def step_rank_assets(self) -> List[str]:
        """
        对所有ETF进行打分排序
        
        返回: 按得分降序排列的ETF列表
        """
        scores = {}
        slopes = {}
        rsquared = {}
        annual_returns = {}
        lookback_days = {}
        
        for vt_symbol, am in self.ams.items():
            if not am.inited:
                scores[vt_symbol] = 0.0
                continue
            
            # 确定回溯期
            if self.auto_day:
                lookback = self.calculate_dynamic_lookback(am)
            else:
                lookback = self.m_days
            
            lookback_days[vt_symbol] = lookback
            
            # 计算动量得分
            score, slope, annual_ret, r2 = self.calculate_momentum_score(am, lookback)
            
            # 风控：近3日跌幅超过5% → 得分归零
            if not self.check_drop_risk(am, lookback):
                score = 0.0
                self.write_log(f"风控触发: {vt_symbol} 近3日跌幅超过5%，得分归零")
            
            scores[vt_symbol] = score
            slopes[vt_symbol] = slope
            rsquared[vt_symbol] = r2
            annual_returns[vt_symbol] = annual_ret
        
        # 保存变量
        self.scores = scores
        self.slopes = slopes
        self.rsquared = rsquared
        self.annual_returns = annual_returns
        self.lookback_days = lookback_days
        
        # 过滤：0 < score < 6
        qualified = []
        for vt_symbol, score in scores.items():
            if self.score_min < score < self.score_max:
                qualified.append((vt_symbol, score))
        
        # 按得分降序排列
        sorted_assets = sorted(qualified, key=lambda x: x[1], reverse=True)
        
        return [asset for asset, score in sorted_assets]

    # ========== 仓位管理 ==========

    def update_total_equity(self, bars: Dict[str, BarData]) -> float:
        """更新总权益"""
        cash = self.current_capital
        position_value = 0.0
        
        for vt_symbol, bar in bars.items():
            pos = self.get_pos(vt_symbol)
            if pos != 0:
                size = self.get_size(vt_symbol)
                position_value += pos * bar.close_price * size
        
        self.total_equity = cash + position_value
        
        if self.total_equity <= 0:
            self.total_equity = 1.0
        
        return self.total_equity

    def calculate_target_positions(
        self,
        total_equity: float,
        target_symbols: List[str],
        bars: Dict[str, BarData]
    ) -> Dict[str, int]:
        """
        计算目标仓位
        
        全仓买入得分最高的ETF（数量 = target_num）
        """
        targets = {}
        
        if not target_symbols or total_equity <= 0:
            return targets
        
        # 取前target_num个
        selected = target_symbols[:self.target_num]
        
        # 计算每个ETF的目标权重
        weight_per_asset = self.position_pct / len(selected)
        
        for vt_symbol in selected:
            bar = bars.get(vt_symbol)
            if not bar:
                continue
            
            price = bar.close_price
            size = self.get_size(vt_symbol)
            
            if price <= 0 or size <= 0:
                continue
            
            target_value = total_equity * weight_per_asset
            target_volume = int(target_value / (price * size))
            
            if target_volume > 0:
                targets[vt_symbol] = target_volume
        
        return targets

    # ========== 主逻辑 ==========

    def on_bars(self, bars: Dict[str, BarData]) -> None:
        """Bar推送 - 策略核心"""
        if not bars:
            return
        
        current_bar = list(bars.values())[0]
        current_date = current_bar.datetime.date()
        
        # 更新ArrayManager
        for vt_symbol, bar in bars.items():
            am = self.ams.get(vt_symbol)
            if am:
                am.update_bar(bar)
        
        # 检查数据是否就绪
        if not all(am.inited for am in self.ams.values()):
            # 数据未就绪，仍需调用rebalance_portfolio以更新状态
            self.rebalance_portfolio(bars)
            return
        
        # 更新总权益
        total_equity = self.update_total_equity(bars)
        
        # 打分排序
        ranked_assets = self.step_rank_assets()
        self.selected_assets = ranked_assets
        
        if ranked_assets:
            best_asset = ranked_assets[0]
            best_score = self.scores.get(best_asset, 0)
            
            # 日志输出（每5天记录一次）
            if not self.last_trade_date or (current_date - self.last_trade_date).days % 5 == 0:
                self.write_log(
                    f"【打分排名】Top1: {best_asset} "
                    f"得分: {best_score:.4f} | "
                    f"年化收益: {self.annual_returns.get(best_asset, 0):.2%} | "
                    f"R²: {self.rsquared.get(best_asset, 0):.4f} | "
                    f"回溯期: {self.lookback_days.get(best_asset, 0)}天"
                )
        else:
            self.write_log(f"【打分排名】无ETF满足条件 (得分需在 {self.score_min} ~ {self.score_max} 之间)")
        
        # 计算目标仓位
        target_positions = self.calculate_target_positions(total_equity, ranked_assets, bars)
        
        # 设置目标仓位（先设置所有标的的目标仓位）
        for vt_symbol in self.vt_symbols:
            target_vol = target_positions.get(vt_symbol, 0)
            self.set_target(vt_symbol, target_vol)
        
        # 记录调仓日志（仅当有实际变化时）
        need_trade = False
        trade_log = []
        
        for vt_symbol in self.vt_symbols:
            current_pos = self.get_pos(vt_symbol)
            target_vol = target_positions.get(vt_symbol, 0)
            
            if current_pos != target_vol:
                need_trade = True
                if target_vol == 0:
                    trade_log.append(f"卖出 {vt_symbol} ({current_pos}手)")
                elif current_pos == 0:
                    trade_log.append(f"买入 {vt_symbol} ({target_vol}手)")
                else:
                    trade_log.append(f"调整 {vt_symbol}: {current_pos}手 -> {target_vol}手")
        
        # 每天调用rebalance_portfolio执行调仓
        self.rebalance_portfolio(bars)
        
        # 如果有实际调仓，记录日志
        if need_trade:
            self.last_trade_date = current_date
            
            # 更新当前目标
            if ranked_assets:
                self.current_target = ranked_assets[0]
            
            self.write_log(
                f"【调仓执行】{', '.join(trade_log[:3])}"
                f"{' ... 共' + str(len(trade_log)) + '笔' if len(trade_log) > 3 else ''}"
            )
            self.write_log(
                f"总权益: {total_equity:,.0f} | "
                f"持仓数: {len(target_positions)}个ETF | "
                f"日期: {current_date}"
            )
        
        self.put_event()