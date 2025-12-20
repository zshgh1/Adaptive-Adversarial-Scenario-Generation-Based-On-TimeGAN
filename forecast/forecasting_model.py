from typing import Dict, Any, Optional, Tuple, List
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
import random
import logging
import sys

import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
from abc import ABC, abstractmethod
import os
import json
from datetime import datetime
import joblib

import torch.nn.functional as F

import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import pmdarima as pm
from pmdarima import auto_arima


from prophet import Prophet
from prophet.plot import plot_plotly, plot_components_plotly
import plotly.graph_objects as go
from abc import ABC
import logging
from datetime import datetime, timedelta


# ===================== 日志配置 =====================
def setup_logging():
    """配置日志系统：同时输出到控制台和文件"""
    # 创建日志目录
    log_dir = "./logs"
    os.makedirs(log_dir, exist_ok=True)

    # 日志文件名（带时间戳）
    log_filename = os.path.join(log_dir, f"forecasting_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

    # 日志格式
    log_format = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # 根日志器
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)

    # 控制台处理器
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(log_format)
    console_handler.setLevel(logging.INFO)

    # 文件处理器
    file_handler = logging.FileHandler(log_filename, encoding="utf-8")
    file_handler.setFormatter(log_format)
    file_handler.setLevel(logging.INFO)

    # 添加处理器
    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)

    return root_logger


# 初始化日志
logger = setup_logging()


class EnhancedLSTM(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.input_size = 1
        self.hidden_size = args.lstm_hidden_size
        self.num_layers = args.lstm_num_layers
        self.dropout = args.lstm_dropout
        self.horizon = args.horizon
        self.sequence_length = args.sequence_length  # 新增：记录序列长度
        logger.info(
            f"初始化轨迹衔接版EnhancedLSTM - hidden_size:{self.hidden_size}, num_layers:{self.num_layers}, horizon:{self.horizon}")

        # 1. LSTM层（保留）
        self.lstm = nn.LSTM(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout if self.num_layers > 1 else 0,
            batch_first=True,
            bidirectional=False
        )
        self.lstm_norm = nn.LayerNorm(self.hidden_size)

        # 2. 位置感知注意力（增强最近历史的权重）
        self.attention = nn.MultiheadAttention(
            embed_dim=self.hidden_size,
            num_heads=4,
            dropout=self.dropout,
            batch_first=True
        )
        # 注意力时间权重：给最近的时间步更高权重（解决趋势捕捉）
        self.attn_time_weight = nn.Parameter(torch.linspace(0.5, 1.0, self.sequence_length).unsqueeze(0).unsqueeze(-1))
        self.attn_pos_encoding = nn.Parameter(torch.randn(1, self.sequence_length, self.hidden_size))
        self.attention_norm = nn.LayerNorm(self.hidden_size)
        self.attention_dropout = nn.Dropout(self.dropout)

        # 3. 可学习初始状态（保留）
        self.h0 = nn.Parameter(torch.randn(self.num_layers, 1, self.hidden_size))
        self.c0 = nn.Parameter(torch.randn(self.num_layers, 1, self.hidden_size))

        # 4. 多尺度连续特征融合（替代离散3点，捕捉趋势）
        self.scale_fusion = nn.Sequential(
            # 融合：全局平均+全局最大+最后时序特征（连续趋势）
            nn.Linear(self.hidden_size * 3, self.hidden_size),
            nn.LeakyReLU(0.1),
            nn.LayerNorm(self.hidden_size),
            nn.Dropout(self.dropout)
        )

        # 5. 输出层（新增：历史衔接+时序平滑）
        self.fc1 = nn.Linear(self.hidden_size, self.hidden_size // 2)
        self.fc2 = nn.Linear(self.hidden_size // 2, self.horizon)
        self.dropout = nn.Dropout(self.dropout)
        self.act = nn.LeakyReLU(0.1)
        self.fc_norm = nn.BatchNorm1d(self.hidden_size // 2)
        # 时序平滑层（解决预测段跳变）
        self.traj_smooth = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=3, padding=1, padding_mode='replicate')

        self._init_weights()

    def _init_weights(self):
        # （保留原有初始化）
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('leaky_relu', 0.1))
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.01)
            elif isinstance(m, nn.LayerNorm) or isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Conv1d):
                nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def forward(self, x):
        batch_size = x.size(0)
        seq_len = x.size(1)
        # ========== 新增：获取历史最后一个值（用于衔接） ==========
        historical_last_val = x[:, -1, 0]  # 历史轨迹的最后一个值 [batch]

        # 1. 可学习初始状态
        h0 = self.h0.repeat(1, batch_size, 1).to(x.device)
        c0 = self.c0.repeat(1, batch_size, 1).to(x.device)

        # 2. LSTM前向
        lstm_out, (hn, cn) = self.lstm(x, (h0, c0))
        lstm_out = self.lstm_norm(lstm_out)

        # 3. 注意力（增强最近历史的权重）
        attn_input = lstm_out * self.attn_time_weight[:, :seq_len, :].to(x.device)  # 最近时间步权重更高
        attn_input = attn_input + self.attn_pos_encoding[:, :seq_len, :].to(x.device)
        attn_out, _ = self.attention(attn_input, attn_input, attn_input)
        attn_out = self.attention_dropout(attn_out)
        attn_out = attn_out + lstm_out  # 残差
        attn_out = self.attention_norm(attn_out)

        # 4. 多尺度连续特征融合（替代离散3点）
        avg_feat = torch.mean(attn_out, dim=1)  # 全局平均特征（整体趋势）
        max_feat = torch.max(attn_out, dim=1)[0]  # 全局最大特征（峰值趋势）
        last_feat = attn_out[:, -1, :]  # 最后时序特征（最近趋势）
        fused_feat = torch.cat([avg_feat, max_feat, last_feat], dim=-1)
        fused_feat = self.scale_fusion(fused_feat)

        # 5. 输出层（新增：历史衔接+平滑）
        out = self.act(self.fc1(fused_feat))
        out = self.fc_norm(out)
        out = self.dropout(out)
        out = self.fc2(out)  # [batch, horizon]

        # ========== 新增1：历史衔接（解决不连续） ==========
        out = out + historical_last_val.unsqueeze(1)  # 预测值 = 增量 + 历史最后一个值

        # ========== 新增2：时序平滑（解决预测段跳变） ==========
        out = out.unsqueeze(1)  # [batch, 1, horizon]
        out = self.traj_smooth(out)
        out = out.squeeze(1)  # [batch, horizon]

        return out


class TimeSeriesTransformer(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.input_dim = 1
        self.d_model = args.transformer_d_model
        self.nhead = args.transformer_nhead
        self.num_encoder_layers = args.transformer_num_layers  # 编码器层数
        self.num_decoder_layers = args.transformer_num_layers  # 解码器层数（与编码器一致）
        self.dropout = args.transformer_dropout
        self.sequence_length = args.sequence_length
        self.horizon = args.horizon  # 预测步长（Decoder输出长度）
        self.local_window = 5  # 聚焦最近历史窗口
        logger.info(
            f"初始化完整Transformer(Encoder-Decoder) - d_model:{self.d_model}, "
            f"encoder_layers:{self.num_encoder_layers}, decoder_layers:{self.num_decoder_layers}, "
            f"horizon:{self.horizon}")

        # ==================== 1. 共享投影层 ====================
        # Encoder输入投影（历史轨迹→d_model）
        self.encoder_input_proj = nn.Sequential(
            nn.Linear(self.input_dim, self.d_model),
            nn.LayerNorm(self.d_model),
            nn.LeakyReLU(0.1),
            nn.Dropout(self.dropout)
        )
        # Decoder输入投影（构造的初始序列→d_model）
        self.decoder_input_proj = nn.Sequential(
            nn.Linear(self.input_dim, self.d_model),
            nn.LayerNorm(self.d_model),
            nn.LeakyReLU(0.1),
            nn.Dropout(self.dropout)
        )

        # ==================== 2. 位置编码 ====================
        # Encoder位置编码（适配历史序列长度）
        self.encoder_pos_encoding = nn.Parameter(torch.randn(1, self.sequence_length, self.d_model))
        # Decoder位置编码（适配预测步长horizon）
        self.decoder_pos_encoding = nn.Parameter(torch.randn(1, self.horizon, self.d_model))
        # 时间权重：Encoder给最近历史更高权重
        self.encoder_time_weight = nn.Parameter(torch.linspace(0.5, 1.0, self.sequence_length).unsqueeze(0).unsqueeze(-1))
        self.pos_dropout = nn.Dropout(self.dropout)

        # ==================== 3. Transformer核心（Encoder-Decoder） ====================
        # 编码器层
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=self.nhead,
            dim_feedforward=self.d_model * 4,
            dropout=self.dropout,
            activation='relu',
            batch_first=True,
            norm_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=self.num_encoder_layers)

        # 解码器层
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=self.d_model,
            nhead=self.nhead,
            dim_feedforward=self.d_model * 4,
            dropout=self.dropout,
            activation='relu',
            batch_first=True,
            norm_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=self.num_decoder_layers)

        # ==================== 4. 特征聚合与输出层 ====================
        # 输出投影（d_model→1，再拼接成horizon长度）
        self.output_proj = nn.Sequential(
            nn.Linear(self.d_model, self.d_model // 2),
            nn.LayerNorm(self.d_model // 2),
            nn.LeakyReLU(0.1),
            nn.Dropout(self.dropout),
            nn.Linear(self.d_model // 2, 1)
        )
        # 时序平滑层（轨迹专用）
        self.traj_smooth = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=3, padding=1, padding_mode='replicate')

        # 权重初始化
        self._init_weights()

    def _init_weights(self):
        """轨迹适配的权重初始化"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                gain = nn.init.calculate_gain('relu')
                nn.init.xavier_uniform_(m.weight, gain=gain)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Conv1d):
                nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def _generate_decoder_input(self, batch_size, historical_last_val, device):
        """
        构造Decoder的输入序列（无外部目标输入，保持输入仅为历史轨迹）
        :param batch_size: 批次大小
        :param historical_last_val: 历史轨迹最后一个值 [batch]
        :param device: 设备
        :return: decoder_input: [batch, horizon, 1]（历史最后值扩展为horizon长度）
        """
        # 用历史最后一个值构造Decoder初始输入（保证轨迹衔接）
        decoder_input = historical_last_val.unsqueeze(1).repeat(1, self.horizon)  # [batch, horizon]
        decoder_input = decoder_input.unsqueeze(-1)  # [batch, horizon, 1]
        return decoder_input.to(device)

    def _generate_decoder_mask(self, seq_len, device):
        """
        生成Decoder自注意力掩码（下三角掩码，防止看到未来步）
        :param seq_len: Decoder序列长度（horizon）
        :param device: 设备
        :return: mask: [seq_len, seq_len]
        """
        mask = (torch.triu(torch.ones(seq_len, seq_len, device=device)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, x):
        """
        前向传播：输入为历史轨迹 [batch, seq_len, 1]，输出为预测轨迹 [batch, horizon]
        """
        batch_size = x.size(0)
        seq_len = x.size(1)
        device = x.device

        # ========== Step 1: 提取历史最后值（用于Decoder输入+轨迹衔接） ==========
        historical_last_val = x[:, -1, 0]  # [batch]

        # ========== Step 2: Encoder处理历史轨迹 ==========
        # Encoder输入投影
        encoder_input = self.encoder_input_proj(x)  # [batch, seq_len, d_model]
        # 位置编码 + 时间权重（最近历史更高权重）
        encoder_input = encoder_input * self.encoder_time_weight[:, :seq_len, :].to(device)
        encoder_input = encoder_input + self.encoder_pos_encoding[:, :seq_len, :].to(device)
        encoder_input = self.pos_dropout(encoder_input)
        # Encoder前向
        encoder_output = self.encoder(encoder_input)  # [batch, seq_len, d_model]

        # ========== Step 3: 构造Decoder输入 + 位置编码 ==========
        # 生成Decoder输入序列（无外部目标）
        decoder_input = self._generate_decoder_input(batch_size, historical_last_val, device)  # [batch, horizon, 1]
        # Decoder输入投影
        decoder_input = self.decoder_input_proj(decoder_input)  # [batch, horizon, d_model]
        # Decoder位置编码
        decoder_input = decoder_input + self.decoder_pos_encoding[:, :self.horizon, :].to(device)
        decoder_input = self.pos_dropout(decoder_input)

        # ========== Step 4: Decoder前向（自回归生成预测） ==========
        # 生成Decoder自注意力掩码（防止看未来）
        tgt_mask = self._generate_decoder_mask(self.horizon, device)
        # Decoder前向（结合Encoder输出做交叉注意力）
        decoder_output = self.decoder(
            tgt=decoder_input,          # Decoder输入 [batch, horizon, d_model]
            memory=encoder_output,      # Encoder输出 [batch, seq_len, d_model]
            tgt_mask=tgt_mask           # Decoder自注意力掩码
        )  # [batch, horizon, d_model]

        # ========== Step 5: 输出投影 + 轨迹平滑 ==========
        # 投影到1维（每个预测步）
        output = self.output_proj(decoder_output)  # [batch, horizon, 1]
        output = output.squeeze(-1)  # [batch, horizon]

        # 轨迹平滑（解决跳变，保持连续性）
        output = output.unsqueeze(1)  # [batch, 1, horizon]
        output = self.traj_smooth(output)
        output = output.squeeze(1)  # [batch, horizon]

        return output


# ========== 新增：ARIMA模型类（适配现有代码） ==========
class ARIMAModel(ABC):
    """Enhanced ARIMA model with automatic parameter selection（修正版）."""

    def __init__(self, args):
        """Initialize ARIMA model.

        Args:
            args: 全局参数对象
        """
        super().__init__()
        self.args = args  # 关联全局参数

        # 关键修复1：参数兜底（避免args缺失属性报错）
        self.order = getattr(args, 'order', (5, 1, 0))
        self.auto_arima = getattr(args, 'auto_arima', True)
        self.max_p = getattr(args, 'max_p', 5)
        self.max_d = getattr(args, 'max_d', 2)
        self.max_q = getattr(args, 'max_q', 5)
        self.seasonal = getattr(args, 'seasonal', False)
        self.m = getattr(args, 'm', 12)  # 季节性周期

        # 模型状态
        self.model = None
        self.residuals = None
        self.aic = None
        self.bic = None
        self.fitted_model = None
        self.scaler = None  # 绑定scaler，用于逆变换

    def set_scaler(self, scaler):
        """绑定数据缩放器（和LSTM/Transformer对齐）"""
        self.scaler = scaler
        logger.info("ARIMA模型已绑定MinMaxScaler")

    def _check_stationarity(self, data: pd.Series) -> Tuple[bool, float]:
        """补充：平稳性检验（ARIMA核心前置步骤）"""
        try:
            result = adfuller(data.dropna())
            p_value = result[1]
            is_stationary = p_value < 0.05
            logger.info(f"ARIMA平稳性检验 - p值:{p_value:.4f}, 平稳性:{is_stationary}")
            return is_stationary, p_value
        except Exception as e:
            logger.warning(f"平稳性检验失败: {e}，默认判定为非平稳")
            return False, 0.99

    def find_optimal_order(self, data: pd.Series) -> Tuple[int, int, int]:
        """Find optimal ARIMA order using auto_arima（增加异常处理）"""
        try:
            # 先做平稳性检验，辅助选参
            self._check_stationarity(data)

            auto_model = auto_arima(
                data,
                max_p=self.max_p,
                max_d=self.max_d,
                max_q=self.max_q,
                seasonal=self.seasonal,
                m=self.m,
                suppress_warnings=True,
                error_action='ignore',
                stepwise=True
            )

            order = auto_model.order
            seasonal_order = auto_model.seasonal_order if self.seasonal else None

            logger.info(f"Auto ARIMA找到最优阶数: {order}")
            if seasonal_order:
                logger.info(f"季节性阶数: {seasonal_order}")

            return order

        except Exception as e:
            logger.warning(f"Auto ARIMA选参失败: {e}，使用默认阶数 {self.order}")
            return self.order

    def predict(
            self,
            horizon: int,
            sequence: np.ndarray,  # 缩放后的历史序列
            confidence_intervals: bool = True
    ) -> Dict[str, np.ndarray]:
        """
        修正：返回字典格式（和LSTM/Transformer兼容），处理scaler逆变换
        """
        try:
            # 1. 数据预处理（剔除空值，避免拟合失败）
            y = pd.Series(sequence).dropna()
            if len(y) < 3:  # 最小序列长度校验
                logger.error(f"ARIMA输入序列过短（长度{len(y)}），返回全0预测")
                forecast = np.zeros(horizon)
                lower_bound = forecast - 0.1
                upper_bound = forecast + 0.1
                return {"forecast": forecast, "lower_bound": lower_bound, "upper_bound": upper_bound}

            # 2. 选择ARIMA阶数
            if self.auto_arima:
                order = self.find_optimal_order(y)
            else:
                order = self.order

            # 3. 拟合模型（核心修复：移除disp=False参数）
            if self.seasonal:
                self.model = sm.tsa.statespace.SARIMAX(
                    y,
                    order=order,
                    seasonal_order=(1, 1, 1, self.m),
                    enforce_stationarity=False,
                    enforce_invertibility=False
                )
                # SARIMAX支持verbose参数
                self.fitted_model = self.model.fit(verbose=False)
            else:
                # ARIMA无verbose/disp参数，直接fit
                self.model = ARIMA(y, order=order)
                self.fitted_model = self.model.fit()  # 移除disp=False

            self.residuals = self.fitted_model.resid
            self.aic = self.fitted_model.aic
            self.bic = self.fitted_model.bic

            # 4. 生成预测（后续代码不变）
            forecast_result = self.fitted_model.get_forecast(steps=horizon)
            forecast = forecast_result.predicted_mean.values

            # 5. 逆变换（恢复原始尺度）
            if self.scaler is not None:
                forecast = self.scaler.inverse_transform(forecast.reshape(-1, 1)).flatten()

            # 6. 处理置信区间
            lower_bound = None
            upper_bound = None
            if confidence_intervals:
                conf_int = forecast_result.conf_int()
                lower_bound = conf_int.iloc[:, 0].values
                upper_bound = conf_int.iloc[:, 1].values
                # 置信区间也需逆变换
                if self.scaler is not None:
                    lower_bound = self.scaler.inverse_transform(lower_bound.reshape(-1, 1)).flatten()
                    upper_bound = self.scaler.inverse_transform(upper_bound.reshape(-1, 1)).flatten()

            # 7. 返回字典（和LSTM/Transformer完全兼容）
            return {
                "forecast": forecast,
                "lower_bound": lower_bound,
                "upper_bound": upper_bound
            }

        except Exception as e:
            logger.error(f"ARIMA预测失败: {e}", exc_info=True)
            # 兜底返回：全0预测 + 空置信区间
            forecast = np.zeros(horizon)
            return {
                "forecast": forecast,
                "lower_bound": None,
                "upper_bound": None
            }


class ProphetModel(ABC):
    """Enhanced Prophet model (适配现有集成体系版本)."""

    def __init__(self, args):
        """初始化Prophet模型（对齐ARIMA接口，接收args参数）"""
        super().__init__()
        self.args = args

        # 从args读取配置（无则用默认值）
        self.yearly_seasonality = getattr(args, 'prophet_yearly_seasonality', True)
        self.weekly_seasonality = getattr(args, 'prophet_weekly_seasonality', True)
        self.daily_seasonality = getattr(args, 'prophet_daily_seasonality', False)
        self.seasonality_mode = getattr(args, 'prophet_seasonality_mode', 'additive')
        self.changepoint_prior_scale = getattr(args, 'prophet_changepoint_prior_scale', 0.05)
        self.interval_width = getattr(args, 'prophet_interval_width', 0.8)

        # 模型状态
        self.model = None
        self.fitted_model = None
        self.forecast_df = None
        self.scaler = None  # 绑定缩放器（和其他模型对齐）
        self.is_fitted = False  # 拟合状态标记

    def set_scaler(self, scaler):
        """绑定数据缩放器（逆变换用）"""
        self.scaler = scaler
        logger.info("Prophet模型已绑定MinMaxScaler")

    def _generate_dummy_ds(self, sequence: np.ndarray) -> pd.DataFrame:
        """
        核心适配：从sequence生成Prophet需要的ds（时间列）+ y（值列）DataFrame
        （原有输入是无时间的sequence，生成虚拟时间序列）
        """
        # 生成虚拟时间（从当前时间往前推len(sequence)个小时）
        end_date = datetime.now()
        start_date = end_date - timedelta(hours=len(sequence) - 1)
        dates = pd.date_range(start=start_date, end=end_date, freq='H')

        # 构造Prophet要求的DataFrame（ds=时间，y=序列值）
        prophet_data = pd.DataFrame({
            'ds': dates,
            'y': sequence  # sequence是缩放后的值，先不逆变换
        })
        return prophet_data

    def _add_regressors(self, data: pd.DataFrame) -> List[str]:
        """添加回归项（适配无历史时间的场景，简化逻辑）"""
        regressors = []

        # 趋势项
        data['trend_index'] = np.arange(len(data))
        regressors.append('trend_index')

        # 滞后项（填充空值）
        data['lag_1'] = data['y'].shift(1).fillna(method='bfill').fillna(method='ffill')
        regressors.append('lag_1')

        # 滚动统计（填充空值）
        data['rolling_mean_3'] = data['y'].rolling(window=3).mean().fillna(method='bfill').fillna(method='ffill')
        regressors.append('rolling_mean_3')

        logger.info(f"Prophet添加回归项: {regressors}")
        return regressors

    def fit(self, data: pd.DataFrame) -> 'ProphetModel':
        """拟合Prophet模型（适配内部调用）"""
        try:
            # 初始化Prophet
            self.model = Prophet(
                yearly_seasonality=self.yearly_seasonality,
                weekly_seasonality=self.weekly_seasonality,
                daily_seasonality=self.daily_seasonality,
                seasonality_mode=self.seasonality_mode,
                changepoint_prior_scale=self.changepoint_prior_scale,
                interval_width=self.interval_width
            )

            # 添加回归项
            regressors = self._add_regressors(data)
            for reg in regressors:
                self.model.add_regressor(reg)

            # 拟合模型
            self.fitted_model = self.model.fit(data)
            self.is_fitted = True
            logger.info("Prophet模型拟合完成")

        except Exception as e:
            logger.error(f"Prophet拟合失败: {e}", exc_info=True)
            raise
        return self

    def predict(
            self,
            horizon: int,
            sequence: np.ndarray,  # 适配现有输入：缩放后的历史序列
            confidence_intervals: bool = True
    ) -> Dict[str, np.ndarray]:
        """
        核心适配：对齐ARIMA的predict接口，返回统一格式的字典
        """
        logger.info(f"Prophet开始预测 - 步长:{horizon}, 输入序列长度:{len(sequence)}")

        try:
            # 1. 数据格式转换：sequence → Prophet所需的ds+y DataFrame
            prophet_data = self._generate_dummy_ds(sequence)

            # 2. 拟合模型（每条序列独立拟合，和ARIMA一致）
            self.fit(prophet_data)

            # 3. 生成未来时间框（包含horizon个预测步长）
            future = self.fitted_model.make_future_dataframe(periods=horizon, freq='H')

            # 4. 补充未来回归项（用最后一个值填充，简化逻辑）
            for col in ['trend_index', 'lag_1', 'rolling_mean_3']:
                if col in prophet_data.columns:
                    last_val = prophet_data[col].iloc[-1]
                    future[col] = last_val
                    # 趋势项递增（更合理）
                    if col == 'trend_index':
                        future.loc[len(prophet_data):, 'trend_index'] = np.arange(len(prophet_data),
                                                                                  len(prophet_data) + horizon)

            # 5. 预测
            self.forecast_df = self.fitted_model.predict(future)

            # 6. 提取预测值（仅取最后horizon个）
            forecast = self.forecast_df['yhat'].tail(horizon).values

            # 7. 逆变换（恢复原始尺度，和其他模型对齐）
            if self.scaler is not None:
                forecast = self.scaler.inverse_transform(forecast.reshape(-1, 1)).flatten()

            # 8. 处理置信区间
            lower_bound = None
            upper_bound = None
            if confidence_intervals:
                lower_bound = self.forecast_df['yhat_lower'].tail(horizon).values
                upper_bound = self.forecast_df['yhat_upper'].tail(horizon).values
                # 置信区间逆变换
                if self.scaler is not None:
                    lower_bound = self.scaler.inverse_transform(lower_bound.reshape(-1, 1)).flatten()
                    upper_bound = self.scaler.inverse_transform(upper_bound.reshape(-1, 1)).flatten()

            # 9. 返回统一格式（和ARIMA/LSTM一致）
            result = {
                'forecast': forecast,
                'lower_bound': lower_bound,
                'upper_bound': upper_bound
            }
            logger.info(f"Prophet预测完成 - 预测值范围:[{forecast.min():.2f}, {forecast.max():.2f}]")
            return result

        except Exception as e:
            logger.error(f"Prophet预测失败: {e}", exc_info=True)
            # 兜底返回（避免集成流程崩溃）
            forecast = np.zeros(horizon)
            return {
                'forecast': forecast,
                'lower_bound': None,
                'upper_bound': None
            }




class LossUtils:
    @staticmethod
    def calculate_trend(x):
        """计算序列的趋势（斜率）：相邻步的平均变化率"""
        x_change = x[:, 1:] - x[:, :-1]  # [batch, seq_len-1]
        trend = torch.mean(x_change, dim=1)  # [batch]：每个样本的平均趋势
        return trend

    @staticmethod
    def dynamic_weight(epoch, total_epochs, base_weight, target_weight, stage="early"):
        """动态调整损失权重：前期/后期权重渐变"""
        if stage == "early":
            # 前期：权重从base_weight降到target_weight
            return base_weight - (base_weight - target_weight) * (epoch / total_epochs)
        else:
            # 后期：权重从base_weight升到target_weight
            return base_weight + (target_weight - base_weight) * (epoch / total_epochs)

    @staticmethod
    def multi_step_weight(horizon, mode="increase"):
        """多步损失权重：increase（远步权重递增）/ decrease（递减）"""
        if mode == "increase":
            # 远步权重更高（轨迹长程预测更重要）：0.5→1.0
            return torch.linspace(0.5, 1.0, horizon)
        else:
            # 近步权重更高：1.0→0.5
            return torch.linspace(1.0, 0.5, horizon)

    @staticmethod
    def physical_constraint_loss(pred, min_val=0.0, max_val=100.0):
        """物理约束损失：惩罚超出合理范围的预测值"""
        # 低于最小值的惩罚
        lower_penalty = torch.mean(F.relu(min_val - pred))
        # 高于最大值的惩罚
        upper_penalty = torch.mean(F.relu(pred - max_val))
        return lower_penalty + upper_penalty

class EnsembleModel(nn.Module):
    """Ensemble model that combines multiple forecasting models."""

    def __init__(self, args):
        super().__init__()
        self.args = args

        self.sequence_length = args.sequence_length
        self.learning_rate = args.learning_rate
        self.batch_size = args.batch_size
        self.epochs = args.epochs
        self.filepath = args.filepath
        self.early_stopping_patience = args.early_stopping_patience
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.save_interval_ratio = getattr(self.args, 'save_interval_ratio', 1 / 3)
        self.loss_threshold =getattr(self.args, 'loss_threshold', 1e-6)
        self.teacher_forcing_ratio =getattr(self.args, 'teacher_forcing_ratio', 0.8)
        self.best_model_state = {}
        logger.info(
            f"初始化EnsembleModel - 设备:{self.device}, 批次大小:{self.batch_size}, 训练轮数:{self.epochs}, 早停耐心值:{self.early_stopping_patience}")

        # Initialize individual models
        self.models = {}
        self.model_predictions = {}
        self.meta_model = None
        self.is_meta_fitted = False

        if 'lstm' in self.args.enabled_models:
            self.models['lstm'] = EnhancedLSTM(args)
            self.models['lstm'].to(self.device)  # 移到指定设备
            self.lstm_optimizer = optim.Adam(self.models['lstm'].parameters(), lr=self.learning_rate)
            self.lstm_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.lstm_optimizer, mode='min', factor=0.5, patience=5, verbose=True
            )
            self.best_model_state['lstm']=None

        if 'transformer' in self.args.enabled_models:
            self.models['transformer'] = TimeSeriesTransformer(args)
            self.models['transformer'].to(self.device)  # 移到指定设备
            self.transformer_optimizer = optim.Adam(self.models['transformer'].parameters(), lr=self.learning_rate)
            self.transformer_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.transformer_optimizer, mode='min', factor=0.5, patience=5, verbose=True
            )
            self.best_model_state['transformer'] = None

        if 'arima' in self.args.enabled_models:
            self.models['arima'] = ARIMAModel(args)
        if 'prophet' in self.args.enabled_models:
            self.models['prophet'] = ProphetModel(args)


        self.criterion = nn.MSELoss()
        self.scaler = None

        self.training_results = {
            'lstm': {'train_losses': [], 'val_losses': []},
            'transformer': {'train_losses': [], 'val_losses': []}
        }

    def set_scaler(self, scaler):
        """从外部传入拟合好的MinMaxScaler（同步绑定到Prophet）"""
        self.scaler = scaler
        if 'arima' in self.models:
            self.models['arima'].set_scaler(scaler)
        if 'prophet' in self.models:  # 新增
            self.models['prophet'].set_scaler(scaler)

    def lstm_trainer(self, train_dataset, val_dataset):
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=True)
        logger.info(
            f"开始训练LSTM（直接多步输出） - 训练批次:{len(train_loader)}, 验证批次:{len(val_loader)}, horizon:{self.args.horizon}, 批次大小:{self.batch_size}")

        best_val_loss = float('inf')
        patience_counter = 0
        horizon = self.args.horizon

        for epoch in range(self.epochs):
            # ========== 训练阶段 ==========
            self.models['lstm'].train()
            train_loss = 0.0

            for batch_idx, (batch_X, batch_y) in enumerate(train_loader):
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                self.lstm_optimizer.zero_grad()

                # 核心修改：直接前向得到horizon个预测值
                multi_step_preds = self.models['lstm'](batch_X)  # [batch, horizon]



                # 多步损失计算（关注远步）
                loss_weights = torch.linspace(1.0, 0.5, horizon).to(self.device)  # 远步权重更高
                step_losses = self.criterion(multi_step_preds * loss_weights, batch_y * loss_weights)

                historical_last = batch_X[:, -1, 0]  # 历史最后一个值
                connection_loss = torch.mean((multi_step_preds[:, 0] - historical_last) ** 2)
                # 3. 平滑损失：预测段内相邻值的误差（解决变化单一）
                smooth_loss = torch.mean((multi_step_preds[:, 1:] - multi_step_preds[:, :-1]) ** 2)


                total_loss = step_losses+ 0.5 * connection_loss + 0.3 * smooth_loss

                # 反向传播
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.models['lstm'].parameters(), max_norm=1.0)
                self.lstm_optimizer.step()
                train_loss += total_loss.item()

                # # epoch内批量级保存
                # if batch_idx in save_batch_steps:
                #     self.save_model(self.filepath, 'lstm', epoch, batch_idx)

            # ========== 验证阶段 ==========
            self.models['lstm'].eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                    # 核心修改：直接前向得到horizon个预测值
                    multi_step_preds = self.models['lstm'](batch_X)  # [batch, horizon]

                    loss_weights = torch.linspace(0.5, 1.0, horizon).to(self.device)
                    step_losses = self.criterion(multi_step_preds * loss_weights, batch_y * loss_weights)
                    val_loss += step_losses.item()

            # 损失平均
            avg_train_loss = train_loss / len(train_loader)
            avg_val_loss = val_loss / len(val_loader)

            # 记录损失
            self.training_results['lstm']["val_losses"].append(avg_val_loss)
            self.training_results['lstm']["train_losses"].append(avg_train_loss)

            # 学习率调度
            self.lstm_scheduler.step(avg_val_loss)

            # 早停（带阈值）
            if avg_val_loss < best_val_loss - self.loss_threshold:
                best_val_loss = avg_val_loss
                patience_counter = 0
                self.best_model_state['lstm'] = self.models['lstm'].state_dict().copy()
                logger.debug(f"lstm Epoch {epoch + 1} - 验证损失更新至最佳:{best_val_loss:.6f}")
            else:
                patience_counter += 1
                logger.debug(
                    f"lstm Epoch {epoch + 1} - 验证损失未改善, 耐心值:{patience_counter}/{self.early_stopping_patience}")

            # 日志打印
            logger.info(
                f"lstm Epoch {epoch + 1}/{self.epochs} - 训练损失:{avg_train_loss:.6f}, 验证损失:{avg_val_loss:.6f}")
            self.save_model(self.filepath, 'lstm', epoch)
            # 早停触发时保存最佳模型
            if patience_counter >= self.early_stopping_patience:
                self.save_model(self.filepath, 'lstm', epoch, is_best=True)
                break


        if self.best_model_state['lstm'] is not None:
            self.models['lstm'].load_state_dict(self.best_model_state['lstm'])
            self.save_model(self.filepath, 'lstm', epoch="_final_best", is_best=True)  # 覆盖为最终最佳
        return self

    def transformer_trainer(self, train_dataset, val_dataset):
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)  # 验证集不shuffle
        logger.info(
            f"开始训练Transformer（轨迹优化版） - 训练批次:{len(train_loader)}, 验证批次:{len(val_loader)}, "
            f"horizon:{self.args.horizon}, 批次大小:{self.batch_size}")

        best_val_loss = float('inf')
        patience_counter = 0
        horizon = self.args.horizon
        loss_utils = LossUtils()
        traj_min = getattr(self.args, "traj_min", 0.0)
        traj_max = getattr(self.args, "traj_max", 100.0)

        for epoch in range(self.epochs):
            # ========== 训练阶段 ==========
            self.models['transformer'].train()
            train_loss = 0.0

            # Transformer对权重更敏感，动态权重调整幅度更小
            conn_weight = loss_utils.dynamic_weight(epoch, self.epochs, 0.6, 0.3, stage="early")
            smooth_weight = loss_utils.dynamic_weight(epoch, self.epochs, 0.4, 0.2, stage="early")
            trend_weight = loss_utils.dynamic_weight(epoch, self.epochs, 0.3, 0.7, stage="late")
            phys_weight = loss_utils.dynamic_weight(epoch, self.epochs, 0.2, 0.5, stage="late")

            for batch_idx, (batch_X, batch_y) in enumerate(train_loader):
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                self.transformer_optimizer.zero_grad()

                # 模型预测
                multi_step_preds = self.models['transformer'](batch_X)  # [batch, horizon]

                # 1. 混合MSE+MAE损失（抗噪）
                mse_loss = F.mse_loss(multi_step_preds, batch_y)
                mae_loss = F.l1_loss(multi_step_preds, batch_y)
                base_loss = 0.8 * mse_loss + 0.2 * mae_loss  # Transformer更依赖MSE

                # 2. 多步加权损失（远步权重递增）
                step_weights = loss_utils.multi_step_weight(horizon, mode="increase").to(self.device)
                weighted_loss = F.mse_loss(multi_step_preds * step_weights, batch_y * step_weights)
                base_loss = 0.9 * base_loss + 0.1 * weighted_loss

                # 3. 衔接损失
                historical_last = batch_X[:, -1, 0]
                connection_loss = F.mse_loss(multi_step_preds[:, 0], historical_last)

                # 4. 平滑损失
                smooth_loss = F.mse_loss(multi_step_preds[:, 1:], multi_step_preds[:, :-1])

                # 5. 历史趋势匹配损失
                history_trend = loss_utils.calculate_trend(batch_X.squeeze(-1))
                pred_trend = loss_utils.calculate_trend(multi_step_preds)
                trend_match_loss = F.mse_loss(pred_trend, history_trend)

                # 6. 物理约束损失
                phys_loss = loss_utils.physical_constraint_loss(multi_step_preds, traj_min, traj_max)

                # 7. 趋势变化率损失（原有，保留并融合）
                y_change = batch_y[:, 1:] - batch_y[:, :-1]
                pred_change = multi_step_preds[:, 1:] - multi_step_preds[:, :-1]
                trend_change_loss = F.mse_loss(pred_change, y_change)

                # 8. 自回归一致性损失
                ar_consist_loss = F.mse_loss(multi_step_preds[:, 1:], multi_step_preds[:, :-1] + y_change)

                # 总损失（动态权重+原有趋势变化率损失）
                total_loss = (
                        base_loss
                        + conn_weight * connection_loss
                        + smooth_weight * smooth_loss
                        + trend_weight * trend_match_loss
                        + phys_weight * phys_loss
                        + 0.5 * trend_change_loss  # 原有趋势损失
                        + 0.4 * ar_consist_loss  # 自回归一致性
                )

                # 反向传播（Transformer梯度更易爆炸，缩小梯度裁剪阈值）
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.models['transformer'].parameters(), max_norm=0.5)
                self.transformer_optimizer.step()
                train_loss += total_loss.item()

            # ========== 验证阶段（和训练损失完全一致） ==========
            self.models['transformer'].eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                    multi_step_preds = self.models['transformer'](batch_X)

                    # 复用训练阶段的损失逻辑
                    mse_loss = F.mse_loss(multi_step_preds, batch_y)
                    mae_loss = F.l1_loss(multi_step_preds, batch_y)
                    base_loss = 0.8 * mse_loss + 0.2 * mae_loss

                    step_weights = loss_utils.multi_step_weight(horizon, mode="increase").to(self.device)
                    weighted_loss = F.mse_loss(multi_step_preds * step_weights, batch_y * step_weights)
                    base_loss = 0.9 * base_loss + 0.1 * weighted_loss

                    historical_last = batch_X[:, -1, 0]
                    connection_loss = F.mse_loss(multi_step_preds[:, 0], historical_last)

                    smooth_loss = F.mse_loss(multi_step_preds[:, 1:], multi_step_preds[:, :-1])

                    history_trend = loss_utils.calculate_trend(batch_X.squeeze(-1))
                    pred_trend = loss_utils.calculate_trend(multi_step_preds)
                    trend_match_loss = F.mse_loss(pred_trend, history_trend)

                    phys_loss = loss_utils.physical_constraint_loss(multi_step_preds, traj_min, traj_max)

                    y_change = batch_y[:, 1:] - batch_y[:, :-1]
                    pred_change = multi_step_preds[:, 1:] - multi_step_preds[:, :-1]
                    trend_change_loss = F.mse_loss(pred_change, y_change)

                    ar_consist_loss = F.mse_loss(multi_step_preds[:, 1:], multi_step_preds[:, :-1] + y_change)

                    val_total_loss = (
                            base_loss
                            + conn_weight * connection_loss
                            + smooth_weight * smooth_loss
                            + trend_weight * trend_match_loss
                            + phys_weight * phys_loss
                            + 0.5 * trend_change_loss
                            + 0.4 * ar_consist_loss
                    )
                    val_loss += val_total_loss.item()

            # 损失平均
            avg_train_loss = train_loss / len(train_loader)
            avg_val_loss = val_loss / len(val_loader)

            # 记录损失
            self.training_results['transformer']["val_losses"].append(avg_val_loss)
            self.training_results['transformer']["train_losses"].append(avg_train_loss)

            # 学习率调度
            self.transformer_scheduler.step(avg_val_loss)

            # 早停逻辑
            if avg_val_loss < best_val_loss - self.loss_threshold:
                best_val_loss = avg_val_loss
                patience_counter = 0
                self.best_model_state['transformer'] = self.models['transformer'].state_dict().copy()
                logger.debug(f"Transformer Epoch {epoch + 1} - 验证损失更新至最佳:{best_val_loss:.6f}")
            else:
                patience_counter += 1
                logger.debug(
                    f"Transformer Epoch {epoch + 1} - 验证损失未改善, 耐心值:{patience_counter}/{self.early_stopping_patience}")

            # 日志打印
            logger.info(
                f"Transformer Epoch {epoch + 1}/{self.epochs} - 训练损失:{avg_train_loss:.6f}, 验证损失:{avg_val_loss:.6f}")

            # 早停触发
            if patience_counter >= self.early_stopping_patience:
                self.save_model(self.filepath, 'transformer', epoch, is_best=True)
                logger.info(f"Transformer早停触发，最佳验证损失:{best_val_loss:.6f}")
                break
            self.save_model(self.filepath, 'transformer', epoch)

        # 加载最佳模型
        if self.best_model_state['transformer'] is not None:
            self.models['transformer'].load_state_dict(self.best_model_state['transformer'])
            self.save_model(self.filepath, 'transformer', epoch="_final_best", is_best=True)
        return self

    def predict(
            self,
            horizon: int,
            model_name,
            sequence,
            confidence_intervals: bool = True,
    ) -> Dict[str, np.ndarray]:
        logger.info(f"开始预测 - 模型:{model_name}, 预测步长:{horizon}, 输入序列长度:{len(sequence)}")

        # 关键修复1：提前初始化lower/upper_bound为None
        lower_bound = None
        upper_bound = None

        # 校验scaler是否已设置
        if self.scaler is None:
            logger.error("EnsembleModel的scaler未设置！请先调用set_scaler传入拟合好的scaler")
            raise RuntimeError("MinMaxScaler not fitted! Call set_scaler first.")

        # ========== Prophet预测逻辑 ==========
        if model_name == 'prophet':
            prophet_result = self.models['prophet'].predict(horizon, sequence, confidence_intervals)
            forecast = prophet_result['forecast']
            lower_bound = prophet_result['lower_bound']
            upper_bound = prophet_result['upper_bound']
        elif model_name == 'arima':
            # ARIMA预测逻辑
            arima_result = self.models['arima'].predict(horizon, sequence, confidence_intervals)
            forecast = arima_result['forecast']
            lower_bound = arima_result['lower_bound']
            upper_bound = arima_result['upper_bound']
        else:
            # LSTM/Transformer预测逻辑（核心修改）
            self.models[model_name].eval()
            with torch.no_grad():
                # 构造输入：[1, seq_len, 1]
                current_sequence = torch.FloatTensor(sequence).unsqueeze(0).unsqueeze(-1).to(self.device)
                # 一次前向得到所有horizon步预测值
                pred = self.models[model_name](current_sequence).squeeze(0).cpu().numpy()  # [horizon]

            # 校验预测长度
            if len(pred) != horizon:
                raise RuntimeError(f"{model_name} 预测长度错误: 预期 {horizon}, 实际 {len(pred)}")

            # 逆变换
            forecast = self.scaler.inverse_transform(pred.reshape(-1, 1)).flatten()
            logger.debug(f"{model_name}预测值范围:[{forecast.min():.2f}, {forecast.max():.2f}]")

            # 计算置信区间
            if confidence_intervals:
                if model_name in self.training_results and len(self.training_results[model_name]['train_losses']) > 0:
                    rmse = np.sqrt(np.mean(self.training_results[model_name]['train_losses'][-10:]))
                    std_error = rmse
                else:
                    std_error = 0.1
                lower_bound = forecast - 1.96 * std_error
                upper_bound = forecast + 1.96 * std_error

        # 返回统一格式
        result = {
            'forecast': forecast,
            'lower_bound': lower_bound,
            'upper_bound': upper_bound
        }
        return result

    # 适配工具方法（跳过Prophet的训练/保存/绘图）


    def plot_training_history(self, model_name) -> None:
        """Plot training and validation loss history."""
        if not self.training_results[model_name]["train_losses"]:
            logger.error(f"{model_name}无训练历史数据，无法绘图")
            raise ValueError("No training history available")

        plt.figure(figsize=(10, 6))
        plt.plot(self.training_results[model_name]["train_losses"], label='Training Loss')
        plt.plot(self.training_results[model_name]["val_losses"], label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('{} Training History'.format(model_name))
        plt.legend()
        plt.grid(True)
        plt.show()
        logger.info(f"{model_name}训练历史绘图完成")

    def save_model(self, filepath, model_name, epoch=None, is_best=False):
        if model_name in ['arima', 'prophet']:
            return
        os.makedirs(filepath, exist_ok=True)
        timestamp = datetime.now().strftime("%m%d_%H%M")

        # 保存模型参数 + scaler
        save_dict = {
            'model_state_dict': self.models[model_name].state_dict(),
            'scaler': self.scaler,
            'args': self.args
        }

        if is_best:
            save_path = f"{filepath}/{model_name}_BEST_{timestamp}.pt"
        else:
            save_path = f"{filepath}/{model_name}_epoch{epoch}_{timestamp}.pt"
        torch.save(save_dict, save_path)

    def load_model(self, filepath: str, model_name):
        if model_name in ['arima', 'prophet']:
            return self.models[model_name]
        logger.info(f"加载模型 + scaler: {model_name}")
        save_dict = torch.load(filepath, map_location=self.device)

        if model_name == 'lstm':
            self.models[model_name] = EnhancedLSTM(save_dict['args'])
        else:
            self.models[model_name] = TimeSeriesTransformer(save_dict['args'])

        self.models[model_name].load_state_dict(save_dict['model_state_dict'])
        self.models[model_name].to(self.device)
        self.models[model_name].eval()

        # 恢复scaler
        self.scaler = save_dict['scaler']
        self.set_scaler(self.scaler)


    def evaluate(
            self,
            test_data,
            model_name,
            metrics: Optional[List[str]] = None
    ) -> Dict[str, float]:
        """
        评估模型性能：随机抽样50条轨迹，按5:5分割每条轨迹，计算各指标的平均值
        """
        logger.info(f"开始评估模型 - 模型:{model_name}, 测试数据轨迹数:{len(test_data)}")

        # 1. 初始化参数：设置默认指标，初始化结果字典（避免+=时报错）
        if metrics is None:
            metrics = ['mae', 'mse', 'rmse', 'mape', 'smape', 'mase']
        results = {metric: 0.0 for metric in metrics}

        random.seed(42)  # 固定随机种子保证可复现
        sample_size = min(50, len(test_data))
        if sample_size == 0:
            logger.warning("测试数据为空，评估结果全为0")
            return results
        sampled_data = random.sample(test_data, sample_size)
        logger.info(f"评估抽样完成 - 抽样轨迹数:{sample_size}")

        # 3. 遍历抽样数据，累加各指标
        valid_count = 0  # 记录有效评估的轨迹数（避免部分轨迹报错导致平均值偏差）
        for idx, i in enumerate(sampled_data):
            try:
                # 修复原代码错误：split_idx = int(i * 0.5) → i是DF，应使用total_len
                total_len = len(i)
                if total_len < 2:  # 至少能分割出1个训练/测试样本
                    logger.debug(f"轨迹{idx}长度不足({total_len})，跳过")
                    continue

                split_idx = int(total_len * 0.5)  # 5:5分割轨迹
                test_data_forecast = i.iloc[:split_idx]
                test_data_true = i.iloc[split_idx:split_idx+self.args.horizon]
                horizon = len(test_data_true)

                seq_len = self.sequence_length
                if len(test_data_forecast) < seq_len:
                    logger.debug(f"轨迹{idx}输入序列过短({len(test_data_forecast)})，跳过")
                    continue
                # 截取最后seq_len个数据作为输入
                test_data_forecast = test_data_forecast.iloc[-seq_len:]

                # 空值/长度校验
                if horizon == 0 or len(test_data_forecast) == 0:
                    logger.debug(f"轨迹{idx}分割后数据为空，跳过")
                    continue
                y_true = test_data_true['y'].values
                test_data_forecast_vel = self.scaler.transform(test_data_forecast['y'].values.reshape(-1, 1)).flatten()
                predictions = self.predict(horizon, model_name, test_data_forecast_vel, confidence_intervals=False)
                y_pred = predictions['forecast']

                # 校验预测值和真实值长度一致
                if len(y_pred) != len(y_true):
                    logger.debug(f"轨迹{idx}预测长度({len(y_pred)})与真实长度({len(y_true)})不一致，跳过")
                    continue
                # 4. 计算并累加各指标（处理除以0等异常）
                if 'mae' in metrics:
                    results['mae'] += mean_absolute_error(y_true, y_pred)

                if 'mse' in metrics:
                    results['mse'] += mean_squared_error(y_true, y_pred)

                if 'rmse' in metrics:
                    results['rmse'] += np.sqrt(mean_squared_error(y_true, y_pred))

                if 'mape' in metrics:
                    # 替换y_true中的0为极小值，避免除以0
                    y_true_mape = np.where(y_true == 0, 1e-8, y_true)
                    mape = np.mean(np.abs((y_true_mape - y_pred) / y_true_mape)) * 100
                    results['mape'] += mape

                if 'smape' in metrics:
                    # 分母加极小值，避免除以0
                    smape = np.mean(2 * np.abs(y_true - y_pred) / (np.abs(y_true) + np.abs(y_pred) + 1e-8)) * 100
                    results['smape'] += smape

                if 'mase' in metrics:
                    # Mean Absolute Scaled Error（处理naive_mae为0的情况）
                    naive_forecast = np.roll(y_true, 1)
                    naive_mae = mean_absolute_error(y_true[1:], naive_forecast[1:])
                    naive_mae = naive_mae if naive_mae != 0 else 1e-8  # 避免除以0
                    mase = mean_absolute_error(y_true, y_pred) / naive_mae
                    results['mase'] += mase

                valid_count += 1  # 有效轨迹数+1

            except Exception as e:
                logger.error(f"轨迹{idx}评估出错: {str(e)}", exc_info=True)
                continue

        # 5. 计算各指标的平均值（除以有效轨迹数，避免无效轨迹影响）
        if valid_count > 0:
            for metric in results:
                results[metric] = results[metric] / valid_count
            logger.info(f"模型评估完成 - 模型:{model_name}, 有效评估轨迹数:{valid_count}, 指标结果:{results}")
        else:
            logger.warning(f"模型{model_name}无有效评估轨迹，结果全为0")

        return results

    def cross_validate(self, data, model_name, n_splits: int = 5, test_size: float = 0.2):
        logger.info(f"开始交叉验证 - 模型:{model_name}, 折数:{n_splits}, 测试集比例:{test_size}")

        cv_results = {'mae': [], 'mse': [], 'rmse': [], 'mape': [], 'smape': []}
        # 合并所有车辆的速度数据（时序交叉验证需连续序列）
        all_vel = pd.concat([df[['y']] for df in data]).reset_index(drop=True)
        n_samples = len(all_vel)
        test_samples = int(n_samples * test_size)
        train_samples = n_samples - test_samples * n_splits

        if train_samples < self.sequence_length:
            logger.error(f"训练数据量不足 - 所需最小长度:{self.sequence_length}, 实际:{train_samples}")
            raise ValueError("训练数据量不足，无法完成交叉验证")

        for i in range(n_splits):
            # 滚动窗口分割：训练集逐步扩展，测试集后移
            train_end = train_samples + i * test_samples
            test_start = train_end
            test_end = test_start + test_samples
            if test_end > n_samples:
                logger.warning(f"折{i + 1}测试集超出数据范围，终止交叉验证")
                break
            # 构造训练/测试数据（按车辆分组的格式）
            train_data = [all_vel.iloc[:train_end]]
            test_data = [all_vel.iloc[test_start:test_end]]
            fold_results = self.evaluate(test_data, model_name)
            for metric, value in fold_results.items():
                if metric in cv_results and not np.isnan(value):
                    cv_results[metric].append(value)
            logger.debug(f"交叉验证折{i + 1}完成 - 指标:{fold_results}")

        # 计算均值和标准差
        cv_summary = {}
        for metric, values in cv_results.items():
            if values:
                cv_summary[f'{metric}_mean'] = np.mean(values)
                cv_summary[f'{metric}_std'] = np.std(values)

        logger.info(f"交叉验证完成 - 模型:{model_name}, 结果:{cv_summary}")
        return cv_summary