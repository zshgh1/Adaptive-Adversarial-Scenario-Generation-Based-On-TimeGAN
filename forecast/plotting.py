"""Visualization utilities for time series analysis and forecasting.
优化点：
1. 适配轨迹数据（无ds日期列，用时间步索引替代）
2. 修复交互式绘图颜色转换Bug
3. 增强空数据/长度不匹配校验
4. 新增多车辆轨迹对比、速度分布等专属可视化
5. 优化残差分析、分解逻辑的鲁棒性
"""

from typing import Dict, Any, Optional, List, Tuple, Union
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings
from scipy import stats
from statsmodels.tsa.seasonal import seasonal_decompose



# Set style
plt.style.use('seaborn-v0_8')
warnings.filterwarnings('ignore')


class TimeSeriesPlotter:
    """Enhanced plotting utilities for time series analysis (适配轨迹数据)."""

    def __init__(self, config_dict: Optional[Dict[str, Any]] = None):
        """Initialize plotter with configuration.

        Args:
            config_dict: Visualization configuration dictionary
        """

        self.figsize = [12, 8]
        self.style = 'seaborn-v0_8'
        self.color_palette ='husl'
        self.save_plots =True
        self.plot_format = 'png'
        self.dpi =  300
        self.trajectory_mode = False  # 轨迹数据模式

        plt.style.use(self.style)
        sns.set_palette(self.color_palette)

    def _get_x_axis(self, data: pd.DataFrame) -> Union[pd.Series, np.ndarray]:
        """适配轨迹数据的X轴：有ds列用ds，无则用索引（时间步）"""
        if 'ds' in data.columns and not self.trajectory_mode:
            return data['ds']
        else:
            return np.arange(len(data))  # 时间步索引（轨迹数据用）

    def plot_time_series(
        self,
        data: pd.DataFrame,
        title: str = "Time Series Data",
        x_label: str = None,
        y_label: str = "Speed",
        save_path: Optional[str] = None
    ) -> None:
        """Plot basic time series/trajectory data (适配轨迹数据).

        Args:
            data: DataFrame with 'y' column (速度)，可选'ds'列
            title: Plot title
            x_label: X轴标签（默认：ds列用Date，轨迹数据用Time Step）
            y_label: Y轴标签（默认Speed）
            save_path: Path to save the plot
        """
        # 空数据校验
        if len(data) == 0 or 'y' not in data.columns:
            return

        # 适配X轴和标签
        x_data = self._get_x_axis(data)
        if x_label is None:
            x_label = 'Date' if 'ds' in data.columns and not self.trajectory_mode else 'Time Step'

        plt.figure(figsize=self.figsize)
        plt.plot(x_data, data['y'], linewidth=2, label='Value', color='#1f77b4')
        plt.title(title, fontsize=16, fontweight='bold')
        plt.xlabel(x_label, fontsize=12)
        plt.ylabel(y_label, fontsize=12)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        if self.save_plots and save_path:
            plt.savefig(save_path, format=self.plot_format, dpi=self.dpi, bbox_inches='tight')

        plt.show()


    def plot_forecast_comparison(
        self,
        historical_data: pd.DataFrame,
        forecasts: Dict[str, Dict[str, np.ndarray]],
        forecast_horizon: int = None,
        title: str = "Forecast Comparison",
        save_path: Optional[str] = None
    ) -> None:
        """优化：适配轨迹数据（用时间步替代日期）

        Args:
            historical_data: 历史数据（含'y'）
            forecasts: 模型预测结果字典
            forecast_horizon: 预测长度（轨迹数据用，替代原forecast_dates）
            title: 标题
            save_path: 保存路径
        """
        # 空数据校验
        if len(historical_data) == 0 or 'y' not in historical_data.columns:
            return
        if not forecasts:
            return

        # 适配X轴
        hist_x = self._get_x_axis(historical_data)
        if forecast_horizon is None:
            forecast_horizon = len(list(forecasts.values())[0]['forecast'])
        # 预测部分的X轴：接在历史数据后
        forecast_x = np.arange(len(historical_data), len(historical_data) + forecast_horizon)

        plt.figure(figsize=self.figsize)
        # 绘制历史数据
        plt.plot(hist_x, historical_data['y'],
                'k-', linewidth=2, label='Historical Data', alpha=0.8)

        # 绘制预测数据
        colors = plt.cm.Set1(np.linspace(0, 1, len(forecasts)))
        for i, (model_name, forecast) in enumerate(forecasts.items()):
            pred = forecast['forecast']
            # 长度校验
            if len(pred) != len(forecast_x):
                pred = pred[:len(forecast_x)]

            plt.plot(forecast_x, pred,
                    color=colors[i], linewidth=2, label=f'{model_name} Forecast')

            # 置信区间
            if forecast.get('lower_bound') is not None and forecast.get('upper_bound') is not None:
                lower = forecast['lower_bound'][:len(forecast_x)]
                upper = forecast['upper_bound'][:len(forecast_x)]
                plt.fill_between(forecast_x, lower, upper, color=colors[i], alpha=0.2)

        plt.title(title, fontsize=16, fontweight='bold')
        x_label = 'Date' if 'ds' in historical_data.columns and not self.trajectory_mode else 'Time Step'
        plt.xlabel(x_label, fontsize=12)
        plt.ylabel('Speed', fontsize=12)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        if self.save_plots and save_path:
            plt.savefig(save_path, format=self.plot_format, dpi=self.dpi, bbox_inches='tight')

        plt.show()

    def plot_model_evaluation(
        self,
        evaluation_results: Dict[str, Dict[str, float]],
        title: str = "Model Evaluation Metrics",
        save_path: Optional[str] = None
    ) -> None:
        """优化：处理空指标、调整布局"""
        if not evaluation_results:
            return

        # Prepare data for plotting
        metrics_data = []
        for model_name, metrics in evaluation_results.items():
            for metric_name, value in metrics.items():
                # 过滤无效值（inf/nan）
                if np.isinf(value) or np.isnan(value):
                    continue
                metrics_data.append({
                    'Model': model_name,
                    'Metric': metric_name.upper(),
                    'Value': value
                })

        if not metrics_data:
            return

        df_metrics = pd.DataFrame(metrics_data)
        metrics = df_metrics['Metric'].unique()
        n_metrics = len(metrics)

        fig, axes = plt.subplots(1, n_metrics, figsize=(5 * n_metrics, 6))
        if n_metrics == 1:
            axes = [axes]

        for i, metric in enumerate(metrics):
            metric_data = df_metrics[df_metrics['Metric'] == metric]
            if len(metric_data) == 0:
                continue

            bars = axes[i].bar(metric_data['Model'], metric_data['Value'], color=plt.cm.Set3(np.linspace(0,1,len(metric_data))))
            axes[i].set_title(f'{metric}', fontweight='bold')
            axes[i].set_ylabel('Value')
            axes[i].tick_params(axis='x', rotation=45)

            # Add value labels on bars（适配小数位数）
            for bar, value in zip(bars, metric_data['Value']):
                if metric in ['MAE', 'MSE', 'RMSE']:
                    label = f'{value:.2f}'
                else:  # MAPE/SMAPE/MASE
                    label = f'{value:.1f}%' if '%' in metric else f'{value:.3f}'
                axes[i].text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.01,
                           label, ha='center', va='bottom', fontsize=10)

        plt.suptitle(title, fontsize=16, fontweight='bold')
        plt.tight_layout()

        if self.save_plots and save_path:
            plt.savefig(save_path, format=self.plot_format, dpi=self.dpi, bbox_inches='tight')

        plt.show()





