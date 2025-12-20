from typing import Dict, Any, Optional, List, Union, Tuple
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
import joblib
from sklearn.preprocessing import MinMaxScaler
import os

"""Main application script for the Time Series Ensemble Forecasting project."""

import argparse
import sys
from pathlib import Path
from typing import Dict, Any, Optional
import pandas as pd
import numpy as np
from datetime import datetime
from forecasting_model import EnsembleModel,logger
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from plotting import TimeSeriesPlotter
import matplotlib.pyplot as plt


def data_preprocess(
        file_path: str,
) -> List:
    logger.info(f"开始数据预处理 - 文件路径:{file_path}")
    try:
        df = pd.read_csv(file_path)  # 读取轨迹数据文件
        logger.info(f"数据读取完成 - 总行数:{len(df)}, 列名:{df.columns.tolist()}")

        # 保留车辆ID、速度列，并添加'y'列（与速度列一致，适配模型输入要求）
        df = df[['Vehicle_ID', 'v_Vel']].copy()
        df.rename(columns={'v_Vel': 'y'}, inplace=True)  # 核心：重命名v_Vel为y
        vehicle_groups = df.groupby('Vehicle_ID', group_keys=False)
        vehicle_dfs = [group for _, group in vehicle_groups]

        logger.info(f"数据预处理完成 - 车辆数量:{len(vehicle_dfs)}, 总轨迹点数:{len(df)}")
        return vehicle_dfs
    except Exception as e:
        logger.error(f"数据预处理失败: {str(e)}", exc_info=True)
        raise


class TimeSeriesEnsemble:
    """Main application class for time series ensemble forecasting."""

    def __init__(self, args):
        """Initialize the application."""
        self.args = args
        self.ensemble_model = None
        self.data = None
        self.cv_folds = args.cv_folds
        self.ensemble_model = EnsembleModel(args)
        self.scaler = MinMaxScaler()
        self.sequence_length = args.sequence_length
        self.plotter = TimeSeriesPlotter()
        logger.info("TimeSeriesEnsemble初始化完成")
        self.data_process()
    def data_process(self):
        data = data_preprocess(self.args.datapath)
        self.data = data.copy()
        X, y_target = self._create_sequences(self.data)

        split_idx = int(len(X) * 0.8)
        self.X_train, self.X_val = X[:split_idx], X[split_idx:]
        self.y_train, self.y_val = y_target[:split_idx], y_target[split_idx:]
        logger.info(f"数据集分割完成 - 训练集:{len(self.X_train)}, 验证集:{len(self.X_val)}")

    def _create_sequences(self, data: List[pd.DataFrame]) -> Tuple[torch.Tensor, torch.Tensor]:
        logger.info(f"构建多步标签序列 - 输入长度:{self.sequence_length}, 关注horizon:{self.args.horizon}")
        X, y = [], []
        all_vel = np.concatenate([df['y'].values for df in data])
        self.scaler.fit(all_vel.reshape(-1, 1))
        self.ensemble_model.set_scaler(self.scaler)

        valid_vehicles = 0
        total_sequences = 0
        horizon = self.args.horizon  # 多步标签长度

        for vehicle_idx, df in enumerate(data):
            vel_data = df['y'].values
            vel_data_scaled = self.scaler.transform(vel_data.reshape(-1, 1)).flatten()
            seq_len = self.sequence_length

            # 每个样本需要：输入序列 + 后续horizon步的真实值
            max_start_idx = len(vel_data_scaled) - seq_len - horizon
            if max_start_idx < 0:
                continue
            valid_vehicles += 1

            for start_idx in range(max_start_idx + 1):
                # 输入序列：[start_idx, start_idx+seq_len]
                input_seq = vel_data_scaled[start_idx:start_idx + seq_len]
                # 多步标签：[start_idx+seq_len, start_idx+seq_len+horizon]
                multi_step_label = vel_data_scaled[start_idx + seq_len:start_idx + seq_len + horizon]

                X.append(input_seq)
                y.append(multi_step_label)
                total_sequences += 1

        X_tensor = torch.FloatTensor(X).unsqueeze(-1)  # [N, seq_len, 1]
        y_tensor = torch.FloatTensor(y)  # [N, horizon]
        logger.info(f"多步序列构建完成 - 标签维度:{y_tensor.shape}")
        return X_tensor, y_tensor

    def train_models(self):
        try:

            train_dataset = TensorDataset(self.X_train, self.y_train)
            val_dataset = TensorDataset(self.X_val, self.y_val)

            # 训练Transformer
            self.ensemble_model.transformer_trainer(train_dataset, val_dataset)
            # 训练LSTM
            self.ensemble_model.lstm_trainer(train_dataset, val_dataset)

            # 将拟合好的scaler传入ensemble_model
            self.ensemble_model.scaler = self.scaler
            logger.info("所有模型训练完成")

        except Exception as e:
            logger.error(f"模型训练失败: {str(e)}", exc_info=True)
            raise

    def make_forecast(self, horizon, sequence, true_values: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        生成集成预测，返回包含历史值、预测值、真实值的结果

        Args:
            horizon: 预测步长
            sequence: 输入的历史序列（缩放后）
            true_values: 预测段的真实值（缩放后），可选

        Returns:
            包含各模型预测结果的字典，每个结果含historical/forecast/true/lower_bound/upper_bound
        """
        logger.info(f"开始集成预测 - 预测步长:{horizon}, 输入序列长度:{len(sequence)}")

        if self.ensemble_model is None:
            logger.error("模型未训练，无法预测")
            raise ValueError("Model not trained. Please train models first.")

        if horizon is None:
            horizon = self.args.horizon

        # 逆变换历史值（用于可视化）
        historical = self.scaler.inverse_transform(sequence.reshape(-1, 1)).flatten()

        forecasts = {}
        for name, model in self.ensemble_model.models.items():
            # 获取模型预测结果
            pred_result = self.ensemble_model.predict(horizon, name, sequence)

            # 逆变换预测值
            forecast = pred_result['forecast']

            # 逆变换真实值（如果传入）
            true = None
            if true_values is not None:
                # 确保真实值长度和预测步长一致
                true = self.scaler.inverse_transform(
                    true_values[:horizon].reshape(-1, 1)
                ).flatten()

            # 整合结果：历史值+预测值+真实值+置信区间
            forecasts[name] = {
                'historical': historical,  # 历史值（逆变换后）
                'forecast': forecast,  # 预测值（逆变换后）
                'true': true,  # 预测段真实值（逆变换后）
                'lower_bound': pred_result['lower_bound'],  # 置信下限
                'upper_bound': pred_result['upper_bound']  # 置信上限
            }

        logger.info("集成预测完成 - 所有模型预测结果已生成（含历史值/真实值）")
        return forecasts

    def evaluate_models(self, test_split: float = 0.2) -> Dict[str, Dict[str, float]]:
        """Evaluate individual models and ensemble.
        Args:
            test_split: Fraction of data to use for testing

        Returns:
            Dictionary with evaluation results
        """
        logger.info(f"开始模型评估 - 测试集比例:{test_split}")

        if self.ensemble_model is None:
            logger.error("模型未训练，无法评估")
            raise ValueError("Model not trained. Please train models first.")

        # Split data
        split_idx = int(len(self.data) * (1 - test_split))
        test_data = self.data[split_idx:]
        logger.info(f"评估数据集分割完成 - 测试集轨迹数:{len(test_data)}")

        if len(test_data) == 0:
            logger.error("测试数据为空，无法评估")
            raise ValueError("Test data is empty. Increase test_split or use more data.")

        # Evaluate individual models
        individual_results = {}
        for name, model in self.ensemble_model.models.items():
            individual_results[name] = self.ensemble_model.evaluate(test_data, name)
            if self.args.is_train:
                self.ensemble_model.plot_training_history(name)

        logger.info(f"模型评估完成 - 评估结果:{individual_results}")
        return individual_results

    def plot_results(self, historical_data, horizon, forecast_results, evaluation_results):
        """Plot forecasting and evaluation results.

        Args:
            forecast_results: Results from make_forecast
            evaluation_results: Results from evaluate_models
        """
        logger.info("开始绘制结果图表")

        try:
            # Plot ensemble forecast
            self.plotter.plot_forecast_comparison(
                historical_data=historical_data,
                forecasts=forecast_results,
                forecast_horizon=horizon,
                title="Time Series Forecast"
            )

            # # Plot interactive forecast
            # self.plotter.plot_interactive_forecast(
            #     historical_data=historical_data,
            #     forecasts=forecast_results,
            #     forecast_horizon=horizon,
            #     title="Interactive Forecast Comparison"
            # )

            # Plot evaluation results if available
            if evaluation_results:
                self.plotter.plot_model_evaluation(
                    evaluation_results=evaluation_results,
                    title="Model Performance Comparison"
                )
            logger.info("结果图表绘制完成")

        except Exception as e:
            logger.error(f"绘图失败: {str(e)}", exc_info=True)
            raise

    def save_results(self, forecast_results: Dict[str, Any],
                     evaluation_results: Optional[Dict[str, Dict[str, float]]] = None) -> None:
        """Save results to files.

        Args:
            forecast_results: Results from make_forecast
            evaluation_results: Results from evaluate_models
        """
        logger.info("开始保存结果文件")

        try:
            # 补充forecast_dates和ensemble_forecast（兼容原有保存逻辑）
            forecast_dates = np.arange(len(next(iter(forecast_results.values()))['forecast']))
            ensemble_forecast = {
                'forecast': np.mean([v['forecast'] for v in forecast_results.values()], axis=0)
            }

            # Save forecast results
            forecast_df = pd.DataFrame({
                'date': forecast_dates,
                'ensemble_forecast': ensemble_forecast['forecast']
            })

            # Add individual forecasts
            for name, forecast in forecast_results.items():
                forecast_df[f'{name}_forecast'] = forecast['forecast']

            # Save to CSV
            results_dir = Path('results')
            results_dir.mkdir(exist_ok=True)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            forecast_file = results_dir / f"forecast_results_{timestamp}.csv"
            forecast_df.to_csv(forecast_file, index=False)
            logger.info(f"预测结果已保存 - 路径:{forecast_file}")

            # Save evaluation results
            if evaluation_results:
                eval_df = pd.DataFrame(evaluation_results).T
                eval_file = results_dir / f"evaluation_results_{timestamp}.csv"
                eval_df.to_csv(eval_file)
                logger.info(f"评估结果已保存 - 路径:{eval_file}")

        except Exception as e:
            logger.error(f"结果保存失败: {str(e)}", exc_info=True)
            raise

    def run_full_pipeline(self) -> None:
        """Run the complete forecasting pipeline (展示10组轨迹预测对比)."""
        logger.info("=" * 50)
        logger.info("开始时间序列集成预测全流程（展示10组轨迹）")
        logger.info("=" * 50)

        try:
            if (self.args.is_train):
                # 1. 训练模型
                self.train_models()
            else:
                self.ensemble_model.load_model(f"result/{self.args.exp}/{self.args.lstm_model_path}", 'lstm')
                self.ensemble_model.load_model(f"result/{self.args.exp}/{self.args.transformer_model_path}", 'transformer')

            # 2. 准备10组验证集样本（确保索引不重复且合法）
            total_val_samples = len(self.X_val)
            if total_val_samples < 10:
                logger.warning(f"验证集样本不足10个（仅{total_val_samples}个），使用全部样本")
                sample_indices = list(range(total_val_samples))
            else:
                # 随机抽取10个不重复的样本索引
                sample_indices = random.sample(range(total_val_samples), 10)

            logger.info(f"选定10组验证集样本索引: {sample_indices}")

            # 初始化10组轨迹的结果存储
            all_forecast_results = {}  # 存储10组轨迹的预测结果
            all_historical_data = []  # 存储10组轨迹的历史数据
            all_true_values = []  # 存储10组轨迹的真实值

            # 3. 循环处理10组样本
            for idx, sample_idx in enumerate(sample_indices):
                logger.info(f"\n===== 处理第{idx + 1}组轨迹（样本索引:{sample_idx}）=====")
                # 抽取当前组的输入序列和真实值
                X_val_sample = self.X_val[sample_idx].squeeze(-1).numpy()  # 输入序列（缩放后）
                y_val_sample = self.y_val[sample_idx].numpy()  # 对应预测段真实值（缩放后）

                # 逆变换历史值（用于可视化）
                historical_vals = self.scaler.inverse_transform(X_val_sample.reshape(-1, 1)).flatten()
                historical_data = pd.DataFrame({'y': historical_vals})
                # 逆变换真实值（用于后续验证）
                true_vals = self.scaler.inverse_transform(y_val_sample[:self.args.horizon].reshape(-1, 1)).flatten()

                # 存储当前组的基础数据
                all_historical_data.append(historical_data)
                all_true_values.append(true_vals)

                # 3.1 预测（传入真实值）
                forecast_results = self.make_forecast(
                    horizon=self.args.horizon,
                    sequence=X_val_sample,
                    true_values=y_val_sample
                )
                all_forecast_results[f"轨迹{idx + 1}"] = forecast_results

                # 3.2 单组轨迹的独立可视化（可选，如需单独查看）
                # self.plot_results(
                #     historical_data=historical_data,
                #     horizon=self.args.horizon,
                #     forecast_results=forecast_results,
                #     evaluation_results=None  # 单组不评估，最后统一评估
                # )

            # 4. 批量绘制10组轨迹的对比图（子图布局）
            self.plot_10_trajectories(
                historical_data_list=all_historical_data,
                forecast_results_dict=all_forecast_results,
                horizon=self.args.horizon
            )

            # 5. 整体模型评估（基于全部测试集）
            evaluation_results = self.evaluate_models()
            logger.info(f"\n模型整体评估结果: {evaluation_results}")

            if evaluation_results:
                self.plotter.plot_model_evaluation(
                    evaluation_results=evaluation_results,
                    title="Model Performance Comparison"
                )

            # 6. 保存10组轨迹的预测结果
            self.save_10_trajectories_results(all_forecast_results, evaluation_results)

            logger.info("\n" + "=" * 50)
            logger.info("10组轨迹预测全流程执行完成")
            logger.info("=" * 50)

        except Exception as e:
            logger.error(f"全流程执行失败: {str(e)}", exc_info=True)
            raise

    def plot_10_trajectories(self, historical_data_list: List[pd.DataFrame],
                             forecast_results_dict: Dict[str, Dict[str, Any]],
                             horizon: int):
        """
        批量绘制10组轨迹的预测对比图（5行2列子图）
        """
        logger.info("开始绘制10组轨迹的预测对比图")

        # ========== 解决中文乱码的核心配置 ==========
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Noto Sans CJK SC']  # 支持中文的字体（选系统里有的）
        plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示为方块的问题
        plt.rcParams['font.family'] = 'sans-serif'  # 统一字体家族
        # ==========================================

        # 创建5行2列的子图布局
        fig, axes = plt.subplots(5, 2, figsize=(20, 25))
        axes = axes.flatten()  # 展平为一维数组，方便循环

        # 定义颜色映射（区分LSTM/Transformer/真实值）
        colors = {
            'lstm': 'blue',
            'transformer': 'red',
            'arima': 'green',
            'prophet': 'purple',  # 新增Prophet颜色
            'true': 'black'
        }

        # 循环填充每个子图
        for idx, (traj_name, forecast_results) in enumerate(forecast_results_dict.items()):
            if idx >= len(axes):  # 防止超出子图数量
                break

            ax = axes[idx]
            historical_data = historical_data_list[idx]
            # 1. 绘制历史值
            hist_x = np.arange(len(historical_data))
            ax.plot(hist_x, historical_data['y'], 'k-', linewidth=1.5, label='历史值', alpha=0.8)

            # 2. 绘制预测段X轴（接在历史值后）
            forecast_x = np.arange(len(historical_data), len(historical_data) + horizon)

            # 3. 绘制真实值
            true_vals = forecast_results['lstm']['true']  # 所有模型的真实值一致，取其一即可
            if true_vals is not None and len(true_vals) == len(forecast_x):
                ax.plot(forecast_x, true_vals, color=colors['true'], linewidth=2, label='真实值', alpha=0.9)

            # 4. 绘制各模型预测值
            for model_name, pred_data in forecast_results.items():
                pred_vals = pred_data['forecast'][:len(forecast_x)]
                ax.plot(forecast_x, pred_vals, color=colors[model_name],
                        linewidth=1.5, label=f'{model_name}预测', alpha=0.8)

                # 绘制置信区间（可选）
                if pred_data.get('lower_bound') is not None:
                    lower = pred_data['lower_bound'][:len(forecast_x)]
                    upper = pred_data['upper_bound'][:len(forecast_x)]
                    ax.fill_between(forecast_x, lower, upper, color=colors[model_name], alpha=0.1)

            # 子图样式设置
            ax.set_title(traj_name, fontsize=12, fontweight='bold')
            ax.set_xlabel('时间步', fontsize=10)
            ax.set_ylabel('速度', fontsize=10)
            ax.legend(fontsize=8, loc='upper right')
            ax.grid(True, alpha=0.3)

        # 整体标题和布局调整
        fig.suptitle('10组轨迹的速度预测对比（历史值+真实值+模型预测值）', fontsize=16, fontweight='bold')
        plt.tight_layout(rect=[0, 0, 1, 0.98])  # 预留整体标题空间

        # 保存图片
        results_dir = Path('results')
        results_dir.mkdir(exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        fig.savefig(results_dir / f"10_trajectories_forecast_{timestamp}.png",
                    dpi=300, bbox_inches='tight')
        plt.show()

    def save_10_trajectories_results(self, all_forecast_results: Dict[str, Dict[str, Any]],
                                     evaluation_results: Optional[Dict[str, Dict[str, float]]] = None):
        """
        保存10组轨迹的预测结果（按轨迹编号区分）
        """
        logger.info("开始保存10组轨迹的预测结果")
        results_dir = Path('results')
        results_dir.mkdir(exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # 1. 保存汇总的预测结果
        summary_data = []
        for traj_name, forecast_results in all_forecast_results.items():
            # 提取当前轨迹的核心数据
            lstm_pred = forecast_results['lstm']['forecast']
            transformer_pred = forecast_results['transformer']['forecast']
            true_vals = forecast_results['lstm']['true']
            historical_len = len(forecast_results['lstm']['historical'])

            # 构造每行数据（轨迹名 + 时间步 + 真实值 + 各模型预测值）
            for step in range(len(true_vals)):
                summary_data.append({
                    '轨迹编号': traj_name,
                    '预测时间步': step + 1,
                    '真实值': true_vals[step],
                    'LSTM预测值': lstm_pred[step],
                    'Transformer预测值': transformer_pred[step]
                })

        # 保存汇总CSV
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv(results_dir / f"10_trajectories_forecast_summary_{timestamp}.csv",
                          index=False, encoding='utf-8')

        # 2. 保存整体评估结果
        if evaluation_results:
            eval_df = pd.DataFrame(evaluation_results).T
            eval_df.to_csv(results_dir / f"model_evaluation_results_{timestamp}.csv",
                           encoding='utf-8')

        logger.info(f"10组轨迹结果保存完成 - 汇总文件: 10_trajectories_forecast_summary_{timestamp}.csv")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Time Series Ensemble Forecasting")
    parser.add_argument('--lstm_hidden_size', default=64, type=int)
    parser.add_argument('--lstm_num_layers', default=2, type=int)
    parser.add_argument('--lstm_dropout', default=0.0, type=float)

    parser.add_argument('--transformer_d_model', default=128, type=int)
    parser.add_argument('--transformer_nhead', default=8, type=int)
    parser.add_argument('--transformer_num_layers', default=2, type=int)
    parser.add_argument('--transformer_dropout', default=0.0, type=float)

    parser.add_argument('--order', default=(5, 1, 0), type=tuple)
    parser.add_argument('--auto_arima', default=True, type=bool)
    parser.add_argument('--max_p', default=5, type=int)
    parser.add_argument('--max_d', default=2, type=int)
    parser.add_argument('--max_q', default=5, type=int)
    parser.add_argument('--seasonal', default=False, type=bool)
    parser.add_argument('--m', default=12, type=int)

    parser.add_argument('--prophet_yearly_seasonality', default=True, type=bool)
    parser.add_argument('--prophet_weekly_seasonality', default=True, type=bool)
    parser.add_argument('--prophet_daily_seasonality', default=False, type=bool)
    parser.add_argument('--prophet_seasonality_mode', default='additive', type=str)
    parser.add_argument('--prophet_changepoint_prior_scale', default=0.05, type=float)
    parser.add_argument('--prophet_interval_width', default=0.8, type=float)


    parser.add_argument('--sequence_length', default=40, type=int)
    parser.add_argument('--learning_rate', default=0.0005, type=float)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--epochs', default=300, type=int)
    parser.add_argument('--exp', default='test_new', type=str)
    parser.add_argument('--early_stopping_patience', default=100, type=int)
    parser.add_argument('--cv_folds', default=5, type=int)

    parser.add_argument('--horizon', default=20, type=int)
    parser.add_argument('--is_train', default=False, type=bool)
    parser.add_argument('--lstm_model_path', default="lstm_epoch141_1218_1210.pt", type=str)
    parser.add_argument('--transformer_model_path', default="transformer_epoch115_1218_2115.pt", type=str)
    parser.add_argument('--loss_threshold', default=1e-6, type=float)
    parser.add_argument('--teacher_forcing_ratio', default=0.7, type=float)
    parser.add_argument('--save_interval_ratio', default=1/10, type=float)
    parser.add_argument( '--enabled_models',default=['transformer','lstm','arima'])

    args = parser.parse_args()
    args.filepath = os.path.abspath(f"./result/{args.exp}/")
    args.datapath = "../data/smoothed_data.csv"

    # 打印参数配置
    logger.info(f"运行参数配置: {args}")

    # 初始化并运行
    app = TimeSeriesEnsemble(args)
    app.run_full_pipeline()


if __name__ == "__main__":
    main()