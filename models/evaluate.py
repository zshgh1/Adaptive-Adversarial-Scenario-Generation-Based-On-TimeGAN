# Local modules
import os
import pickle
from typing import Dict, Union

# 3rd party modules
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pylab import mpl
from scipy.signal import savgol_filter
from typing import Dict, List
from scipy.fftpack import fft
import math
import json
import time


def getTrajectoryEnergy(vellist):
    steptime=len(vellist)
    Fs = 10
    vellist_fq = fft(vellist)
    vellist_fq_half = vellist_fq[0:int(steptime / 2 + 1)]
    A = np.abs(vellist_fq_half)
    f = np.arange(int(steptime / 2 + 1)) * Fs / steptime
    A_adj = np.zeros(int(steptime / 2 + 1))
    A_adj[0] = A[0] / steptime
    A_adj[len(A_adj) - 1] = A[len(A) - 1] / steptime
    A_adj[1:len(A_adj) - 2] = 2 * A[1:len(A) - 2] / steptime
    Energy=0
    for i in np.arange(len(f) - 1):
        Energy = Energy + A_adj[i] * f[i] * f[i] * (f[i + 1] - f[i])
    return Energy
def getEnergylist(vellist):
    steptime=len(vellist)
    Fs = 10
    vellist_fq = fft(vellist)
    vellist_fq_half = vellist_fq[0:int(steptime / 2 + 1)]
    A = np.abs(vellist_fq_half)
    f = np.arange(int(steptime / 2 + 1)) * Fs / steptime
    A_adj = np.zeros(int(steptime / 2 + 1))
    A_adj[0] = A[0] / steptime
    A_adj[len(A_adj) - 1] = A[len(A) - 1] / steptime
    A_adj[1:len(A_adj) - 2] = 2 * A[1:len(A) - 2] / steptime
    EnergyList=[]
    for i in np.arange(len(f) - 1):
        EnergyList.append(A_adj[i])
    return EnergyList

class Controller:
    """控制器类，封装PID/IDM/CACC/LCM四种算法，与车辆绑定"""

    def __init__(self, car):
        """初始化控制器，绑定到具体车辆"""
        self.car = car  # 关联的车辆对象
        self.reset()  # 初始化控制器状态

    def reset(self):
        """重置控制器状态（如PID积分项）"""
        if self.car.m_mode == "PID":
            self.car.m_integral = 0
            self.car.m_preverror = 0
        # 其他控制器若有状态变量，在此添加重置逻辑

    def compute_acceleration(self, forecar_vel, forecar_acc=None, speed_front_pre=None,
                             acc_front_pre=None, acc_self_pre=None, fore_car_vellist=None,
                             secondfore_vel=None, secondfore_acc=None, secondfore_dis=None):

        """根据车辆模式选择对应算法计算加速度"""
        mode = self.car.m_mode
        if mode == "PID":
            return self._pid(forecar_vel)
        elif mode == "IDM":
            return self._idm(forecar_vel)
        elif mode == "CACC":
            return self._cacc(forecar_vel, forecar_acc)
        elif mode == "CACC_TPF":
            return self._cacc_tpf(forecar_vel, forecar_acc, secondfore_vel, secondfore_acc, secondfore_dis)

        elif mode == "LCM":
            return self._lcm(forecar_vel)
        elif mode == "PID_safe":
            return self._pid_safe(forecar_vel, forecar_acc, speed_front_pre, acc_front_pre, acc_self_pre)
        elif mode == "OVM":
            return self._ovm(forecar_vel)
        elif mode == "FVD":
            return self._fvd(forecar_vel)
        elif mode == "GM4":
            return self._gm4(forecar_vel)
        else:
            return 0


    # ------------------------------
    # 各控制器核心算法
    # ------------------------------
    def _pid(self, forecar_vel):
        """PID控制器算法"""
        error = forecar_vel - self.car.m_speed
        dt = 0.1  # 时间步长

        # 积分项累加
        self.car.m_integral += error * dt
        # 微分项计算
        derivative = (error - self.car.m_preverror) / dt if dt != 0 else 0

        # 计算加速度（使用车辆自身的PID参数）
        acc = (self.car.m_kp * error
               + self.car.m_ki * self.car.m_integral
               + self.car.m_kd * derivative)

        # 更新前向误差
        self.car.m_preverror = error
        return acc

    def cacc_safe_acceleration(self, speed_front_pre, acc_front_pre, acc_front_pro, acc_self_pre):

        # # print(f"speed_front_pre{speed_front_pre}, acc_front_pre{acc_front_pre}, acc_front_pro{acc_front_pro},  acc_self_pre{acc_self_pre}")
        # distance_real = self.car.m_dis + self.car.m_distance_error_com
        # speed_front_real = speed_front_pre + self.car.m_speed_front_error
        # speed_self_real = self.car.m_speed + acc_self_pre * self.car.m_time_com_self
        # dx_com = (distance_real + speed_front_real * self.car.m_time_com_self +
        #           acc_front_pre * self.car.m_time_com_self ** 2 / 2 - self.car.m_speed * self.car.m_time_com_self -
        #           acc_self_pre * self.car.m_time_com_self ** 2 / 2)
        # dp_f = (
        #                    speed_front_real + acc_front_pre * self.car.m_time_com_self) * self.car.m_period + acc_front_pro * self.car.m_period ** 2 / 2
        # de_f = (
        #                    speed_front_real + acc_front_pre * self.car.m_time_com_self + acc_front_pro * self.car.m_period) ** 2 / (
        #                2 * self.car.m_acc_front_min)
        # parameter1 = self.car.m_s0 - dx_com - dp_f - de_f
        # parameter2 = 2 * self.car.m_acc_self_min * (parameter1 - speed_self_real * self.car.m_period)
        # result_para = speed_self_real / self.car.m_period - self.car.m_acc_self_min / 2
        #
        # expr = result_para ** 2 + (parameter2 - speed_self_real ** 2) / (self.car.m_period ** 2)
        #
        # if expr < 0:
        #     # print(f"警告：sqrt 输入为负数 ({expr})，已修正为 0")  # 调试用
        #     expr = 0  # 或根据业务逻辑设置合理的默认值（如取绝对值）
        # acc_self_pro = math.sqrt(expr) - result_para
        # # print(acc_self_pro)  # 调试用
        # return acc_self_pro

        gap = self.car.m_dis
        distance_real = gap + self.car.m_distance_error_com * gap
        dp_f = speed_front_pre * self.car.m_period + acc_front_pro * self.car.m_period ** 2 / 2
        de_f = (speed_front_pre + acc_front_pro * self.car.m_period) ** 2 / (2 * self.car.m_acc_front_min)
        parameter1 = self.car.m_s0 - distance_real - dp_f + de_f
        # print(f"parameter1{parameter1},distance_real{distance_real},dp_f{dp_f},de_f{de_f}")
        parameter2 = 2 * self.car.m_acc_front_min * (parameter1 + self.car.m_speed * self.car.m_period)
        # print(f"parameter2{parameter2}")
        result_para = self.car.m_speed / self.car.m_period - self.car.m_acc_self_min / 2
        # print(f"result_para{result_para}")
        expr = result_para ** 2 + (parameter2 - self.car.m_speed ** 2) / (self.car.m_period ** 2)

        if expr < 0:
            # print(f"警告：sqrt 输入为负数 ({expr})，已修正为 0")  # 调试用
            expr = 0  # 或根据业务逻辑设置合理的默认值（如取绝对值）
        acc_self_pro = math.sqrt(expr) - result_para
        # print(f"acc_self_pro{acc_self_pro}")
        acc_self_pro = min(acc_self_pro, 5)
        acc_self_pro = max(acc_self_pro, -5)
        # print(acc_self_pro)  # 调试用
        return acc_self_pro

    def _idm(self, forecar_vel):
        """IDM控制器算法"""
        current_speed = self.car.m_speed

        vf = self.car.m_vf  # 期望速度
        # vf=forecar_vel
        # if forecar_vel<20:
        #     vf=20

        # 计算安全距离s_star
        v_term = current_speed * self.car.m_h_T
        s_star = self.car.m_s0 + max(0, v_term)  # 避免负项

        # 计算IDM加速度（使用车辆自身的IDM参数）
        acc = self.car.m_a * (
                1 - (current_speed / vf) ** self.car.m_delta
                - (s_star / self.car.m_dis) ** 2
        )
        return acc
    def _cacc(self, forecar_vel, forecar_acc):
        """CACC控制器算法"""
        current_speed = self.car.m_speed

        # 计算期望距离（使用车辆自身的CACC参数）
        expect_dis = self.car.m_h_T * current_speed + self.car.m_s0
        e = self.car.m_dis - expect_dis  # 距离误差
        error = forecar_vel - current_speed  # 速度误差

        # 计算加速度（包含前车加速度前馈）
        acc = (self.car.m_j1 * e
               + self.car.m_j2 * error
               + self.car.m_j3 * (forecar_acc or 0))
        return acc
    def _cacc_tpf(self, forecar_vel, forecar_acc,secondfore_vel,secondfore_acc,secondfore_dis):
        if secondfore_vel==None:
            """CACC控制器算法"""
            current_speed = self.car.m_speed

            # 计算期望距离（使用车辆自身的CACC参数）
            expect_dis = self.car.m_h_T * current_speed + self.car.m_s0
            e = self.car.m_dis - expect_dis  # 距离误差
            error = forecar_vel - current_speed  # 速度误差

            # 计算加速度（包含前车加速度前馈）
            acc = (self.car.m_j1 * e
                   + self.car.m_j2 * error
                   + self.car.m_j3 * (forecar_acc or 0))
            return acc
        current_speed = self.car.m_speed
        # 计算期望距离（使用车辆自身的CACC参数）
        expect_dis = self.car.m_h_T * current_speed + self.car.m_s0
        e1 = self.car.m_dis - expect_dis  # 距离误差
        e2 = secondfore_dis - (expect_dis * 2 + 3)  # 距离误差
        error1 = forecar_vel - current_speed  # 速度误差
        error2 = secondfore_vel - current_speed*2  # 速度误差
        e=self.car.m_alpha*error1+(1-self.car.m_alpha)*error2
        error=self.car.m_alpha*e1+(1-self.car.m_alpha)*e2
        accError=self.car.m_alpha*forecar_acc+(1-self.car.m_alpha)*secondfore_acc

        # 计算加速度（包含前车加速度前馈）
        acc = (self.car.m_j1 * e
               + self.car.m_j2 * error
               + self.car.m_j3 * (accError or 0))
        return acc

    def _lcm(self, forecar_vel):
        """LCM控制器算法"""
        current_speed = self.car.m_speed
        vf = self.car.m_vf  # 期望速度

        vf = forecar_vel
        if forecar_vel < 10:
            vf = 10

        # 计算期望距离（使用车辆自身的LCM参数）
        expect_dis = self.car.m_h_T * current_speed + self.car.m_s0

        # 计算LCM加速度
        acc = self.car.m_A * (
                1 - current_speed / vf
                - math.exp(1 - self.car.m_dis / expect_dis)
        )
        return acc

    def _pid_safe(self, forecar_vel, acc_front_pro, speed_front_pre, acc_front_pre, acc_self_pre):

        """PID控制器算法"""
        current_speed = self.car.m_speed
        error = forecar_vel - current_speed

        # 计算PID各项
        self.car.m_integral += error * 0.1  # 积分项
        derivative = (error - self.car.m_preverror) / 0.1  # 微分项

        # 计算加速度
        acc = (self.car.m_kp * error +
               self.car.m_ki * self.car.m_integral +
               self.car.m_kd * derivative)

        acc_self_pro = self.cacc_safe_acceleration(speed_front_pre, acc_front_pre, acc_front_pro, acc_self_pre)

        if acc > acc_self_pro:
            acc = acc_self_pro
        if acc < -8:
            acc = -8

        # 更新历史误差
        self.car.m_preverror = error
        return acc

    def _ovm(self, forecar_vel):
        current_speed = self.car.m_speed

        vf = forecar_vel
        if forecar_vel < 5:
            vf = 5

        arg1 = vf - current_speed
        error = forecar_vel - current_speed
        a_term = self.car.m_a_ovm * arg1
        b_term = self.car.m_b_ovm * error
        return a_term + b_term

    def _fvd(self, forecar_vel):
        current_speed = self.car.m_speed
        error = forecar_vel - current_speed
        expect_dis = self.car.m_h_T * current_speed + self.car.m_s0
        e = self.car.m_dis - expect_dis  # 距离误差
        arg1 = self.car.m_C1 * (self.car.m_dis - self.car.m_s0) - self.car.m_C2
        arg2 = (self.car.m_V1 + self.car.m_V2 * np.tanh(arg1)) - current_speed
        a = self.car.m_alpha * arg2 + self.car.m_lambda_ * error
        return a

    def _gm4(self, forecar_vel):
        current_speed = self.car.m_speed
        error = forecar_vel - current_speed
        a = self.car.m_alpha * (current_speed / (self.car.m_dis + self.car.m_s0)) * error
        return a

class Car:
    def __init__(self, id, speed=0, dis=0, mode="", modenum="", Length=3):
        # 基础属性（简化命名，明确含义）
        self.m_id = id  # 车辆ID
        self.m_Length = Length  # 车辆长度
        self.m_speed = speed  # 当前速度
        self.m_dis = dis  # 与前车的距离
        self.m_mode = mode  # 控制器模式（如"PID"）
        self.m_modenum = modenum  # 控制器参数字符串

        # 轨迹数据（统一管理历史记录）
        self.m_vellist = [speed]  # 速度历史列表
        self.m_dislist = [dis]  # 距离历史列表
        self.m_vellist_list = []  # 多轮测试速度轨迹集合
        self.m_dislist_list = []  # 多轮测试距离轨迹集合

        # 评估指标（单独分组，便于维护）
        self.m_test_veldiffsumlist = []  # 速度误差总和列表
        self.m_test_dissumlist = []  # 距离误差总和列表
        self.m_is_collosion = []  # 碰撞记录
        self.m_score = []  # 评分记录
        self.m_energy = []  # 能量消耗
        self.m_energylist = []  # 能量消耗历史
        self.m_ttclist = []

        # 控制器参数（按类型分组，减少属性冗余）
        self._init_controller_params()

        # 解析参数并初始化控制器（核心：每个车辆绑定自己的控制器）
        self.parse_parameters()
        self.controller = Controller(self) if mode else None  # 头车无控制器

    def _init_controller_params(self):
        """初始化所有控制器的默认参数（集中管理）"""

        self.m_h_T = 0.8
        self.m_s0 = 2.0
        self.m_vf = 33.3
        # PID参数
        self.m_kp = 0.3
        self.m_ki = 0.05
        self.m_kd = 0.1
        self.m_integral = 0  # PID积分项（状态变量）
        self.m_preverror = 0  # PID前向误差（状态变量）

        # IDM参数
        self.m_a = 2.0
        self.m_b = 1.0
        self.m_delta = 2.0


        # LCM参数
        self.m_A = 4.0


        # CACC参数
        self.m_j1 = 0.25
        self.m_j2 = 1.76
        self.m_j3 = 0.26

        # CACC_TPF参数
        self.m_j1 = 0.25
        self.m_j2 = 1.76
        self.m_j3 = 0.26
        self.m_alpha = 0.7

        # SAFE_PID参数
        self.m_distance_error_com = 0.5  # 通讯距离误差
        self.m_speed_front_error = -0.1  # 前车速度误差
        self.m_acc_front_min = -8.0  # 前车最小加速度
        self.m_acc_self_min = -8.0  # 本车最小加速度
        self.m_time_perception = 0.3  # 传感器时延
        self.m_time_communicate = 0.2  # 通讯时延
        self.m_time_wait_per = 0.05  # 传感器等待时延
        self.m_time_wait_com = 0.05  # 通讯等待时延
        self.m_time_per_self = self.m_time_perception + self.m_time_wait_per  # 本车传感器总时延
        self.m_time_com_self = self.m_time_communicate + self.m_time_wait_com  # 本车通讯总时延
        self.m_period = 1.0  # 执行周期

        # FVD参数
        self.m_C1 = 0.13
        self.m_C2 = 2.79
        self.m_V1 = 16
        self.m_V2 = 16
        self.m_alpha = 0.41
        self.m_lambda_ = 0.5

        # OVM参数
        self.m_a_ovm = 1.75
        self.m_b_ovm = 0.6

        # GM4参数
        self.m_alpha = 0.5



    def parse_parameters(self):
        """解析参数字符串，更新控制器参数（优化解析逻辑）"""
        if not self.m_modenum:
            return  # 头车或无参数时直接返回

        # 分割参数（兼容格式："kp=0.3,ki=0.25"）
        try:
            params = {}
            for pair in self.m_modenum.split(","):
                key, value = pair.split("=")
                params[key.strip()] = float(value.strip())
        except Exception as e:
            print(f"参数解析错误: {e}")
            return

        # 根据模式更新对应参数（避免硬编码索引，提高容错性）
        if self.m_mode == "PID":
            self.m_kp = params.get("kp", self.m_kp)
            self.m_ki = params.get("ki", self.m_ki)
            self.m_kd = params.get("kd", self.m_kd)
        elif self.m_mode == "IDM":
            self.m_a = params.get("a", self.m_a)
            self.m_b = params.get("b", self.m_b)
            self.m_delta = params.get("delta", self.m_delta)
            self.m_h_T = params.get("T", self.m_h_T)
        elif self.m_mode == "CACC":
            self.m_j1 = params.get("j1", self.m_j1)
            self.m_j2 = params.get("j2", self.m_j2)
            self.m_j3 = params.get("j3", self.m_j3)
            self.m_s0 = params.get("veh_s0", self.m_s0)
            self.m_h_T = params.get("veh_Tc", self.m_h_T)
        elif self.m_mode == "CACC_TPF":
            self.m_j1 = params.get("j1", self.m_j1)
            self.m_j2 = params.get("j2", self.m_j2)
            self.m_j3 = params.get("j3", self.m_j3)
            self.m_s0 = params.get("veh_s0", self.m_s0)
            self.m_h_T = params.get("veh_Tc", self.m_h_T)
            self.m_alpha = params.get("alpha", self.m_alpha)
        elif self.m_mode == "LCM":
            self.m_A = params.get("A", self.m_A)
            self.m_h_T = params.get("h_T", self.m_h_T)
            self.m_s0 = params.get("s0", self.m_s0)
        elif self.m_mode == "PID_safe":
            self.m_kp = params.get("kp", self.m_kp)
            self.m_ki = params.get("ki", self.m_ki)
            self.m_kd = params.get("kd", self.m_kd)
            self.m_distance_error_com = params.get("distance_error_com", self.m_distance_error_com) # 通讯距离误差
            self.m_speed_front_error = params.get("speed_front_error", self.m_speed_front_error)  # 前车速度误差
            self.m_acc_front_min = params.get("acc_front_min", self.m_acc_front_min)  # 前车最小加速度
            self.m_acc_self_min = params.get("acc_self_min", self.m_acc_self_min)  # 本车最小加速度
            self.m_time_perception = params.get("time_perception", self.m_time_perception)  # 传感器时延
            self.m_time_communicate = params.get("time_communicate", self.m_time_communicate)  # 通讯时延
            self.m_time_wait_per = params.get("time_wait_per", self.m_time_wait_per)  # 传感器等待时延
            self.m_time_wait_com = params.get("time_wait_com", self.m_time_wait_com)  # 通讯等待时延
            self.m_period = params.get("period", self.m_period)  # 执行周期
        elif self.m_mode == "FVD":
            self.m_C1 = params.get("C1", self.m_C1)
            self.m_C2 = params.get("C2", self.m_C2)
            self.m_V1 = params.get("V1", self.m_V1)
            self.m_V2 = params.get("V2", self.m_V2)
            self.m_alpha = params.get("alpha", self.m_alpha)
            self.m_lambda_ = params.get("lambda", self.m_lambda_)
            self.m_s0 = params.get("s0", self.m_s0)
        elif self.m_mode == "OVM":
            self.m_a_ovm = params.get("a", self.m_a_ovm)
            self.m_b_ovm = params.get("b", self.m_b_ovm)
            self.m_vf = params.get("vf", self.m_vf)
        elif self.m_mode == "GM4":
            self.m_alpha = params.get("alpha", self.m_alpha)

    def reset(self):
        """重置车辆状态（用于多轮测试，调用控制器重置方法）"""
        # 重置速度和距离历史
        self.m_vellist.clear()
        self.m_dislist.clear()
        # 重置控制器状态（如PID积分项）
        if self.controller:
            self.controller.reset()

class Trajectory:
    def __init__(self, vellist):
        self.m_vellist = vellist.flatten() if hasattr(vellist, 'flatten') else vellist
        self.m_maxvel = max(self.m_vellist)
        self.m_minvel = min(self.m_vellist)
        self.m_meanvel = sum(self.m_vellist) / len(self.m_vellist)
        self.m_Distance = sum(self.m_vellist)
        self.m_length = len(self.m_vellist)
        self.m_score = 0
        # 处理空列表情况，避免索引错误
        self.acclist = [self.m_vellist[i + 1] - self.m_vellist[i] for i in range(len(self.m_vellist) - 1)]
        self.maxacc = max(self.acclist)
        self.minacc = min(self.acclist)

        self.m_velErrorLCM = 0
        self.m_disErrorLCM = 0
        self.m_energyLCM = 0
        self.m_scoreLCM = 0
        self.m_is_collosion_LCM = 0
        self.m_ttcLCM = float('inf')

        self.m_velErrorCACC = 0
        self.m_disErrorCACC = 0
        self.m_energyCACC = 0
        self.m_scoreCACC = 0
        self.m_is_collision_CACC = 0
        self.m_ttcCACC = float('inf')

        self.m_velErrorCACC_TPF = 0
        self.m_disErrorCACC_TPF = 0
        self.m_energyCACC_TPF = 0
        self.m_scoreCACC_TPF = 0
        self.m_is_collision_CACC_TPF = 0
        self.m_ttcCACC_TPF = float('inf')

        self.m_velErrorIDM = 0
        self.m_disErrorIDM = 0
        self.m_energyIDM = 0
        self.m_scoreIDM = 0
        self.m_is_collision_IDM = 0
        self.m_ttcIDM = float('inf')

        self.m_velErrorPID = 0
        self.m_disErrorPID = 0
        self.m_energyPID = 0
        self.m_scorePID = 0
        self.m_is_collision_PID = 0
        self.m_ttcPID = float('inf')

        self.m_velErrorPID_safe = 0
        self.m_disErrorPID_safe = 0
        self.m_energyPID_safe = 0
        self.m_scorePID_safe = 0
        self.m_is_collision_PID_safe = 0
        self.m_ttcPID_safe = float('inf')

        self.m_velErrorOVM = 0
        self.m_disErrorOVM = 0
        self.m_energyOVM = 0
        self.m_scoreOVM = 0
        self.m_is_collision_OVM = 0
        self.m_ttcOVM = float('inf')

        self.m_velErrorFVD = 0
        self.m_disErrorFVD = 0
        self.m_energyFVD = 0
        self.m_scoreFVD = 0
        self.m_is_collision_FVD = 0
        self.m_ttcFVD = float('inf')

        self.m_velErrorGM4 = 0
        self.m_disErrorGM4 = 0
        self.m_energyGM4 = 0
        self.m_scoreGM4 = 0
        self.m_is_collision_GM4e = 0
        self.m_ttcGM4 = float('inf')




def reinitcar(carlist):
    """初始化车辆（调用Car类的reset方法，简化逻辑）"""
    for car in carlist:
        if car.m_id == 0:
            continue  # 头车不重置

        # 初始速度和距离（跟随头车）
        init_vel = carlist[0].m_vellist[0]
        init_dis = init_vel * car.m_h_T + car.m_s0

        # 重置车辆状态
        car.m_speed = init_vel
        car.m_dis = init_dis
        car.m_vellist = [init_vel]
        car.m_dislist = [init_dis]
        car.reset()  # 调用车辆自身的重置方法

        # 重置评估指标
        car.m_score.append(0)
        car.m_is_collosion.append(False)

def getmodenum(mode,mode_params):
    """获取控制器参数字符串（使用字典优化，便于扩展）"""

    return mode_params.get(mode, "")
def TestWithgenerated_data(generated_data_test, mode, platoonSize,mode_params):

    modenum = getmodenum(mode,mode_params)
    generated_data = [Trajectory(i) for i in generated_data_test]

    # 创建车辆列表（每个车辆自动初始化控制器）
    carlist = [
        Car(
            id=i,
            speed=0,
            dis=0,
            mode="" if i == 0 else mode,
            modenum="" if i == 0 else modenum,
            Length=3
        )
        for i in range(platoonSize)
    ]

    for n in range(len(generated_data)):
        # 设置头车轨迹
        carlist[0].m_vellist = [max(0, v) for v in generated_data[n].m_vellist.copy()]
        reinitcar(carlist)  # 初始化车辆

        # 遍历时间步
        for i in range(len(carlist[0].m_vellist) - 1):
            for car in carlist[1:]:  # 处理跟驰车辆
                forecar = carlist[car.m_id - 1]
                forecar_vel = forecar.m_vellist[i]

                # 计算前车加速度（用于CACC）
                forecar_acc = (forecar.m_vellist[i] - forecar.m_vellist[i - 1]) * 10 if i > 0 else 0

                forecar_vel_pre = (forecar.m_vellist[i - 1]) if i > 0 else forecar.m_vellist[i]
                forecar_acc_pre = (forecar.m_vellist[i - 1] - forecar.m_vellist[i - 2]) * 10 if i > 1 else 0
                self_acc_pre = (car.m_vellist[i - 1] - car.m_vellist[i - 2]) * 10 if i > 0 else 0

                if (mode == "CACC_TPF"):
                    if (car.m_id < 2):
                        acc = car.controller.compute_acceleration(
                            forecar_vel=forecar_vel,
                            forecar_acc=forecar_acc,
                        )
                    else:

                        secondfore = carlist[car.m_id - 2]
                        secondfore_vel = secondfore.m_vellist[i]
                        secondfore_acc = (secondfore.m_vellist[i] - secondfore.m_vellist[i - 1]) * 10 if i > 0 else 0
                        secondfore_dis = (car.m_dis + forecar.m_dislist[i]) if i > 0 else 0
                        acc = car.controller.compute_acceleration(
                            forecar_vel=forecar_vel,
                            forecar_acc=forecar_acc,
                            secondfore_vel=secondfore_vel,
                            secondfore_acc=secondfore_acc,
                            secondfore_dis=secondfore_dis
                        )
                else:
                    acc = car.controller.compute_acceleration(
                        forecar_vel=forecar_vel,
                        forecar_acc=forecar_acc,
                        speed_front_pre=forecar_vel_pre,
                        acc_front_pre=forecar_acc_pre,
                        acc_self_pre=self_acc_pre,
                    )
                if(acc>3 and acc<-6):
                    print("mode:{},acc:{}".format(mode,acc))
                # 更新车辆状态（速度和距离）
                car.m_speed += acc * 0.1
                car.m_speed = max(0, car.m_speed)  # 速度非负

                speed_error = forecar_vel - car.m_speed
                car.m_dis += speed_error / 10  # 距离更新

                if car.m_dis <= 0:
                    car.m_dis = 0
                    car.m_speed = forecar_vel
                    car.m_is_collosion[n] = True

                    #print("mode:{},collosion".format(mode))

                # 记录轨迹
                car.m_vellist.append(car.m_speed)
                car.m_dislist.append(car.m_dis)
        # 保存评估数据
        for car in carlist:
            car.m_dislist_list.append(car.m_dislist.copy())
            car.m_vellist_list.append(car.m_vellist.copy())
            car.m_energylist.append(getEnergylist(car.m_vellist))
            if car.m_id == 0:
                continue  # 头车无需计算误差
            # 计算平均误差
            dissum = 0.0
            veldiffsum = 0.0
            min_ttc = float('inf')  # 初始化最小TTC为无穷大

            dislist=[]
            veldifflist=[]

            for j in range(len(carlist[0].m_vellist)-1):
                # 期望距离（使用车辆自身的LCM参数，兼容其他模式）
                expect_dis = car.m_h_T * car.m_vellist[j] + car.m_s0
                dissum += abs(car.m_dislist[j] - expect_dis)
                veldiffsum += abs(car.m_vellist[j] - carlist[car.m_id - 1].m_vellist[j])

                dislist.append(abs(car.m_dislist[j] - expect_dis))
                veldifflist.append(abs(car.m_vellist[j] - carlist[car.m_id - 1].m_vellist[j]))
                velocity_diff = carlist[car.m_id - 1].m_vellist[j] - car.m_vellist[j]
                if velocity_diff > 0:  # 只有前车速度大于后车时才计算TTC
                    ttc = car.m_dislist[j] / velocity_diff
                    if ttc < min_ttc:
                        min_ttc = ttc  # 更新最小TTC
            # 保存平均误差
            car.m_test_dissumlist.append(dissum / len(carlist[0].m_vellist) * 100)
            car.m_test_veldiffsumlist.append(veldiffsum / len(carlist[0].m_vellist) * 100)
            # 能量计算（假设函数已定义）
            car.m_energy.append(getTrajectoryEnergy(car.m_vellist))
            car.m_ttclist.append(min_ttc)
    return carlist

def evaluate_trajectory(oridataset, scaling,ctrl,platoonSize,mode_params):
    dataset = oridataset[:, :, 2]
    file_path = "metrics_storage.json"
    with open(file_path, "r") as f:
        metrics = json.load(f)
    carlist = TestWithgenerated_data(dataset[:, :] * scaling, ctrl, platoonSize,mode_params)
    followers = carlist[1:]
    follower_count = len(followers)
    ctrl_trajectories = []
    scores = []
    for traj_idx in range(len(dataset)):
        # 初始化轨迹对象（原始速度轨迹）
        trajectory = Trajectory(dataset[traj_idx] * scaling)

        # 3. 累加所有跟随车的指标
        vel_sum = 0.0  # 速度误差和总和
        dis_sum = 0.0  # 距离误差和总和
        energy_sum = 0.0  # 能量消耗总和
        ttc_sum = 0.0  # 能量消耗总和
        collision = 0
        for car in followers:
            # 累加单条轨迹中所有跟随车的指标
            vel_sum += car.m_test_veldiffsumlist[traj_idx]
            dis_sum += car.m_test_dissumlist[traj_idx]
            energy_sum += car.m_energy[traj_idx]
            collision = car.m_is_collosion[traj_idx]
            ttc_sum += car.m_ttclist[traj_idx]

        # 4. 计算跟随车指标的平均值
        avg_vel = vel_sum / follower_count
        avg_dis = dis_sum / follower_count
        avg_energy = energy_sum / follower_count
        avg_ttc = ttc_sum / follower_count

        # 5. 归一化函数（统一逻辑：值越小评分越高）
        def normalize(x, metric_info):
            """将指标x归一化到[0,1]，x越小得分越高（因误差/能量越小越好）"""
            max_val = metric_info["max"]
            min_val = metric_info["min"]
            if max_val == min_val:  # 避免除零
                return 1.0 if x == min_val else 0.0
            if x > max_val:  # 避免除零
                return 1.0
            if x < min_val:  # 避免除零
                return 0.0
                    # 核心：反向归一化（x越小，得分越高）
            return ( x-min_val) / (max_val - min_val)

        # 6. 计算各指标得分（权重：距离2分，速度2分，能量1分，总分5分）
        try:
            # 从预存指标中获取当前控制器的参考值
            dis_score = normalize(avg_dis, metrics[ctrl]["m_test_dissumlist"]) * 2
            vel_score = normalize(avg_vel, metrics[ctrl]["m_test_veldiffsumlist"]) * 2
            energy_score = normalize(avg_energy, metrics[ctrl]["m_energy"]) * 1
            ttc_score = (1 - normalize(avg_ttc, metrics[ctrl]["m_ttclist"])) * 2

        except KeyError as e:
            print(f"警告：{ctrl} 缺失指标 {e}，使用默认得分0")
            dis_score = vel_score = energy_score= ttc_score = 0.0

        # 7. 计算总分（归一化到100分制）
        total_score = (dis_score + vel_score + energy_score+ttc_score) / 7 * 100
        if(collision):
            total_score=100
        scores.append(total_score)

        # 将得分存入轨迹对象（动态绑定属性，如m_scoreCACC、m_scorePID等）
        setattr(trajectory, f"m_score{ctrl}", total_score)
        setattr(trajectory, f"m_velError{ctrl}", avg_dis)
        setattr(trajectory, f"m_disError{ctrl}", avg_vel)
        setattr(trajectory, f"m_energy{ctrl}", avg_energy)
        setattr(trajectory, f"m_is_collosion_{ctrl}", collision)
        setattr(trajectory, f"m_ttc{ctrl}", avg_ttc)

        ctrl_trajectories.append(trajectory)

    # 存储当前控制器的所有轨迹结果
    sorted_indices = np.argsort(scores)[::-1]  # 降序排序索引
    oridataset = oridataset[sorted_indices]  # 按索引排序数据

    if len(oridataset) > 1000:
        oridataset = oridataset[:1000]

    return oridataset


def get_evaluate_metrics(dataset, scaling,platoonSize,mode_params,controllers) -> Dict:
    """
    计算并比较不同控制器的评估指标（速度误差、距离误差、能量消耗）
    :param dataset: 输入的轨迹数据集
    :return: 包含所有控制器指标的字典
    """
    # 1. 定义控制器列表及对应的测试函数参数

    metrics_keys = ["m_test_veldiffsumlist", "m_test_dissumlist", "m_energy", "m_ttclist"]  # 统一指标键

    # 2. 批量运行测试并计算指标
    all_metrics = {}
    for ctrl in controllers:
        ctrl_name = ctrl
        scaling = scaling

        try:
            carlist = TestWithgenerated_data(dataset[:, :, 2] * scaling, ctrl_name, platoonSize, mode_params)

            # 计算该控制器的指标（跳过头车，从索引1开始）
            metrics = _calculate_metrics(carlist[1:], metrics_keys)  # 传入跟驰车辆
            all_metrics[ctrl_name] = metrics

        except Exception as e:
            print(f"计算{ctrl_name}指标时出错: {str(e)}")
            all_metrics[ctrl_name] = {"error": str(e)}  # 记录错误信息

    # 3. 打印结果并保存到文件
    _print_metrics(all_metrics)
    _save_metrics(all_metrics, "metrics_storage.json")

    return all_metrics


def _calculate_metrics(carlist: List, keys: List[str]) -> Dict:
    """
    计算单组控制器的指标（最大值、最小值、平均值），剔除前10%和后10%的极端数据
    :param carlist: 车辆列表（仅包含跟驰车辆，不含头车）
    :param keys: 需要计算的指标键列表
    :return: 包含统计结果的字典
    """
    metrics = {}
    count = len(carlist)

    if count == 0:
        return {"warning": "无有效跟驰车辆数据"}

    for key in keys:
        # 1. 收集所有车辆的指标数据（过滤空列表）
        all_values = []
        for car in carlist:
            attr = getattr(car, key, [])
            if isinstance(attr, list) and len(attr) > 0:
                all_values.extend(attr)

        if not all_values:  # 处理空数据情况
            metrics[key] = {"max": None, "min": None, "avg": None}
            continue

        # 2. 排序并剔除前10%和后10%的极端值
        sorted_values = sorted(all_values)
        total_count = len(sorted_values)

        # 计算剔除的比例（10%），向上取整避免极端情况（如数据量小于10时保留大部分数据）
        trim_count = max(1, int(total_count * 0.05))  # 至少剔除1个，最多剔除10%

        # 剔除后的数据（中间80%）
        trimmed_values = sorted_values[trim_count: total_count - trim_count]

        # 处理剔除后数据为空的极端情况（如原始数据量≤2）
        if not trimmed_values:
            trimmed_values = sorted_values  # 若全部剔除后为空，则保留原始数据
            print(f"警告：{key} 数据量过少（{total_count}条），无法剔除极端值，将使用全部数据")

        # 3. 基于剔除后的数据计算统计值
        metrics[key] = {
            "max": max(trimmed_values),
            "min": min(trimmed_values),
            "avg": sum(trimmed_values) / len(trimmed_values),
            "trimmed_count": len(trimmed_values),  # 新增：剔除后的数据量
            "original_count": total_count  # 新增：原始数据量，便于追溯
        }

    return metrics


def _print_metrics(metrics: Dict) -> None:
    """打印所有控制器的指标结果"""
    for ctrl_name, ctrl_metrics in metrics.items():
        print("\n" + "=" * 50)
        print(f"{ctrl_name} 指标统计：")

        if "error" in ctrl_metrics:
            print(f"  错误: {ctrl_metrics['error']}")
            continue

        for metric, stats in ctrl_metrics.items():
            print(f"  {metric}:")
            print(f"    最大值: {stats['max']:.4f}" if stats['max'] is not None else "    最大值: 无数据")
            print(f"    最小值: {stats['min']:.4f}" if stats['min'] is not None else "    最小值: 无数据")
            print(f"    平均值: {stats['avg']:.4f}" if stats['avg'] is not None else "    平均值: 无数据")

    print("\n" + "=" * 50)


def _save_metrics(metrics: Dict, filename: str) -> None:
    """将指标保存到JSON文件"""
    try:
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=4, ensure_ascii=False)
        print(f"指标已保存到 {filename}")
    except IOError as e:
        print(f"保存文件失败: {str(e)}")