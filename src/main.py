# File: /python-tsp-solver/python-tsp-solver/src/main.py

import sys
import os
import time # 导入 time 模块
import logging # 导入 logging 模块
from datetime import datetime # 导入 datetime 模块

# 将项目根目录 (python-tsp-solver) 添加到 sys.path
# 这样可以确保 'from src.module' 这样的导入能够正确工作。
# os.path.dirname(__file__) 是 'src' 目录。
# os.path.join(os.path.dirname(__file__), '..') 是 'python-tsp-solver' 目录。
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import numpy as np
import matplotlib.pyplot as plt
from src.algorithm import GeneticAlgorithmTSP
from src.aco_algorithm import AntColonyOptimizationTSP # 导入蚁群算法
from src.utils import load_coordinates, total_distance

# 全局变量，用于存储日志文件名，方便后续绘图函数引用
current_log_file_name = ""

def setup_logging(algorithm_name): # 添加 algorithm_name 参数
    """配置日志记录器，将日志同时输出到控制台和文件。"""
    global current_log_file_name
    log_dir = "logs"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # 使用 algorithm_name 参数动态生成文件名
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    current_log_file_name = f"{timestamp}_{algorithm_name}.log" # 使用 algorithm_name
    log_file_path = os.path.join(log_dir, current_log_file_name)

    logger = logging.getLogger() # 获取根记录器
    logger.setLevel(logging.INFO) # 设置全局日志级别

    # 清除已存在的处理器，防止重复记录 (尤其是在交互式环境中多次运行时)
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    # 文件处理器 - 记录详细信息
    file_handler = logging.FileHandler(log_file_path, mode='w', encoding='utf-8')
    file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(module)s - %(message)s')
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    # 控制台处理器 - 记录简洁信息，模拟 print
    console_handler = logging.StreamHandler(sys.stdout)
    console_formatter = logging.Formatter('%(message)s') # 只输出消息
    console_handler.setFormatter(console_formatter)
    # 为了让控制台也能看到 INFO 级别以上的日志，可以为控制台处理器单独设置级别
    # console_handler.setLevel(logging.INFO) 
    logger.addHandler(console_handler)

    logging.info(f"日志功能已启动。日志文件: {log_file_path}")

def plot_route_and_convergence(coords, route, progress, average_progress=None, title="TSP 解决方案"):
    """绘制路线图和收敛曲线
    
    Args:
        coords: 城市坐标
        route: 最优路径
        progress: 最优距离的收敛过程列表
        average_progress: 平均距离的收敛过程列表 (可选)
        title: 图表总标题
    """
    # 解决 matplotlib 中文乱码问题
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体为黑体
    plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题

    plt.figure(figsize=(12, 6))

    # 绘制路线图
    plt.subplot(1, 2, 1)
    for i in range(len(route)):
        plt.scatter(coords[route[i], 0], coords[route[i], 1], c='blue', s=50)
        plt.text(coords[route[i], 0] + 0.5, coords[route[i], 1] + 0.5, str(route[i]), fontsize=9)

    # 绘制起点和终点标记
    plt.scatter(coords[route[0], 0], coords[route[0], 1], c='green', s=100, marker='o', label='起点城市')
    plt.scatter(coords[route[-1], 0], coords[route[-1], 1], c='red', s=100, marker='x', label='终点城市')


    for i in range(len(route) - 1):
        start_node = route[i]
        end_node = route[i+1]
        plt.plot([coords[start_node, 0], coords[end_node, 0]],
                 [coords[start_node, 1], coords[end_node, 1]],
                 'gray', linestyle='-', linewidth=1)
    
    plt.title("最优路径")
    plt.xlabel("X 坐标")
    plt.ylabel("Y 坐标")
    plt.legend()
    plt.grid(True)

    # 绘制收敛曲线
    plt.subplot(1, 2, 2)
    plt.plot(progress, label='最优距离') # 添加标签
    if average_progress is not None:
        plt.plot(average_progress, label='平均距离', linestyle='--') # 绘制平均距离并添加标签
    plt.title("收敛曲线 (距离 vs. 迭代/代数)")
    plt.xlabel("迭代/代数")
    plt.ylabel("总距离")
    plt.legend() # 显示图例
    plt.grid(True)

    plt.suptitle(title)
    plt.tight_layout(rect=[0, 0, 1, 0.96]) # 调整布局以适应总标题
    plt.show()

def main():
    # --- 用户选择算法 ---
    # 请在此处取消注释您希望运行的算法类型，或修改为 input() 方式动态选择:
    # algorithm_type = "GA"  # 遗传算法
    algorithm_type = "ACO" # 蚁群算法
    # 例如，若要动态选择:
    # chosen_algo = input("请选择算法类型 (GA 或 ACO): ").strip().upper()
    # while chosen_algo not in ["GA", "ACO"]:
    #     print("无效的输入，请输入 GA 或 ACO.")
    #     chosen_algo = input("请选择算法类型 (GA 或 ACO): ").strip().upper()
    # algorithm_type = chosen_algo

    # --- 配置日志 ---
    # 确保 setup_logging 在 algorithm_type 确定后调用
    setup_logging(algorithm_type) 
    logger = logging.getLogger(__name__) # 获取在setup_logging中配置的logger
    logger.info(f"选择的算法: {algorithm_type}")

    # 获取当前脚本文件所在的目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # 构建数据文件的绝对路径
    data_file = os.path.join(current_dir, '..', 'data', 'coordinates.txt')

    try:
        city_coordinates = load_coordinates(data_file)
    except FileNotFoundError:
        logger.error(f"错误: 数据文件未找到于 {data_file}") # 使用 logger 实例
        logger.error("请确保 'coordinates.txt' 文件位于 'python-tsp-solver/data/' 目录下。") # 使用 logger 实例
        return
    except ValueError as e:
        logger.error(f"错误: 加载坐标时出错: {e}") # 使用 logger 实例
        return

    num_cities = len(city_coordinates)
    if num_cities == 0:
        logger.error("错误: 未加载任何城市坐标。") # 使用 logger 实例
        return
    
    logger.info(f"成功加载 {num_cities} 个城市坐标。")

    # --- 定义起点和终点 ---
    start_city_index = 0
    end_city_index = num_cities - 1 # 默认最后一个城市为终点

    if algorithm_type == "GA":
        # 遗传算法参数
        population_size = 100       # 种群大小
        elite_rate = 0.1            # 精英保留比例 (例如 0.1 表示保留10%的精英)
        mutation_rate = 0.05       # 变异率
        crossover_rate = 0.8        # 交叉概率
        generations = 2000           # 迭代代数
        tournament_size = 3         # 锦标赛选择的锦标赛大小

        logging.info(f"\\n开始遗传算法优化...")
        # 更新打印信息以反映精英比例和锦标赛大小
        logging.info(f"参数设置: 种群大小={population_size}, 精英保留比例={elite_rate}, 变异率={mutation_rate}, 交叉率={crossover_rate}, 迭代代数={generations}, 锦标赛大小={tournament_size}")
        logging.info(f"起点城市索引: {start_city_index}, 终点城市索引: {end_city_index}")

        solver = GeneticAlgorithmTSP(
            cities_coords=city_coordinates,
            population_size=population_size,
            elite_rate=elite_rate, # 传递精英比例
            mutation_rate=mutation_rate,
            crossover_rate=crossover_rate, 
            generations=generations,
            start_city_idx=start_city_index,
            end_city_idx=end_city_index,
            tournament_size=tournament_size # 传递锦标赛大小
        )
        algorithm_name = "遗传算法"
        # 更新绘图标题参数以反映精英比例和锦标赛大小
        plot_title_params = f"种群: {population_size}, 精英比例: {elite_rate}, 变异率: {mutation_rate}, 交叉率: {crossover_rate}, 代数: {generations}, 锦标赛: {tournament_size}"

    elif algorithm_type == "ACO":
        # ACO 参数 (保持用户上次设定的优化参数)
        n_ants_aco = 50           # 蚂蚁数量
        n_iterations_aco = 200    # 迭代次数 (用户指定)
        alpha_aco = 1.0           # 信息素重要程度因子
        beta_aco = 5.0            # 启发函数重要程度因子 (用户指定)
        evaporation_rate_aco = 0.7 # 信息素挥发率 (调整以增加探索)
        q_aco = 100               # 信息素强度常数
        pheromone_init_aco = 0.1  # 初始信息素浓度
        elite_weight_aco = 1.5    # 精英蚂蚁权重 (调整以减少最优路径主导)

        algorithm_name = f"ACO (精英策略, 蚂蚁数={n_ants_aco}, 迭代={n_iterations_aco}, alpha={alpha_aco}, beta={beta_aco}, evap={evaporation_rate_aco}, elite_w={elite_weight_aco})"
        logger.info(f"ACO 参数: 蚂蚁数量={n_ants_aco}, 迭代次数={n_iterations_aco}, alpha={alpha_aco}, beta={beta_aco}, 挥发率={evaporation_rate_aco}, Q={q_aco}, 初始信息素={pheromone_init_aco}, 精英权重={elite_weight_aco}")
        
        solver = AntColonyOptimizationTSP(
            cities_coords=city_coordinates,
            n_ants=n_ants_aco,
            n_iterations=n_iterations_aco,
            alpha=alpha_aco,
            beta=beta_aco,
            evaporation_rate=evaporation_rate_aco,
            q=q_aco,
            pheromone_init=pheromone_init_aco,
            start_city_idx=start_city_index,
            end_city_idx=end_city_index,
            elite_weight=elite_weight_aco # 传递精英权重
        )
        plot_title_params = f"蚂蚁: {n_ants_aco}, 迭代: {n_iterations_aco}, Alpha: {alpha_aco}, Beta: {beta_aco}, Rho: {evaporation_rate_aco}, 精英权重: {elite_weight_aco}"

    else:
        logger.error(f"错误: 未知的算法类型 '{algorithm_type}'") # 使用 logger 实例
        return

    try:
        start_time = time.time() # 记录开始时间
        # 修改此处以接收四个返回值
        best_route_indices, best_distance, convergence_progress, average_distances_progress = solver.run()
        end_time = time.time() # 记录结束时间
        execution_time = end_time - start_time # 计算运行时间
    except ValueError as e:
        logger.error(f"算法运行时出错: {e}") # 使用 logger 实例
        return
    except Exception as e: # 捕获开发过程中可能出现的更广泛的异常
        logger.exception(f"算法运行时发生意外错误:") # logger.exception 记录异常信息
        return

    logger.info("\\n优化完成。") # 使用 logger 实例
    if best_route_indices:
        logger.info(f"最佳路径索引: {best_route_indices}") # 使用 logger 实例
        logger.info(f"最短总距离: {best_distance:.4f}") # 使用 logger 实例
        logger.info(f"算法运行时间: {execution_time:.4f} 秒")  # 使用 logger 实例
        if average_distances_progress:
            logger.info(f"最后一代/迭代的平均距离: {average_distances_progress[-1]:.4f}") # 使用 logger 实例
        # 绘制结果，传入平均距离数据
        plot_route_and_convergence(city_coordinates, best_route_indices, convergence_progress, average_distances_progress,
                                   title=f"{algorithm_name} TSP 解决方案 ({plot_title_params})")
    else:
        logger.info("未能找到有效路径。") # 使用 logger 实例

if __name__ == "__main__":
    main()