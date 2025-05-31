# File: /python-tsp-solver/python-tsp-solver/src/main.py

import sys
import os

# Add the project root directory (python-tsp-solver) to sys.path
# This ensures that imports like 'from src.module' work correctly.
# os.path.dirname(__file__) is the 'src' directory.
# os.path.join(os.path.dirname(__file__), '..') is the 'python-tsp-solver' directory.
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import numpy as np
import matplotlib.pyplot as plt
from src.algorithm import GeneticAlgorithmTSP
from src.aco_algorithm import AntColonyOptimizationTSP # Import ACO
from src.utils import load_coordinates, total_distance

def plot_route_and_convergence(coords, route, progress, title="TSP Solution"):
    """绘制路线图和收敛曲线"""
    plt.figure(figsize=(12, 6))

    # 绘制路线图
    plt.subplot(1, 2, 1)
    for i in range(len(route)):
        plt.scatter(coords[route[i], 0], coords[route[i], 1], c='blue', s=50)
        plt.text(coords[route[i], 0] + 0.5, coords[route[i], 1] + 0.5, str(route[i]), fontsize=9)

    # 绘制起点和终点标记
    plt.scatter(coords[route[0], 0], coords[route[0], 1], c='green', s=100, marker='o', label='Start City')
    plt.scatter(coords[route[-1], 0], coords[route[-1], 1], c='red', s=100, marker='x', label='End City')


    for i in range(len(route) - 1):
        start_node = route[i]
        end_node = route[i+1]
        plt.plot([coords[start_node, 0], coords[end_node, 0]],
                 [coords[start_node, 1], coords[end_node, 1]],
                 'gray', linestyle='-', linewidth=1)
    
    plt.title("Optimal Route")
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.legend()
    plt.grid(True)

    # 绘制收敛曲线
    plt.subplot(1, 2, 2)
    plt.plot(progress)
    plt.title("Convergence Curve (Distance vs. Generation)")
    plt.xlabel("Generation")
    plt.ylabel("Total Distance")
    plt.grid(True)

    plt.suptitle(title)
    plt.tight_layout(rect=[0, 0, 1, 0.96]) # 调整布局以适应总标题
    plt.show()

def main():
    # 获取当前脚本文件所在的目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # 构建数据文件的绝对路径
    data_file = os.path.join(current_dir, '..', 'data', 'coordinates.txt')

    try:
        city_coordinates = load_coordinates(data_file)
    except FileNotFoundError:
        print(f"错误: 数据文件未找到于 {data_file}")
        print("请确保 'coordinates.txt' 文件位于 'python-tsp-solver/data/' 目录下。")
        return
    except ValueError as e:
        print(f"错误: 加载坐标时出错: {e}")
        return

    num_cities = len(city_coordinates)
    if num_cities == 0:
        print("错误: 未加载任何城市坐标。")
        return
    
    print(f"成功加载 {num_cities} 个城市坐标。")

    # --- 选择算法 ---
    # algorithm_type = "GA"  # 可选 "GA" 或 "ACO"
    algorithm_type = "ACO" 

    start_city_index = 0
    end_city_index = num_cities - 1

    if algorithm_type == "GA":
        # 遗传算法参数
        population_size = 300
        elite_size = 60
        mutation_rate = 0.005
        generations = 200 # 之前是 2000

        print(f"\\n开始遗传算法优化...")
        print(f"参数设置: 种群大小={population_size}, 精英数量={elite_size}, 变异率={mutation_rate}, 迭代代数={generations}")
        print(f"起点城市索引: {start_city_index}, 终点城市索引: {end_city_index}")

        solver = GeneticAlgorithmTSP(
            cities_coords=city_coordinates,
            population_size=population_size,
            elite_size=elite_size,
            mutation_rate=mutation_rate,
            generations=generations,
            start_city_idx=start_city_index,
            end_city_idx=end_city_index
        )
        algorithm_name = "GA"
        plot_title_params = f"Pop: {population_size}, Mut: {mutation_rate}, Gen: {generations}"

    elif algorithm_type == "ACO":
        # 蚁群算法参数 (这些是初始值，可能需要调整)
        n_ants = 50          # 蚂蚁数量 (例如: 城市数量的 10%-50%)
        n_iterations = 200   # 迭代次数
        alpha = 1.0          # 信息素重要程度因子
        beta = 3.0           # 启发函数重要程度因子 (距离的倒数)
        evaporation_rate = 0.3 # 信息素挥发率
        q_val = 100.0          # 信息素强度常量
        pheromone_init_val = 0.1 # 初始信息素值

        print(f"\\n开始蚁群算法优化...")
        print(f"参数设置: 蚂蚁数量={n_ants}, 迭代次数={n_iterations}, alpha={alpha}, beta={beta}, rho={evaporation_rate}, Q={q_val}")
        print(f"起点城市索引: {start_city_index}, 终点城市索引: {end_city_index}")

        solver = AntColonyOptimizationTSP(
            cities_coords=city_coordinates,
            n_ants=n_ants,
            n_iterations=n_iterations,
            alpha=alpha,
            beta=beta,
            evaporation_rate=evaporation_rate,
            q=q_val,
            pheromone_init=pheromone_init_val,
            start_city_idx=start_city_index,
            end_city_idx=end_city_index
        )
        algorithm_name = "ACO"
        plot_title_params = f"Ants: {n_ants}, Iter: {n_iterations}, Alpha: {alpha}, Beta: {beta}, Rho: {evaporation_rate}"

    else:
        print(f"错误: 未知的算法类型 '{algorithm_type}'")
        return

    try:
        best_route_indices, best_distance, convergence_progress = solver.run()
    except ValueError as e:
        print(f"算法运行时出错: {e}")
        return
    except Exception as e: # Catching generic Exception for broader issues during development
        print(f"算法运行时发生意外错误: {e}")
        import traceback
        traceback.print_exc()
        return


    print("\\n优化完成。")
    if best_route_indices:
        print(f"最佳路径索引: {best_route_indices}")
        print(f"最短总距离: {best_distance:.4f}")
        # 绘制结果
        plot_route_and_convergence(city_coordinates, best_route_indices, convergence_progress,
                                   title=f"{algorithm_name} TSP Solution ({plot_title_params})")
    else:
        print("未能找到有效路径。")


    # --- 参数影响讨论 (可以为ACO也添加类似的讨论) ---
    if algorithm_type == "GA":
        print("\\n--- GA参数影响讨论 ---")
        # ... (保留之前的GA参数讨论)
        print("1. 种群大小 (Population Size):")
        print("   - 较小: 收敛快，但可能陷入局部最优。多样性不足。")
        print("   - 较大: 搜索范围更广，更可能找到全局最优，但计算成本高，收敛慢。")
        print("   建议: 对于130个城市，可以尝试50-200的范围。")
        print("\\n2. 精英大小 (Elite Size):")
        print("   - 较小: 更多新个体参与进化，可能跳出局部最优，但也可能丢失当前最优解。")
        print("   - 较大: 保证优秀基因的传递，加快收敛，但可能导致过早收敛，多样性降低。")
        print("   建议: 通常设置为种群大小的5%-20%。")
        print("\\n3. 变异率 (Mutation Rate):")
        print("   - 较小: 维持种群稳定性，但可能难以跳出局部最优。")
        print("   - 较大: 增加种群多样性，有助于探索新解空间，但过高可能破坏优秀基因，使算法接近随机搜索。")
        print("   建议: 通常设置一个较小的值，如0.001到0.1。")
        print("\\n4. 迭代代数 (Generations):")
        print("   - 较少: 可能未充分收敛，解的质量不高。")
        print("   - 较多: 算法有更多时间寻找更优解，但计算时间增加。超过一定代数后，解的提升可能非常缓慢。")
        print("   建议: 观察收敛曲线，当曲线趋于平缓时可以考虑停止。对于130个城市，可能需要数百到数千代。")
        print("\\n5. 交叉算子和选择策略:")
        print("   - 本例中使用的是顺序交叉 (OX1) 和精英选择+轮盘赌选择。")
        print("   - 不同的交叉算子 (如部分匹配交叉 PMX, 循环交叉 CX) 和选择策略 (如锦标赛选择) 会影响搜索效率和解的质量。")
        print("   - 例如，锦标赛选择通常比轮盘赌选择有更好的选择压力。")

    elif algorithm_type == "ACO":
        print("\\n--- ACO参数影响讨论 ---")
        print("1. 蚂蚁数量 (Number of Ants):")
        print("   - 较少: 计算速度快，但搜索的并行性不足，可能错过优良路径。")
        print("   - 较多: 增强搜索能力，有助于发现更多路径，但计算成本增加。")
        print("   建议: 通常设置为城市数量的一定比例，例如10%-50%。")
        print("\\n2. 迭代次数 (Number of Iterations):")
        print("   - 较少: 算法可能未充分收敛。")
        print("   - 较多: 允许信息素充分积累和优化，但计算时间增加。")
        print("\\n3. 信息素重要程度因子 (Alpha):")
        print("   - 较高: 蚂蚁更倾向于选择信息素浓度高的路径，可能导致过早收敛。")
        print("   - 较低: 减弱信息素的引导作用，增加路径的随机性。")
        print("\\n4. 启发函数重要程度因子 (Beta):")
        print("   - 较高: 蚂蚁更倾向于选择距离较短的路径（局部最优）。")
        print("   - 较低: 减弱局部信息的引导作用。")
        print("   Alpha 和 Beta 的相对大小决定了算法是更倾向于利用现有信息还是探索新路径。")
        print("\\n5. 信息素挥发率 (Evaporation Rate / Rho):")
        print("   - 较高: 加快旧信息的遗忘速度，有助于跳出局部最优，探索新区域。")
        print("   - 较低: 信息素保留时间长，有助于加强已发现的优良路径。过低可能导致算法停滞。")
        print("\\n6. 信息素强度常量 (Q):")
        print("   - 影响每次迭代中信息素的增加量。其值需要与路径长度和蚂蚁数量相协调。")
        print("\\n7. 初始信息素值 (Pheromone Init):")
        print("   - 初始信息素的设置会影响算法早期的搜索行为。")


    print("\\n实验建议:")
    print("   - 固定其他参数，单独改变一个参数，观察其对收敛速度和最终解质量的影响。")
    print("   - 多次运行算法（由于随机性），比较平均结果。")
    print("   - 绘制不同参数下的收敛曲线进行对比分析。")

if __name__ == "__main__":
    main()