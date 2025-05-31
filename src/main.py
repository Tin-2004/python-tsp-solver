# File: /python-tsp-solver/python-tsp-solver/src/main.py

import numpy as np
import matplotlib.pyplot as plt
from src.algorithm import GeneticAlgorithmTSP  # Removed TSPSolver
from src.utils import load_coordinates, format_results, total_distance
import os

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

    # 遗传算法参数
    population_size = 100  # 种群大小
    elite_size = 20      # 精英个体数量 (保留多少最优个体到下一代)
    mutation_rate = 0.01 # 变异率
    generations = 200    # 迭代代数
    
    # 要求从第一个城市出发 (索引为0)，在最后一个城市结束 (索引为 num_cities - 1)
    start_city_index = 0
    end_city_index = num_cities - 1

    print(f"\n开始遗传算法优化...")
    print(f"参数设置: 种群大小={population_size}, 精英数量={elite_size}, 变异率={mutation_rate}, 迭代代数={generations}")
    print(f"起点城市索引: {start_city_index}, 终点城市索引: {end_city_index}")

    ga_tsp = GeneticAlgorithmTSP(
        cities_coords=city_coordinates,
        population_size=population_size,
        elite_size=elite_size,
        mutation_rate=mutation_rate,
        generations=generations,
        start_city_idx=start_city_index,
        end_city_idx=end_city_index
    )

    try:
        best_route_indices, best_distance, convergence_progress = ga_tsp.run()
    except ValueError as e:
        print(f"算法运行时出错: {e}")
        return

    print("\n优化完成。")
    print(f"最佳路径索引: {best_route_indices}")
    print(f"最短总距离: {best_distance:.4f}")

    # 绘制结果
    plot_route_and_convergence(city_coordinates, best_route_indices, convergence_progress,
                               title=f"GA TSP Solution (Pop: {population_size}, Mut: {mutation_rate}, Gen: {generations})")

    # --- 参数影响讨论 ---
    print("\n--- 不同参数设置对收敛性的影响讨论 ---")
    print("1. 种群大小 (Population Size):")
    print("   - 较小: 收敛快，但可能陷入局部最优。多样性不足。")
    print("   - 较大: 搜索范围更广，更可能找到全局最优，但计算成本高，收敛慢。")
    print("   建议: 对于130个城市，可以尝试50-200的范围。")
    print("\n2. 精英大小 (Elite Size):")
    print("   - 较小: 更多新个体参与进化，可能跳出局部最优，但也可能丢失当前最优解。")
    print("   - 较大: 保证优秀基因的传递，加快收敛，但可能导致过早收敛，多样性降低。")
    print("   建议: 通常设置为种群大小的5%-20%。")
    print("\n3. 变异率 (Mutation Rate):")
    print("   - 较小: 维持种群稳定性，但可能难以跳出局部最优。")
    print("   - 较大: 增加种群多样性，有助于探索新解空间，但过高可能破坏优秀基因，使算法接近随机搜索。")
    print("   建议: 通常设置一个较小的值，如0.001到0.1。")
    print("\n4. 迭代代数 (Generations):")
    print("   - 较少: 可能未充分收敛，解的质量不高。")
    print("   - 较多: 算法有更多时间寻找更优解，但计算时间增加。超过一定代数后，解的提升可能非常缓慢。")
    print("   建议: 观察收敛曲线，当曲线趋于平缓时可以考虑停止。对于130个城市，可能需要数百到数千代。")
    print("\n5. 交叉算子和选择策略:")
    print("   - 本例中使用的是顺序交叉 (OX1) 和精英选择+轮盘赌选择。")
    print("   - 不同的交叉算子 (如部分匹配交叉 PMX, 循环交叉 CX) 和选择策略 (如锦标赛选择) 会影响搜索效率和解的质量。")
    print("   - 例如，锦标赛选择通常比轮盘赌选择有更好的选择压力。")
    print("\n实验建议:")
    print("   - 固定其他参数，单独改变一个参数，观察其对收敛速度和最终解质量的影响。")
    print("   - 多次运行算法（由于随机性），比较平均结果。")
    print("   - 绘制不同参数下的收敛曲线进行对比分析。")

if __name__ == "__main__":
    main()