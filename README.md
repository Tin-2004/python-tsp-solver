# Python TSP 求解器 (遗传算法与蚁群优化)

本程序使用遗传算法 (Genetic Algorithm, GA) 和蚁群优化 (Ant Colony Optimization, ACO) 来解决旅行商问题 (TSP)。
目标是找到一条从指定起点城市出发，访问所有其他城市一次，并最终到达指定终点城市的最短路径。
程序还会输出算法的执行时间、每一代/迭代的最优和平均路径长度，并可视化最终路径及收敛过程。

## 项目结构
```
python-tsp-solver
├── src
│   ├── __init__.py
│   ├── main.py
│   ├── algorithm.py         # 遗传算法实现
│   ├── aco_algorithm.py     # 蚁群优化算法实现
│   └── utils.py
├── data
│   └── coordinates.txt
├── requirements.txt
└── README.md
```

### 文件描述

- **src/__init__.py**: 将 `src` 目录标记为 Python 包。

- **src/main.py**: 程序的入口点。
    - 加载城市坐标数据。
    - 允许用户选择使用遗传算法或蚁群优化算法。
    - 初始化并运行所选算法。
    - 输出最优路径、最短距离、算法执行时间以及最后一代/迭代的平均距离。
    - 调用绘图函数展示最优路径和距离收敛曲线 (包括最优距离和平均距离)。

- **src/algorithm.py**: 包含遗传算法 (`GeneticAlgorithmTSP`) 的实现。
    - `__init__(self, cities_coords, population_size, elite_rate, mutation_rate, generations, crossover_rate, start_city_idx, end_city_idx)`: 初始化算法参数，包括城市坐标、种群大小、精英保留比例、变异率、迭代代数、交叉率、起点和终点城市索引。
    - `run(self)`: 执行遗传算法的迭代过程，返回最优路径、最短距离、每代最优距离列表和每代平均距离列表。
    - 内部方法包括个体创建、适应度计算、选择、交叉、变异等。

- **src/aco_algorithm.py**: 包含蚁群优化算法 (`AntColonyOptimizationTSP`) 的实现。
    - `__init__(self, cities_coords, n_ants, n_iterations, alpha, beta, evaporation_rate, q, pheromone_init, start_city_idx, end_city_idx)`: 初始化算法参数，包括城市坐标、蚂蚁数量、迭代次数、信息素重要程度因子 (alpha)、启发函数重要程度因子 (beta)、信息素挥发率、信息素强度常量 (Q)、初始信息素值、起点和终点城市索引。
    - `run(self)`: 执行蚁群算法的迭代过程，返回最优路径、最短距离、每次迭代最优距离列表和每次迭代平均距离列表。
    - 内部方法包括路径构建、信息素更新等。

- **src/utils.py**: 包含实用程序函数。
    - `load_coordinates(file_path)`: 从指定文件加载城市坐标。
    - `calculate_distance(coord1, coord2)`: 计算两点间的欧氏距离。
    - `total_distance(route, cities_coords)`: 计算给定路径的总距离。
    - `format_results(path, distance)`: 格式化输出结果文本 (中文)。

- **data/coordinates.txt**: 包含城市坐标数据，格式为 `{x, y}`，每行一个城市。

- **requirements.txt**: 列出了项目所需的 Python 库: `numpy` 和 `matplotlib`。

## 使用方法

1.  确保已安装 Python (推荐版本 3.6+)。
2.  克隆或下载本仓库到本地。
3.  打开终端或命令行，进入项目根目录 `python-tsp-solver`。
4.  通过运行以下命令安装所需的库:
    ```powershell
    pip install -r requirements.txt
    ```
5.  运行主程序:
    ```powershell
    python src/main.py
    ```
    程序将默认使用遗传算法执行。您可以在 `src/main.py` 文件中修改 `algorithm_type`变量来选择 "GA" 或 "ACO"。

## 算法与参数

### 通用
- **起点城市索引 (start_city_index)**: 路径的起始城市。
- **终点城市索引 (end_city_index)**: 路径的结束城市。

### 遗传算法 (GA) 参数 (`src/main.py` 中配置)
- **种群大小 (population_size)**: 每一代中个体的数量。较大的种群能更好地探索解空间，但计算成本更高。
- **精英保留比例 (elite_rate)**: 每代中直接保留到下一代的最优个体的比例。有助于保留已找到的优良解。
- **变异率 (mutation_rate)**: 个体基因发生变异的概率。适当的变异率有助于跳出局部最优，过高则可能破坏优良解。
- **交叉率 (crossover_rate)**: 进行交叉操作的概率。较高的交叉率促使父代基因的组合，产生新的个体。
- **迭代代数 (generations)**: 算法运行的总代数。更多的代数通常能带来更好的收敛，但计算时间更长。

### 蚁群优化 (ACO) 参数 (`src/main.py` 中配置)
- **蚂蚁数量 (n_ants)**: 每次迭代中蚂蚁的数量。
- **迭代次数 (n_iterations)**: 算法运行的总迭代次数。
- **信息素重要程度因子 (alpha)**: 控制路径上信息素浓度的相对重要性。
- **启发函数重要程度因子 (beta)**: 控制城市间距离的启发信息的相对重要性（通常是距离的倒数）。
- **信息素挥发率 (evaporation_rate)**: 每次迭代后信息素减少的比例。有助于避免算法过早收敛到局部最优解。
- **信息素强度常量 (q_val / Q)**: 在一次迭代中，蚂蚁在其路径上释放的信息素总量的一个参数。
- **初始信息素值 (pheromone_init_val)**: 路径上初始的信息素浓度。

## 输出与评估
程序运行后，会提供以下输出：
- **控制台输出**:
    - 所选算法及其参数。
    - 优化得到的最佳路径索引。
    - 最短总距离。
    - 算法总运行时间。
    - 最后一代（GA）或最后一次迭代（ACO）的种群/路径平均距离。
- **图形化输出**:
    - **最优路径图**: 可视化城市节点和找到的最优路径，并标记起点和终点。
    - **收敛曲线图**: 展示算法迭代过程中最优距离和平均距离的变化情况。

通过调整上述参数并观察输出结果，可以分析不同参数组合对算法性能（如收敛速度、解的质量）的影响。

---
*最后更新: 2025年5月31日*