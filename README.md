# Python TSP 求解器 (遗传算法与蚁群优化)

本程序利用遗传算法 (GA) 和蚁群优化 (ACO) 解决旅行商问题 (TSP)，旨在找到从指定起点到终点、访问所有城市一次的最短路径。程序会输出关键性能指标，可视化最优路径与收敛过程，并自动保存运行日志和结果图像。

## 项目结构
```
python-tsp-solver
├── src
│   ├── __init__.py
│   ├── main.py              # 程序主入口
│   ├── algorithm.py         # 遗传算法 (GA) 实现
│   ├── aco_algorithm.py     # 蚁群优化 (ACO) 实现
│   └── utils.py             # 辅助函数
├── data
│   └── coordinates.txt      # 城市坐标数据
├── logs
│   └── *.log                # 运行日志文件
├── images
│   └── *.png                # 保存的图像文件
├── requirements.txt         # 依赖库
└── README.md                # 项目说明文档
```

### 文件描述

- **src/__init__.py**: 将 `src` 目录标记为 Python 包。
- **src/main.py**: 程序入口。负责加载数据，用户选择算法，执行算法，输出结果，并调用绘图函数。会自动将运行日志保存在 `logs` 目录下，并将结果图（最优路径图、收敛曲线图）保存为图片文件到 `images` 目录下。
- **src/algorithm.py**: `GeneticAlgorithmTSP` 类实现。
    - `__init__`: 初始化GA参数。
    - `run`: 执行GA迭代，返回结果。
    - 核心逻辑：个体创建、适应度计算、锦标赛选择、交叉、变异。
- **src/aco_algorithm.py**: `AntColonyOptimizationTSP` 类实现。
    - `__init__`: 初始化ACO参数。
    - `run`: 执行ACO迭代，返回结果。
    - 核心逻辑：路径构建、信息素更新 (集成MMAS特性，如信息素限制和精英策略)。
- **src/utils.py**: 辅助函数，如坐标加载、距离计算、结果格式化等。
- **data/coordinates.txt**: 存储城市坐标，每行格式为 `{x, y}`。
- **logs/**: 存放程序运行日志，文件名包含算法类型和时间戳。
- **images/**: 存放程序生成的图像文件（如最优路径图、收敛曲线图），文件名包含算法类型和时间戳。
- **requirements.txt**: 项目依赖：`numpy`, `matplotlib`。

## 使用方法

1.  确保 Python (推荐 3.6+) 已安装。
2.  克隆或下载本仓库。
3.  进入项目根目录 `python-tsp-solver`。
4.  安装依赖:
    ```powershell
    pip install -r requirements.txt
    ```
5.  运行程序:
    ```powershell
    python src/main.py
    ```
    默认使用遗传算法。可在 `src/main.py` 中修改 `algorithm_type` 为 "GA" 或 "ACO"。

## 算法与参数

### 通用参数
- **起点城市索引 (start_city_index)**
- **终点城市索引 (end_city_index)**

### 遗传算法 (GA)
- **种群大小 (population_size)**
- **精英保留比例 (elite_rate)**
- **变异率 (mutation_rate)**
- **交叉率 (crossover_rate)**
- **迭代代数 (generations)**
- **选择策略**: **锦标赛选择** (优化自轮盘赌)。

### 蚁群优化 (ACO)
- **蚂蚁数量 (n_ants)**
- **迭代次数 (n_iterations)**
- **信息素影响因子 (alpha)**
- **启发式信息影响因子 (beta)**
- **信息素蒸发率 (evaporation_rate)**
- **信息素沉积因子 (q_val / Q)**
- **初始信息素 (pheromone_init_val)**
- **算法增强**: 集成**最大最小蚂蚁系统 (MMAS)** 特性 (信息素上下限、精英更新策略)。

## 输出与评估
程序运行后，将提供以下信息，并通过日志和图像文件进行持久化：
- **控制台输出与日志文件 (`logs/*.log`)**:
    - 选定算法及参数配置。
    - 最优路径的城市索引序列。
    - 最短路径总长度。
    - 算法总执行耗时。
    - 最后一代 (GA) 或最后迭代 (ACO) 的平均路径长度。
- **图形化输出 (自动保存为图像文件到 `images/*.png`)**:
    - **最优路径图**: 可视化城市布局、最优路径，并高亮起点和终点。
    - **收敛曲线图**: 展示算法在迭代过程中最优距离和平均距离的演变趋势。

通过分析这些输出（包括保存的日志和图像），可以评估不同参数对算法性能（如收敛速度、解的质量）的影响。

---
*最后更新: 2025年5月31日*