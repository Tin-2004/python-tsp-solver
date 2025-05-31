# Python TSP Solver using Genetic Algorithm

本程序使用遗传算法解决旅行商问题 (TSP)。
目标是找到一条从指定起点城市出发，访问所有其他城市一次，并最终到达指定终点城市的最短路径。

## 项目结构
```
python-tsp-solver
├── src
│   ├── __init__.py
│   ├── main.py
│   ├── algorithm.py
│   └── utils.py
├── data
│   └── coordinates.txt
├── requirements.txt
└── README.md
```

### 文件描述

- **src/__init__.py**: 将 `src` 目录标记为 Python 包。此文件故意留空。

- **src/main.py**: 程序的入口点。它读取城市坐标数据，调用算法解决 TSP 问题，并输出结果。

- **src/algorithm.py**: 实现解决 TSP 问题的智能算法（例如，遗传算法，蚁群优化）。它包含 `TSPSolver` 类，具有以下方法：
  - `__init__(self, coordinates)`: 初始化城市坐标。
  - `solve(self)`: 执行算法并返回最短路径和总距离。
  - `evaluate(self, path)`: 计算给定路径的总距离。

- **src/utils.py**: 包含实用程序函数，例如加载坐标数据和格式化输出结果。它包括：
  - `load_coordinates(file_path)`: 从指定文件加载城市坐标。
  - `format_results(path, distance)`: 格式化输出结果。

- **data/coordinates.txt**: 包含 130 个城市的坐标，格式为 `{x, y}`，每行一个城市。

- **requirements.txt**: 列出了项目所需的 Python 库，例如 `numpy` 和 `matplotlib`，用于数值计算和可视化。

## 使用方法

1. 通过运行以下命令安装所需的库：
   ```
   pip install -r requirements.txt
   ```

2. 运行主程序：
   ```
   python src/main.py
   ```

## 参数设置分析

算法的性能和收敛性会受到各种参数设置的显著影响。例如：

- **种群规模**：在遗传算法中，较大的种群可能导致更好的解空间探索，但会增加计算时间。
- **变异率**：较高的变异率可以帮助避免局部最小值，但也可能破坏良好的解决方案。
- **代数**：更多的代数允许更好的收敛，但需要更多的计算资源。

通过对这些参数进行实验，可以帮助确定最佳设置，以实现解决 TSP 的最佳结果。

2025年5月31日