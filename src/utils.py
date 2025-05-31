import numpy as np
import re

def load_coordinates(filepath):
    """
    从指定文件路径加载城市坐标。
    文件格式应为：{x, y}, // 任意注释
    """
    coordinates = []
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # 使用正则表达式提取花括号内的数字
            match = re.search(r'\{([\d\.\sE\+\-]+),\s*([\d\.\sE\+\-]+)\}', line)
            if match:
                try:
                    x = float(match.group(1).strip())
                    y = float(match.group(2).strip())
                    coordinates.append((x, y))
                except ValueError as e:
                    print(f"警告: 跳过无法解析的行: {line} - 错误: {e}")
            elif line.startswith("//") or not line: # 跳过注释行和空行
                continue
            else:
                print(f"警告: 跳过格式不正确的行: {line}")
    if not coordinates:
        raise ValueError("未能从文件中加载任何坐标。请检查文件格式和内容。")
    return np.array(coordinates)

def calculate_distance(coord1, coord2):
    """计算两个坐标点之间的欧氏距离"""
    return np.sqrt((coord1[0] - coord2[0])**2 + (coord1[1] - coord2[1])**2)

def total_distance(route, cities_coords):
    """
    计算给定路线的总距离。
    route: 一个城市索引的列表/数组，表示访问顺序。
    cities_coords: 包含所有城市坐标的NumPy数组。
    """
    dist = 0
    for i in range(len(route) - 1):
        dist += calculate_distance(cities_coords[route[i]], cities_coords[route[i+1]])
    return dist

def format_results(path, distance):
    result = "Optimal Path: " + " -> ".join(map(str, path)) + "\n"
    result += f"Total Distance: {distance:.2f}"
    return result