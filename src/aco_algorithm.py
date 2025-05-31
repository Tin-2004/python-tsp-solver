import numpy as np
import random
import logging # 导入 logging 模块

class AntColonyOptimizationTSP:
    def __init__(self, cities_coords, n_ants, n_iterations, alpha, beta, evaporation_rate, q=1.0, pheromone_init=0.1, start_city_idx=0, end_city_idx=None):
        self.cities_coords = np.array(cities_coords)
        self.n_cities = len(cities_coords)
        self.n_ants = n_ants
        self.n_iterations = n_iterations
        self.alpha = alpha  # 信息素影响因子
        self.beta = beta    # 启发式信息（距离）影响因子
        self.evaporation_rate = evaporation_rate # 信息素蒸发率
        self.q = q  # 信息素沉积因子
        self.pheromone_init = pheromone_init # 初始信息素水平
        self.start_city_idx = start_city_idx
        self.end_city_idx = end_city_idx if end_city_idx is not None else self.n_cities - 1

        if self.n_cities == 0:
            raise ValueError("City coordinates cannot be empty.")
        if not (0 <= self.start_city_idx < self.n_cities and 0 <= self.end_city_idx < self.n_cities):
            raise ValueError("Invalid start or end city index.")
        if self.start_city_idx == self.end_city_idx and self.n_cities > 1:
            raise ValueError("Start and end city cannot be the same for a tour of multiple cities.")


        self.distances = self._calculate_distance_matrix()
        self.pheromone = np.full((self.n_cities, self.n_cities), self.pheromone_init)
        # 启发式信息 (距离的倒数)
        self.heuristic = 1.0 / (self.distances + 1e-10) # 添加一个很小的数以避免除以零
        np.fill_diagonal(self.heuristic, 0)

        # MMAS 信息素限制
        self.pheromone_min = None
        self.pheromone_max = None

        # 获取logger实例，用于ACO算法内部的日志记录
        self.logger = logging.getLogger(__name__) # 使用模块名作为logger名


    def _calculate_distance_matrix(self):
        distances = np.zeros((self.n_cities, self.n_cities))
        for i in range(self.n_cities):
            for j in range(i + 1, self.n_cities):
                dist = np.linalg.norm(self.cities_coords[i] - self.cities_coords[j])
                distances[i, j] = distances[j, i] = dist
        return distances

    def _calculate_path_distance(self, path):
        dist = 0
        for i in range(len(path) - 1):
            dist += self.distances[path[i], path[i+1]]
        return dist

    def _construct_solution(self, ant_id):
        current_city = self.start_city_idx
        path = [current_city]
        
        # 创建待访问城市列表，初始时排除起点和终点
        cities_to_visit = list(range(self.n_cities))
        cities_to_visit.remove(self.start_city_idx)
        
        # 如果起点和终点不同且城市数量大于1，则终点城市也应从中间访问中移除，并在最后添加。
        # 对于单城市或起点即终点的情况，此逻辑可能需要调整，但问题通常 подразумевает a tour.
        must_visit_end_city = (self.start_city_idx != self.end_city_idx and self.n_cities > 1)
        if must_visit_end_city and self.end_city_idx in cities_to_visit:
            cities_to_visit.remove(self.end_city_idx)

        # 访问所有中间城市
        while len(cities_to_visit) > 0:
            probabilities = []
            total_prob = 0
            
            for next_city in cities_to_visit:
                if self.distances[current_city, next_city] > 0: # 检查路径是否存在
                    pheromone_val = self.pheromone[current_city, next_city] ** self.alpha
                    heuristic_val = self.heuristic[current_city, next_city] ** self.beta
                    prob = pheromone_val * heuristic_val
                    probabilities.append((next_city, prob))
                    total_prob += prob
                else: # 如果图是连通的且不在同一个城市，则不应发生
                    probabilities.append((next_city, 0))


            if total_prob == 0: # 没有有效的下一步，可能陷入困境或图不连通
                # 后备方案：如果还有未访问的城市，则随机选择一个，否则中断
                if cities_to_visit:
                    next_city = random.choice(cities_to_visit)
                else: # 逻辑正确则不应到达此处
                    break 
            else:
                # 根据概率选择下一个城市
                rand_val = random.uniform(0, total_prob)
                cumulative_prob = 0
                chosen = False
                for city, prob in probabilities:
                    cumulative_prob += prob
                    if cumulative_prob >= rand_val:
                        next_city = city
                        chosen = True
                        break
                if not chosen and probabilities: # 如果 total_prob > 0，则不应发生
                     next_city = probabilities[-1][0]


            path.append(next_city)
            cities_to_visit.remove(next_city)
            current_city = next_city

        # 添加终点城市以完成路径
        if must_visit_end_city:
            path.append(self.end_city_idx)
        elif self.n_cities == 1 and self.start_city_idx == self.end_city_idx: # 单城市情况
             pass # 路径就是 [start_city_idx]
        elif self.start_city_idx == self.end_city_idx and self.n_cities > 1: # 返回起点的旅行
            path.append(self.start_city_idx)


        return path

    def _update_pheromones(self, all_paths, best_path_distance_current_iter, best_overall_path, best_overall_distance):
        # 信息素蒸发
        self.pheromone *= (1 - self.evaporation_rate)

        # 由本轮或全局最优路径的蚂蚁进行信息素沉积
        # 在MMAS中，通常只有一只蚂蚁（迭代最优或全局最优）沉积信息素。
        # 这里，我们使用全局最优蚂蚁进行沉积，这是一种常见的MMAS变体。
        if best_overall_path is not None and best_overall_distance != float('inf'):
            pheromone_deposit = self.q / best_overall_distance
            for i in range(len(best_overall_path) - 1):
                self.pheromone[best_overall_path[i], best_overall_path[i+1]] += pheromone_deposit
                self.pheromone[best_overall_path[i+1], best_overall_path[i]] += pheromone_deposit # 对称TSP
        
        # 强制执行信息素轨迹限制 (MMAS 特征)
        if self.pheromone_max is None: # 在第一次更新时或根据策略初始化限制
            # 初始化MMAS信息素最大值的一种常用方法
            # 此计算假设 q=1，如果 q 是沉积量的一部分，则相应调整
            initial_global_best_estimate = best_overall_distance # 使用首次找到的全局最优值
            if initial_global_best_estimate == float('inf') and all_paths: # 如果全局最优未设置，则使用当前迭代最优值
                initial_global_best_estimate = min(d for p,d in all_paths) 
            
            if initial_global_best_estimate != float('inf'):
                self.pheromone_max = (1 / (self.evaporation_rate * initial_global_best_estimate)) * self.q # 根据 self.q 调整
                # P_min 通常与 P_max 和问题规模相关，例如 P_min = P_max / (2 * n_cities)
                # 或更复杂的公式，涉及选择非最优边的概率
                self.pheromone_min = self.pheromone_max / (2 * self.n_cities) # 简化的 P_min
                # self.pheromone_min = self.pheromone_init # 备选方案：确保其不低于初始值
                self.logger.info(f"MMAS 信息素限制已初始化: MAX={self.pheromone_max:.4f}, MIN={self.pheromone_min:.4f}")
            else:
                # 如果尚未找到有效距离的后备方案（虽然在正确初始化的情况下不太可能发生）
                self.pheromone_min = self.pheromone_init * 0.1 
                self.pheromone_max = self.pheromone_init * 10
                self.logger.warning("由于没有初始最优距离，MMAS 信息素限制使用了后备方案。")


        if self.pheromone_min is not None and self.pheromone_max is not None:
            self.pheromone = np.clip(self.pheromone, self.pheromone_min, self.pheromone_max)
        # 精英蚂蚁系统风格的信息素更新已移除，因为它不是标准的MMAS

    def run(self):
        best_overall_path = None
        best_overall_distance = float('inf')
        convergence_progress = []
        average_distances_progress = [] # 新增：记录每轮迭代的平均距离

        self.logger.info(f"初始信息素水平: {self.pheromone_init}")
        self.logger.debug(f"启发式信息 (部分样本 from city 0): {self.heuristic[0, :5]}") # 改为 debug 级别


        for iteration in range(self.n_iterations):
            all_paths_this_iteration = []
            current_iter_total_distance = 0 # 新增：用于计算本轮总距离
            current_iter_num_paths = 0    # 新增：用于计算本轮路径数量

            current_iter_best_distance = float('inf')

            for ant in range(self.n_ants):
                path = self._construct_solution(ant)
                distance = self._calculate_path_distance(path)
                
                all_paths_this_iteration.append((path, distance))
                current_iter_total_distance += distance # 新增
                current_iter_num_paths += 1             # 新增

                if distance < current_iter_best_distance:
                    current_iter_best_distance = distance

                if distance < best_overall_distance:
                    best_overall_distance = distance
                    best_overall_path = path
            
            # 计算并记录本轮平均距离
            if current_iter_num_paths > 0:
                current_iter_avg_distance = current_iter_total_distance / current_iter_num_paths
                average_distances_progress.append(current_iter_avg_distance)
            else:
                average_distances_progress.append(float('inf')) # 如果没有路径，则记录为无穷大

            # 传递 best_overall_path 和 best_overall_distance 以更新信息素
            self._update_pheromones(all_paths_this_iteration, current_iter_best_distance, best_overall_path, best_overall_distance)
            convergence_progress.append(best_overall_distance) # 记录的是到目前为止的全局最优

            if iteration % 10 == 0 or iteration == self.n_iterations - 1:
                self.logger.info(f"迭代 {iteration+1}/{self.n_iterations} - 当前轮最优: {current_iter_best_distance:.2f}, 全局最优: {best_overall_distance:.2f}, 当前轮平均: {average_distances_progress[-1]:.2f}")

        if not best_overall_path and self.n_cities > 0 : #容错处理，如果最终没有路径但是城市数量大于0
            self.logger.warning("警告: 未找到有效路径，将尝试返回一个基于启发式规则的初始路径（如果可能）。")
            # 尝试生成一个简单的路径作为备选方案。
            # 实际的备选逻辑可能需要更复杂。
            if self.n_cities == 1:
                 best_overall_path = [self.start_city_idx]
                 best_overall_distance = 0


        if best_overall_path is None and self.n_cities > 0:
             # 使用 self.logger 记录错误
             self.logger.error("ACO算法未能找到任何有效路径。")
             raise RuntimeError("ACO算法未能找到任何有效路径。")
        elif self.n_cities == 0:
             return [], 0, [], []


        return best_overall_path, best_overall_distance, convergence_progress, average_distances_progress # 返回平均距离列表

