import numpy as np
import random
import logging # 导入 logging 模块

class AntColonyOptimizationTSP:
    def __init__(self, cities_coords, n_ants, n_iterations, alpha, beta, evaporation_rate, q=1.0, pheromone_init=0.1, start_city_idx=0, end_city_idx=None): # 新增 elite_weight
        self.cities_coords = np.array(cities_coords)
        self.n_cities = len(cities_coords)
        self.n_ants = n_ants
        self.n_iterations = n_iterations
        self.alpha = alpha  # Pheromone influence
        self.beta = beta    # Heuristic (distance) influence
        self.evaporation_rate = evaporation_rate
        self.q = q  # Pheromone deposit factor
        self.pheromone_init = pheromone_init
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
        # Heuristic information (inverse of distance)
        self.heuristic = 1.0 / (self.distances + 1e-10) # Add small epsilon to avoid division by zero
        np.fill_diagonal(self.heuristic, 0)

        # MMAS Pheromone limits
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
        
        # Create a list of cities to visit, excluding start and end cities initially
        cities_to_visit = list(range(self.n_cities))
        cities_to_visit.remove(self.start_city_idx)
        
        # If start and end are different and n_cities > 1, end_city_idx should also be removed
        # from intermediate visits and added at the end.
        # If n_cities is 1, or start_city_idx == end_city_idx, this logic might need adjustment,
        # but the problem implies a tour.
        must_visit_end_city = (self.start_city_idx != self.end_city_idx and self.n_cities > 1)
        if must_visit_end_city and self.end_city_idx in cities_to_visit:
            cities_to_visit.remove(self.end_city_idx)

        # Visit all intermediate cities
        while len(cities_to_visit) > 0:
            probabilities = []
            total_prob = 0
            
            for next_city in cities_to_visit:
                if self.distances[current_city, next_city] > 0: # Check if path exists
                    pheromone_val = self.pheromone[current_city, next_city] ** self.alpha
                    heuristic_val = self.heuristic[current_city, next_city] ** self.beta
                    prob = pheromone_val * heuristic_val
                    probabilities.append((next_city, prob))
                    total_prob += prob
                else: # Should not happen if graph is connected and not on the same city
                    probabilities.append((next_city, 0))


            if total_prob == 0: # No valid next moves, might happen if stuck or graph is disconnected
                # Fallback: choose a random unvisited city if any, or break
                if cities_to_visit:
                    next_city = random.choice(cities_to_visit)
                else: # Should not be reached if logic is correct
                    break 
            else:
                # Select next city based on probabilities
                rand_val = random.uniform(0, total_prob)
                cumulative_prob = 0
                chosen = False
                for city, prob in probabilities:
                    cumulative_prob += prob
                    if cumulative_prob >= rand_val:
                        next_city = city
                        chosen = True
                        break
                if not chosen and probabilities: # Should not happen if total_prob > 0
                     next_city = probabilities[-1][0]


            path.append(next_city)
            cities_to_visit.remove(next_city)
            current_city = next_city

        # Add the end city to complete the path
        if must_visit_end_city:
            path.append(self.end_city_idx)
        elif self.n_cities == 1 and self.start_city_idx == self.end_city_idx: # Single city case
             pass # Path is just [start_city_idx]
        elif self.start_city_idx == self.end_city_idx and self.n_cities > 1: # Tour returning to start
            path.append(self.start_city_idx)


        return path

    def _update_pheromones(self, all_paths, best_path_distance_current_iter, best_overall_path, best_overall_distance):
        # Evaporation
        self.pheromone *= (1 - self.evaporation_rate)

        # Deposition by the ant that constructed the best_overall_path in this iteration or globally
        # In MMAS, typically only one ant (iteration-best or global-best) deposits pheromone.
        # Here, we'll use the global-best ant for deposition, which is a common MMAS variant.
        if best_overall_path is not None and best_overall_distance != float('inf'):
            pheromone_deposit = self.q / best_overall_distance
            for i in range(len(best_overall_path) - 1):
                self.pheromone[best_overall_path[i], best_overall_path[i+1]] += pheromone_deposit
                self.pheromone[best_overall_path[i+1], best_overall_path[i]] += pheromone_deposit # Symmetric TSP
        
        # Enforce pheromone trail limits (MMAS feature)
        if self.pheromone_max is None: # Initialize limits on first update or based on a strategy
            # A common way to initialize pheromone_max for MMAS
            # This assumes q=1 for this calculation, or adjust accordingly if q is part of the deposit amount
            initial_global_best_estimate = best_overall_distance # Use the first found global best
            if initial_global_best_estimate == float('inf') and all_paths:
                initial_global_best_estimate = min(d for p,d in all_paths) # Use current iter best if global not set
            
            if initial_global_best_estimate != float('inf'):
                self.pheromone_max = (1 / (self.evaporation_rate * initial_global_best_estimate)) * self.q # Adjusted for self.q
                # P_min is often related to P_max and problem size, e.g., P_min = P_max / (2 * n_cities)
                # Or a more complex formula involving probability of selecting non-optimal edges
                self.pheromone_min = self.pheromone_max / (2 * self.n_cities) # Simplified P_min
                # self.pheromone_min = self.pheromone_init # Alternative: ensure it doesn't drop below initial
                self.logger.info(f"MMAS Pheromone limits initialized: MAX={self.pheromone_max:.4f}, MIN={self.pheromone_min:.4f}")
            else:
                # Fallback if no valid distance found yet, though unlikely with proper initialization
                self.pheromone_min = self.pheromone_init * 0.1 
                self.pheromone_max = self.pheromone_init * 10
                self.logger.warning("MMAS Pheromone limits fallback used due to no initial best distance.")


        if self.pheromone_min is not None and self.pheromone_max is not None:
            self.pheromone = np.clip(self.pheromone, self.pheromone_min, self.pheromone_max)
        # else: # This part is removed as elite ant pheromone update is not standard MMAS
            # # Fallback to Elite Ant System style if limits are not yet defined (should not happen after first iter)
            # if best_overall_path is not None and best_overall_distance != float('inf'):
            #     elite_pheromone_deposit = self.elite_weight * (self.q / best_overall_distance)
            #     for i in range(len(best_overall_path) - 1):
            #         self.pheromone[best_overall_path[i], best_overall_path[i+1]] += elite_pheromone_deposit
            #         self.pheromone[best_overall_path[i+1], best_overall_path[i]] += elite_pheromone_deposit

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
            # current_iter_best_path = None # Not strictly needed here

            for ant in range(self.n_ants):
                path = self._construct_solution(ant)
                distance = self._calculate_path_distance(path)
                
                all_paths_this_iteration.append((path, distance))
                current_iter_total_distance += distance # 新增
                current_iter_num_paths += 1             # 新增

                if distance < current_iter_best_distance:
                    current_iter_best_distance = distance
                    # current_iter_best_path = path

                if distance < best_overall_distance:
                    best_overall_distance = distance
                    best_overall_path = path
            
            # 计算并记录本轮平均距离
            if current_iter_num_paths > 0:
                current_iter_avg_distance = current_iter_total_distance / current_iter_num_paths
                average_distances_progress.append(current_iter_avg_distance)
            else:
                average_distances_progress.append(float('inf')) # 如果没有路径，则记录为无穷大

            # 传递 best_overall_path 和 best_overall_distance
            self._update_pheromones(all_paths_this_iteration, current_iter_best_distance, best_overall_path, best_overall_distance)
            convergence_progress.append(best_overall_distance) # 记录的是到目前为止的全局最优

            if iteration % 10 == 0 or iteration == self.n_iterations - 1:
                self.logger.info(f"迭代 {iteration+1}/{self.n_iterations} - 当前轮最优: {current_iter_best_distance:.2f}, 全局最优: {best_overall_distance:.2f}, 当前轮平均: {average_distances_progress[-1]:.2f}")

        if not best_overall_path and self.n_cities > 0 : #容错处理，如果最终没有路径但是城市数量大于0
            self.logger.warning("警告: 未找到有效路径，将尝试返回一个基于启发式规则的初始路径（如果可能）。")
            # 尝试生成一个简单的路径作为备选，例如，按顺序访问或基于最近邻居的简单构造
            # 这里仅为示例，实际的备选逻辑可能需要更复杂
            if self.n_cities == 1:
                 best_overall_path = [self.start_city_idx]
                 best_overall_distance = 0
            # else:
                # # 尝试一个非常简单的顺序路径作为最后的手段
                # temp_path = [self.start_city_idx] + [c for c in range(self.n_cities) if c != self.start_city_idx and c != self.end_city_idx] + ([self.end_city_idx] if self.start_city_idx != self.end_city_idx else [])
                # if len(set(temp_path)) == self.n_cities : #确保路径包含了所有城市且不重复（除了首尾可能相同）
                #     best_overall_path = temp_path
                #     best_overall_distance = self._calculate_path_distance(best_overall_path)


        if best_overall_path is None and self.n_cities > 0:
             # 使用 self.logger 记录错误
             self.logger.error("ACO算法未能找到任何有效路径。")
             raise RuntimeError("ACO算法未能找到任何有效路径。")
        elif self.n_cities == 0:
             return [], 0, [], []


        return best_overall_path, best_overall_distance, convergence_progress, average_distances_progress # 返回平均距离列表

