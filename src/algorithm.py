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
import random
from src.utils import total_distance

class GeneticAlgorithmTSP:
    def __init__(self, cities_coords, population_size, elite_rate, mutation_rate, generations, crossover_rate=0.9, start_city_idx=0, end_city_idx=None): # elite_size -> elite_rate
        self.cities_coords = cities_coords
        self.num_cities = len(cities_coords)
        self.population_size = population_size
        self.elite_rate = elite_rate # 新增：精英比例
        # 根据精英比例计算精英数量，确保至少为0
        self.elite_size = max(0, int(self.population_size * self.elite_rate))
        # 如果设置了精英比例但计算出的精英数量为0（由于种群太小或比例太低），并且种群大于0，则至少保留一个精英
        if self.elite_rate > 0 and self.elite_size == 0 and self.population_size > 0:
            self.elite_size = 1
            print(f"提示: 由于种群大小 ({self.population_size}) 和精英比例 ({self.elite_rate}) 导致计算精英数量为0, 已自动设置为保留1个精英。")


        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.generations = generations
        self.start_city_idx = start_city_idx
        # 如果未指定 end_city_idx，则默认为最后一个城市
        self.end_city_idx = end_city_idx if end_city_idx is not None else self.num_cities - 1

        if self.num_cities <= 2:
            raise ValueError("城市数量必须大于2才能进行路径规划。")

        # 需要排序的城市索引 (排除起点和终点)
        self.cities_to_permute_indices = [i for i in range(self.num_cities) if i != self.start_city_idx and i != self.end_city_idx]
        self.num_cities_to_permute = len(self.cities_to_permute_indices)


    def _create_individual(self):
        """创建一个代表中间城市访问顺序的个体"""
        if self.num_cities_to_permute == 0: # 只有起点和终点
            return []
        return random.sample(self.cities_to_permute_indices, self.num_cities_to_permute)

    def _create_initial_population(self):
        population = []
        for _ in range(self.population_size):
            population.append(self._create_individual())
        return population

    def _get_full_route(self, individual_perm):
        """根据个体的中间城市排列，构建从起点到终点的完整路径"""
        if not individual_perm: # 如果 individual_perm 为空 (例如只有2个城市)
             return [self.start_city_idx, self.end_city_idx]
        return [self.start_city_idx] + individual_perm + [self.end_city_idx]

    def _calculate_fitness(self, individual_perm):
        """计算个体的适应度（总距离的倒数）"""
        route = self._get_full_route(individual_perm)
        return 1 / total_distance(route, self.cities_coords)

    def _rank_routes(self, population):
        """根据适应度对种群进行排序"""
        fitness_results = {}
        for i in range(len(population)):
            fitness_results[i] = self._calculate_fitness(population[i])
        return sorted(fitness_results.items(), key=lambda x: x[1], reverse=True)

    def _selection(self, ranked_population):
        """选择操作（精英选择 + 轮盘赌选择）"""
        selection_results = []
        # 精英选择
        for i in range(self.elite_size):
            selection_results.append(ranked_population[i][0]) # 添加精英个体的索引

        # 轮盘赌选择剩余个体
        df = np.array([ranked_population[i][1] for i in range(len(ranked_population))])
        df_sum = np.sum(df)
        if df_sum == 0: # 避免除以零，如果所有适应度都是0
            # 如果所有适应度为0，则随机选择
            for _ in range(len(ranked_population) - self.elite_size):
                selection_results.append(random.choice(range(len(ranked_population))))
        else:
            cumulative_sum = np.cumsum(df / df_sum)
            for _ in range(len(ranked_population) - self.elite_size):
                pick = random.random()
                for i in range(len(ranked_population)):
                    if pick <= cumulative_sum[i]:
                        selection_results.append(ranked_population[i][0])
                        break
        return selection_results

    def _mating_pool(self, population, selection_results):
        """创建交配池"""
        matingpool = []
        for i in range(len(selection_results)):
            index = selection_results[i]
            matingpool.append(population[index])
        return matingpool

    def _crossover(self, parent1, parent2):
        """交叉操作（顺序交叉 OX1）"""
        child = []
        child_p1 = []
        child_p2 = []

        if not parent1 or not parent2: # 如果父代为空（例如只有2个城市的情况）
            return []

        gene_a = int(random.random() * len(parent1))
        gene_b = int(random.random() * len(parent1))

        start_gene = min(gene_a, gene_b)
        end_gene = max(gene_a, gene_b)

        for i in range(start_gene, end_gene):
            child_p1.append(parent1[i])

        child_p2 = [item for item in parent2 if item not in child_p1]

        child = child_p1 + child_p2
        return child

    def _breed_population(self, matingpool):
        """繁衍新一代种群"""
        children = []
        # length 是需要通过交叉或复制产生的非精英子代的数量
        length = self.population_size - self.elite_size 
        
        # matingpool 包含了被选中的个体，精英个体通常排在前面（如果已排序）
        # pool 是 matingpool 的一个打乱顺序的副本，用于从中选择父代
        pool = random.sample(matingpool, len(matingpool))

        # 1. 保留精英个体到下一代 (作为副本)
        for i in range(self.elite_size):
            # 假设 matingpool 的前 elite_size 个是精英
            children.append(list(matingpool[i])) 

        # 2. 通过交叉或复制产生剩余的 length 个子代
        for i in range(length):
            # 从打乱的父代池 pool 中选择父代
            # 使用原始的配对策略: pool[i] 和 pool[len(pool)-1-i]
            # 确保索引有效，尽管对于此特定配对，如果 length 合理，通常是有效的
            parent1 = pool[i % len(pool)] # 使用模运算以防万一
            parent2 = pool[(len(pool)-1-i) % len(pool)]

            if random.random() < self.crossover_rate:
                child = self._crossover(parent1, parent2)
            else:
                # 如果不交叉，则从两个父代中随机选择一个作为子代 (的副本)
                child = list(random.choice([parent1, parent2]))
            children.append(child)
        return children

    def _mutate(self, individual):
        """变异操作（交换变异）"""
        if not individual or len(individual) < 2: # 如果个体为空或长度小于2，则不变异
            return individual
        for swapped in range(len(individual)):
            if random.random() < self.mutation_rate:
                swap_with = int(random.random() * len(individual))
                city1 = individual[swapped]
                city2 = individual[swap_with]
                individual[swapped] = city2
                individual[swap_with] = city1
        return individual

    def _mutate_population(self, population):
        """对整个种群进行变异"""
        mutated_pop = []
        for ind in range(len(population)):
            mutated_ind = self._mutate(population[ind])
            mutated_pop.append(mutated_ind)
        return mutated_pop

    def run(self):
        population = self._create_initial_population()
        best_overall_route_perm = None
        best_overall_distance = float('inf')
        convergence_progress = [] # 记录每代的最优距离
        average_distances_progress = [] # 新增：记录每代的平均距离

        print(f"开始遗传算法，总共 {self.generations} 代，种群大小 {self.population_size}")

        for generation in range(self.generations):
            # 在每一代开始时，我们有当前的 population
            current_generation_population = population # 使用明确的变量名

            ranked_pop = self._rank_routes(current_generation_population)
            current_gen_best_fitness = ranked_pop[0][1]
            current_gen_best_distance = 1 / current_gen_best_fitness

            # 计算并记录当前代的平均距离
            current_gen_total_distance = 0
            if len(current_generation_population) > 0:
                for individual_perm in current_generation_population:
                    full_route = self._get_full_route(individual_perm)
                    current_gen_total_distance += total_distance(full_route, self.cities_coords)
                current_gen_avg_distance = current_gen_total_distance / len(current_generation_population)
            else:
                current_gen_avg_distance = float('inf')
            average_distances_progress.append(current_gen_avg_distance)

            if current_gen_best_distance < best_overall_distance:
                best_overall_distance = current_gen_best_distance
                best_overall_route_perm = current_generation_population[ranked_pop[0][0]]
            
            convergence_progress.append(best_overall_distance) # 记录的是到目前为止的全局最优

            if generation % 10 == 0 or generation == self.generations - 1:
                print(f"代 {generation+1}/{self.generations} - 当前代最优: {current_gen_best_distance:.2f}, 全局最优: {best_overall_distance:.2f}, 当前代平均: {current_gen_avg_distance:.2f}")

            selected_indices = self._selection(ranked_pop)
            matingpool = self._mating_pool(current_generation_population, selected_indices)
            children = self._breed_population(matingpool)
            population = self._mutate_population(children) # 更新 population 以供下一代使用
        
        best_full_route = self._get_full_route(best_overall_route_perm if best_overall_route_perm is not None else self._create_individual()) 
        if best_overall_route_perm is None and self.num_cities > 0:
            print("警告: GA未能确定最优个体排列，将使用随机个体构建最终路径。")
            if not best_full_route or len(best_full_route) < 2 :
                if self.num_cities == 2:
                    best_full_route = [self.start_city_idx, self.end_city_idx]
                    best_overall_distance = total_distance(best_full_route, self.cities_coords)
                else: 
                    raise RuntimeError("GA算法未能找到任何有效路径。") 
        elif self.num_cities == 0:
            return [], 0, [], []

        return best_full_route, best_overall_distance, convergence_progress, average_distances_progress