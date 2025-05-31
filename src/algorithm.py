import numpy as np
import random
from src.utils import total_distance

class GeneticAlgorithmTSP:
    def __init__(self, cities_coords, population_size, elite_size, mutation_rate, generations, start_city_idx=0, end_city_idx=None):
        self.cities_coords = cities_coords
        self.num_cities = len(cities_coords)
        self.population_size = population_size
        self.elite_size = elite_size # 保留最优个体的数量
        self.mutation_rate = mutation_rate
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
        length = len(matingpool) - self.elite_size
        pool = random.sample(matingpool, len(matingpool)) # 打乱交配池

        # 保留精英个体到下一代
        for i in range(self.elite_size):
            children.append(matingpool[i])

        # 交叉产生新的子代
        for i in range(length):
            child = self._crossover(pool[i], pool[len(matingpool)-1-i])
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
        """运行遗传算法"""
        if self.num_cities_to_permute == 0: # 特殊情况：只有起点和终点
            best_perm = []
            best_route = self._get_full_route(best_perm)
            best_distance = total_distance(best_route, self.cities_coords)
            print(f"只有起点和终点，无需优化。")
            print(f"最佳路径: {best_route}")
            print(f"最短距离: {best_distance}")
            return best_route, best_distance, []

        pop = self._create_initial_population()
        
        # 存储每一代的最佳距离，用于绘图
        progress = []
        
        # 初始化全局最优解
        initial_ranked_pop = self._rank_routes(pop)
        overall_best_perm_idx = initial_ranked_pop[0][0]
        overall_best_perm = pop[overall_best_perm_idx]
        overall_best_route = self._get_full_route(overall_best_perm)
        overall_best_distance = total_distance(overall_best_route, self.cities_coords)
        progress.append(overall_best_distance)

        print(f"初始种群最佳距离: {overall_best_distance}")

        for i in range(self.generations):
            ranked_pop = self._rank_routes(pop)
            selection_results = self._selection(ranked_pop)
            matingpool = self._mating_pool(pop, selection_results)
            children = self._breed_population(matingpool)
            next_generation = self._mutate_population(children)
            pop = next_generation

            current_gen_best_perm_idx = self._rank_routes(pop)[0][0]
            current_gen_best_perm = pop[current_gen_best_perm_idx]
            current_gen_best_route = self._get_full_route(current_gen_best_perm)
            current_gen_best_distance = total_distance(current_gen_best_route, self.cities_coords)
            progress.append(current_gen_best_distance) # 记录当前代最佳距离用于收敛曲线

            # 更新全局最优解
            if current_gen_best_distance < overall_best_distance:
                overall_best_distance = current_gen_best_distance
                overall_best_perm = current_gen_best_perm
            
            if (i + 1) % 10 == 0 or i == 0 : # 每10代打印一次信息 (使用当前代最佳)
                print(f"第 {i+1} 代: 当前代最佳距离 = {current_gen_best_distance:.2f}, 全局最佳距离 = {overall_best_distance:.2f}")
        
        # 返回全局最优解
        final_best_route = self._get_full_route(overall_best_perm)
        
        return final_best_route, overall_best_distance, progress