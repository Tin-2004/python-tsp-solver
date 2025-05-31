import sys
import os
import logging # 导入 logging 模块

# 将项目根目录 (python-tsp-solver) 添加到 sys.path
# 这样可以确保 'from src.module' 这样的导入能够正确工作。
# os.path.dirname(__file__) 是 'src' 目录。
# os.path.join(os.path.dirname(__file__), '..') 是 'python-tsp-solver' 目录。
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import numpy as np
import random
from src.utils import total_distance

class GeneticAlgorithmTSP:
    def __init__(self, cities_coords, population_size, elite_rate, mutation_rate, generations, crossover_rate=0.9, start_city_idx=0, end_city_idx=None, tournament_size=1):
        self.cities_coords = cities_coords # 城市坐标
        self.num_cities = len(cities_coords) # 城市数量
        self.population_size = population_size # 种群大小
        self.elite_rate = elite_rate # 精英比例
        # 根据精英比例计算精英数量，确保至少为0
        self.elite_size = max(0, int(self.population_size * self.elite_rate))
        # 如果设置了精英比例但计算出的精英数量为0（由于种群太小或比例太低），并且种群大于0，则至少保留一个精英
        if self.elite_rate > 0 and self.elite_size == 0 and self.population_size > 0:
            self.elite_size = 1
            logging.info(f"提示: 由于种群大小 ({self.population_size}) 和精英比例 ({self.elite_rate}) 导致计算精英数量为0, 已自动设置为保留1个精英。")


        self.mutation_rate = mutation_rate # 变异率
        self.crossover_rate = crossover_rate # 交叉率
        self.generations = generations # 迭代代数
        self.start_city_idx = start_city_idx # 起点城市索引
        # 如果未指定 end_city_idx，则默认为最后一个城市
        self.end_city_idx = end_city_idx if end_city_idx is not None else self.num_cities - 1 # 终点城市索引
        self.tournament_size = tournament_size # 锦标赛选择的锦标赛大小

        if self.num_cities <= 2:
            raise ValueError("城市数量必须大于2才能进行路径规划。")

        # 需要排序的城市索引 (不包括起点和终点)
        self.cities_to_permute_indices = [i for i in range(self.num_cities) if i != self.start_city_idx and i != self.end_city_idx]
        self.num_cities_to_permute = len(self.cities_to_permute_indices) # 需要排列的城市数量


    def _create_individual(self):
        """创建一个代表中间城市访问顺序的个体（染色体）"""
        if self.num_cities_to_permute == 0: # 如果只有起点和终点，没有中间城市
            return []
        # 从需要排列的城市中随机抽样，形成一个排列
        return random.sample(self.cities_to_permute_indices, self.num_cities_to_permute)

    def _create_initial_population(self):
        """创建初始种群"""
        population = []
        for _ in range(self.population_size):
            population.append(self._create_individual())
        return population

    def _get_full_route(self, individual_perm):
        """根据个体的中间城市排列，构建从起点到终点的完整路径"""
        if not individual_perm: # 如果中间城市排列为空 (例如只有起点和终点)
             return [self.start_city_idx, self.end_city_idx]
        return [self.start_city_idx] + individual_perm + [self.end_city_idx]

    def _calculate_fitness(self, individual_perm):
        """计算个体的适应度（总距离的倒数，距离越短适应度越高）"""
        route = self._get_full_route(individual_perm)
        return 1 / total_distance(route, self.cities_coords)

    def _rank_routes(self, population):
        """根据适应度对种群中的个体进行降序排序"""
        fitness_results = {} # 存储每个个体的索引及其适应度
        for i in range(len(population)):
            fitness_results[i] = self._calculate_fitness(population[i])
        # 按适应度值降序排序
        return sorted(fitness_results.items(), key=lambda x: x[1], reverse=True)

    def _selection(self, ranked_population):
        """
        选择操作（精英选择 + 轮盘赌选择）。
        注意：此方法在当前版本的 `run` 方法中未被直接用于生成下一代，
        `_selection_for_breeding` (锦标赛选择) 被用于选择父代。
        此方法可能用于其他选择策略或早期的版本。
        """
        selection_results = []
        # 1. 精英选择：直接选择适应度最高的几个个体
        for i in range(self.elite_size):
            selection_results.append(ranked_population[i][0]) # 添加精英个体的原始索引

        # 2. 轮盘赌选择剩余个体
        # 提取所有个体的适应度值
        df = np.array([ranked_population[i][1] for i in range(len(ranked_population))])
        df_sum = np.sum(df) # 计算总适应度
        if df_sum == 0: # 避免除以零错误，如果所有适应度都是0
            # 如果所有适应度为0，则进行随机选择
            for _ in range(len(ranked_population) - self.elite_size):
                selection_results.append(random.choice(range(len(ranked_population))))
        else:
            # 计算每个个体的选择概率和累积概率
            cumulative_sum = np.cumsum(df / df_sum)
            for _ in range(len(ranked_population) - self.elite_size): # 选择剩余数量的个体
                pick = random.random() # 生成一个0到1之间的随机数
                for i in range(len(ranked_population)):
                    if pick <= cumulative_sum[i]: # 当随机数小于等于累积概率时，选择该个体
                        selection_results.append(ranked_population[i][0])
                        break
        return selection_results # 返回被选中个体的索引列表

    def _selection_for_breeding(self, ranked_population_with_indices, current_population, num_to_select):
        """
        选择操作（锦标赛选择），用于选择父代进行繁殖。
        ranked_population_with_indices: 包含个体原始索引和适应度的列表 [(original_index, fitness), ...]
        current_population: 当前种群的个体列表
        num_to_select: 需要选择的父代个体数量
        返回: 被选中的父代个体列表 (不是索引)。
        """
        selected_parents = []
        population_size = len(current_population)

        if population_size == 0:
            return [] # 无法从空种群中选择

        # 确定实际的锦标赛大小，确保其有效
        actual_tournament_size = min(self.tournament_size, population_size)
        if actual_tournament_size <= 0: # 如果 self.tournament_size 为0或负数，或种群大小为0
            if population_size > 0:
                actual_tournament_size = 1 # 后备：选择1个中最好的（实际上是随机选择）
            else:
                return [] # 如果种群确实为空，仍然无法选择

        if not ranked_population_with_indices: # 如果 current_population 非空，则不应发生此情况
            # 后备：如果 ranked_population_with_indices 由于某种原因为空，则采取随机选择
            for _ in range(num_to_select):
                if current_population:
                    selected_parents.append(list(random.choice(current_population)))
            return selected_parents

        for _ in range(num_to_select): # 迭代选择指定数量的父代
            # 从排序后的种群列表中随机选择 'actual_tournament_size' 个竞争者
            # 每个竞争者是一个元组 (original_index, fitness_value)
            try:
                # 确保可以采样 'actual_tournament_size' 个项目
                # 如果 ranked_population_with_indices 的项目数少于 actual_tournament_size (例如 pop_size < tournament_size)
                # 则采样所有可用的项目。
                num_candidates_to_sample = min(actual_tournament_size, len(ranked_population_with_indices))
                if num_candidates_to_sample == 0 and len(ranked_population_with_indices) > 0: # 例如 actual_tournament_size 被强制为0但列表不为空
                    num_candidates_to_sample = 1 # 如果可能，至少采样一个
                elif num_candidates_to_sample == 0: # 没有候选者可供采样
                    if current_population: # 后备：从当前种群中随机选择
                         selected_parents.append(list(random.choice(current_population)))
                    continue # 继续选择下一个父代


                tournament_contenders = random.sample(ranked_population_with_indices, num_candidates_to_sample)
            except ValueError: 
                # 如果 ranked_population_with_indices 为空或小于 num_candidates_to_sample，则可能发生此情况
                # (尽管 num_candidates_to_sample 的逻辑应防止采样超过可用数量)
                # 后备：从 current_population 中随机选择一个个体
                if current_population:
                    selected_parents.append(list(random.choice(current_population)))
                continue # 继续选择下一个父代

            winner_info = None # 存储胜者的 (original_index, fitness)
            best_fitness_in_tournament = -float('inf') # 初始化锦标赛中的最佳适应度

            # 从竞争者中选出适应度最高的作为胜者
            for original_idx, fitness_val in tournament_contenders:
                if fitness_val > best_fitness_in_tournament:
                    best_fitness_in_tournament = fitness_val
                    winner_info = (original_idx, fitness_val)
            
            if winner_info:
                # winner_info[0] 是胜者在 current_population 中的原始索引
                selected_parents.append(list(current_population[winner_info[0]]))
            else:
                # 后备：如果没有确定胜者 (例如，所有适应度都是-inf，或竞争者列表为空)
                if current_population: # 确保 current_population 不为空
                    selected_parents.append(list(random.choice(current_population)))
        
        return selected_parents

    def _mating_pool(self, population, selection_results):
        """根据选择结果创建交配池"""
        matingpool = []
        for i in range(len(selection_results)):
            index = selection_results[i] # 获取被选中个体的索引
            matingpool.append(population[index]) # 将选中的个体添加到交配池
        return matingpool

    def _crossover(self, parent1, parent2):
        """交叉操作（顺序交叉 OX1）"""
        # 确保父代非空且为列表类型
        p1 = list(parent1) if parent1 is not None else []
        p2 = list(parent2) if parent2 is not None else []

        if not p1 or not p2 or len(p1) != len(p2) or not p1: 
            # 如果父代不适合交叉 (例如长度不同，或排列后为空)
            # 则随机返回其中一个父代，或者如果两者都无效则返回空列表。
            if p1: return list(p1)
            if p2: return list(p2)
            return []

        # gene_a 和 gene_b 是用于从 parent1 复制片段的索引
        # 初始检查已处理 len(p1) 为 0 的情况，确保 len(p1)-1 不为负。
        gene_a = random.randint(0, len(p1) - 1)
        gene_b = random.randint(0, len(p1) - 1)
        
        start_gene = min(gene_a, gene_b) # 交叉片段的起始点
        end_gene = max(gene_a, gene_b)   # 交叉片段的结束点

        child = [None] * len(p1) # 初始化子代
        child_p1_segment_list = [] # 存储从 parent1 复制的片段

        # 将 parent1 的片段复制到子代和 child_p1_segment_list
        for i in range(start_gene, end_gene + 1): 
            child[i] = p1[i]
            child_p1_segment_list.append(p1[i])        
        
        # 将从 p1 复制的片段转换为集合以便高效查找
        child_p1_segment_set = set(child_p1_segment_list)
            
        # 从 p2 中提取不在 p1 片段中的元素 (使用集合进行高效判断)
        p2_remaining_elements = [gene for gene in p2 if gene not in child_p1_segment_set]
        
        # 用 p2_remaining_elements 中的元素填充子代中的 None 值
        fill_idx = 0
        for i in range(len(child)):
            if child[i] is None: # 如果当前位置为空
                if fill_idx < len(p2_remaining_elements):
                    child[i] = p2_remaining_elements[fill_idx]
                    fill_idx += 1
                else:
                    # 如果 p1 和 p2 是同一城市集合的有效排列，
                    # 并且 p2_remaining_elements 被正确填充，则不应发生此情况。
                    # 如果发生，子代可能包含 None，需要处理。
                    break # 如果没有更多来自 p2_remaining 的元素可填充，则中断
        
        # 最后检查 None 值，可能发生在 p2_remaining_elements 不足
        # 或者 num_cities_to_permute 非常小且交叉点在两端时。
        if None in child:
            current_child_genes = [g for g in child if g is not None] # 子代中已有的基因
            all_possible_genes = list(dict.fromkeys(p1 + p2)) # 来自父代的唯一基因集合
            missing_genes = [g for g in all_possible_genes if g not in current_child_genes] # 缺失的基因
            random.shuffle(missing_genes) # 打乱缺失基因的顺序
            
            fill_missing_idx = 0
            for i in range(len(child)):
                if child[i] is None:
                    if fill_missing_idx < len(missing_genes):
                        child[i] = missing_genes[fill_missing_idx]
                        fill_missing_idx += 1
                    else:
                        # 如果子代仍然格式错误，则后备：返回一个父代的副本。
                        return list(p1) if random.random() < 0.5 else list(p2)
        return child


    def _mutate(self, individual_perm):
        """变异操作（反转变异/倒置变异）"""
        if not individual_perm or len(individual_perm) < 2: # 如果个体为空或长度小于2，则无法变异
            return list(individual_perm) # 返回副本
        
        mutated_individual = list(individual_perm) # 在副本上操作
        if random.random() < self.mutation_rate: # 以一定的变异概率执行变异
            # 选择两个不同的随机索引用于反转片段
            idx1, idx2 = sorted(random.sample(range(len(mutated_individual)), 2))
            
            # 需要反转的片段是 individual[idx1...idx2]
            segment_to_reverse = mutated_individual[idx1:idx2+1] # 切片包含 idx2
            segment_to_reverse.reverse() # 反转片段
            mutated_individual[idx1:idx2+1] = segment_to_reverse # 将反转后的片段放回个体
            
        return mutated_individual

    def run(self):
        """执行遗传算法"""
        population = self._create_initial_population() # 初始化种群
        best_overall_route_perm = None # 全局最优路径的中间城市排列
        best_overall_distance = float('inf') # 全局最优距离
        convergence_progress = [] # 记录每代最优距离，用于观察收敛过程
        average_distances_progress = [] # 记录每代平均距离

        # 特殊情况：如果只有起点和终点 (没有中间城市需要排列)
        if self.num_cities_to_permute == 0: 
            # 无需排列，路径固定
            fixed_route_perm = [] # 中间城市排列为空
            best_overall_route_perm = fixed_route_perm
            best_full_route = self._get_full_route(fixed_route_perm)
            best_overall_distance = total_distance(best_full_route, self.cities_coords)
            convergence_progress = [best_overall_distance] * self.generations # 每代最优都是这个固定值
            average_distances_progress = [best_overall_distance] * self.generations # 每代平均也是这个固定值
            logging.info(f"只有起点和终点，路径固定，距离: {best_overall_distance:.2f}")
            return best_full_route, best_overall_distance, convergence_progress, average_distances_progress
        
        # 处理只有一个中间城市的情况
        # 交叉和变异操作可能行为异常。
        # 遗传算法在这里可能效果不佳，但仍允许其运行。
        if self.num_cities_to_permute == 1:
            logging.warning("警告: 只有一个中间城市需要排列，遗传算法可能不是最优选择或行为可能受限。")


        logging.info(f"开始遗传算法，总共 {self.generations} 代，种群大小 {self.population_size}, 精英数量: {self.elite_size}")

        for generation in range(self.generations): # 迭代指定的代数
            current_generation_population = population # 当前代种群

            # 对当前代种群按适应度排序，结果包含个体索引和适应度
            ranked_pop_with_indices = self._rank_routes(current_generation_population) 
            
            current_gen_best_fitness = ranked_pop_with_indices[0][1] # 当前代最优适应度
            current_gen_best_distance = 1 / current_gen_best_fitness # 当前代最优距离
            
            current_gen_total_distance = 0 # 当前代总距离
            if len(current_generation_population) > 0:
                for individual_perm in current_generation_population:
                    full_route = self._get_full_route(individual_perm)
                    current_gen_total_distance += total_distance(full_route, self.cities_coords)
                current_gen_avg_distance = current_gen_total_distance / len(current_generation_population) # 当前代平均距离
            else:
                current_gen_avg_distance = float('inf') # 正常情况下不应发生
            average_distances_progress.append(current_gen_avg_distance)

            # 更新全局最优解
            if current_gen_best_distance < best_overall_distance:
                best_overall_distance = current_gen_best_distance
                # 存储最优个体的中间城市排列 (深拷贝)
                best_overall_route_perm = list(current_generation_population[ranked_pop_with_indices[0][0]]) 
            
            convergence_progress.append(best_overall_distance) # 记录当前全局最优距离

            # 定期打印日志
            if generation % 10 == 0 or generation == self.generations - 1:
                logging.info(f"代 {generation+1}/{self.generations} - 当前代最优: {current_gen_best_distance:.2f}, 全局最优: {best_overall_distance:.2f}, 当前代平均: {current_gen_avg_distance:.2f}")

            next_generation_population = [] # 用于存放下一代种群

            # 1. 精英保留：直接将当前代最优的几个个体复制到下一代
            for i in range(self.elite_size):
                elite_individual_index = ranked_pop_with_indices[i][0]
                next_generation_population.append(list(current_generation_population[elite_individual_index]))

            # 2. 生成剩余的非精英子代，直到达到种群大小
            num_offspring_to_generate = self.population_size - self.elite_size # 需要生成的子代数量
            offspring_generated_count = 0 # 已生成的子代计数
            
            # 为繁殖创建父代池
            # 需要足够的父代来生成 num_offspring_to_generate 个子代。
            # 如果每次交叉产生一个子代，则如果配对，需要 num_offspring_to_generate * 2 次选择。
            # 或者，如果每个父代与另一个父代使用一次，则选择 num_offspring_to_generate 个父代。
            # 这里选择一个潜在的父代池。
            # 交配池中父代的数量可以是 num_offspring_to_generate，然后进行配对。
            if num_offspring_to_generate > 0:
                # 选择一个父代池用于繁殖。
                # 该池的大小可以是 num_offspring_to_generate。
                # 然后从这个池中选择配对。
                # 确保正确传递 ranked_pop_with_indices。
                # 获取足够的父代用于配对 (num_offspring_to_generate * 2)
                parent_pool = self._selection_for_breeding(ranked_pop_with_indices, current_generation_population, num_offspring_to_generate * 2) 

                if not parent_pool or len(parent_pool) < 2 and num_offspring_to_generate > 0 :
                    # 后备：如果父代池太小，则用当前种群中的随机个体填充。
                    # 如果 _selection_for_breeding 有问题或种群非常小，则可能发生此情况。
                    # logging.info("警告: 父代选择池过小或为空，使用随机父代填充。") # 此日志信息可根据需要启用
                    temp_fill_needed = num_offspring_to_generate - offspring_generated_count
                    for _ in range(temp_fill_needed):
                        if current_generation_population: # 检查 current_generation_population 是否非空
                             random_parent_for_child = list(random.choice(current_generation_population))
                             mutated_random_child = self._mutate(random_parent_for_child) # 对其进行变异
                             next_generation_population.append(mutated_random_child)
                             offspring_generated_count +=1
                        else: # 如果 current_generation_population 为空，则无法继续
                            break 
                    # 如果所有子代都必须使用此后备方案，则确保维持种群大小
                    while len(next_generation_population) < self.population_size and current_generation_population:
                        next_generation_population.append(list(random.choice(current_generation_population)))


                idx = 0 # 父代池索引
                while offspring_generated_count < num_offspring_to_generate and idx + 1 < len(parent_pool):
                    parent1 = parent_pool[idx]
                    parent2 = parent_pool[idx+1]
                    
                    child = []
                    # 以一定的交叉率进行交叉，且需要排列的城市数大于等于2
                    if random.random() < self.crossover_rate and self.num_cities_to_permute >= 2 : 
                        child = self._crossover(parent1, parent2)
                        if not child or len(child) != self.num_cities_to_permute: # 交叉失败或产生的子代大小不正确
                            child = list(parent1) if random.random() < 0.5 else list(parent2) # 后备：随机选择一个父代
                    else: # 不进行交叉，随机复制一个父代
                        child = list(parent1) if random.random() < 0.5 else list(parent2) 

                    # 在变异前确保子代是有效的排列
                    if not child or len(child) != self.num_cities_to_permute:
                        # 如果子代仍然无效，则为此位置使用一个全新的随机个体
                        # 这是针对 num_cities_to_permute 非常小时有问题的交叉/复制的后备方案
                        child = self._create_individual()


                    mutated_child = self._mutate(child) # 对子代进行变异
                    next_generation_population.append(mutated_child)
                    offspring_generated_count += 1
                    idx += 2 # 移动到下一对父代

                # 如果生成的子代数量不足 (例如 parent_pool 数量为奇数或太小)
                # 通过变异现有精英或已选父代的副本来填充剩余位置
                # 这是为了确保维持种群大小。
                while offspring_generated_count < num_offspring_to_generate:
                    fallback_parent = []
                    if next_generation_population : # 如果下一代种群中已有精英，优先使用
                        fallback_parent = list(random.choice(next_generation_population[:self.elite_size])) \
                                           if self.elite_size > 0 and next_generation_population \
                                           else (list(random.choice(parent_pool)) if parent_pool else self._create_individual())
                    elif parent_pool: # 否则从父代池中选择
                        fallback_parent = list(random.choice(parent_pool))
                    else: # 绝对后备：创建一个新的随机个体
                        fallback_parent = self._create_individual()

                    mutated_fallback_child = self._mutate(fallback_parent)
                    next_generation_population.append(mutated_fallback_child)
                    offspring_generated_count += 1
            
            population = next_generation_population # 更新种群为新生成的下一代

        # 获取最终的最优完整路径
        # best_overall_route_perm 在遗传算法运行后应该是有效的。
        best_full_route = self._get_full_route(best_overall_route_perm) 
        
        # 对结果一致性进行最终检查和错误处理
        if best_overall_route_perm is None:
            if self.num_cities_to_permute > 0 : # 如果有需要排列的城市但没有找到 best_overall_route_perm
                logging.warning("警告: GA未能确定最优个体排列，将尝试使用最后一代的最优个体。")
                if population: # 检查种群是否非空
                    last_gen_ranked = self._rank_routes(population) # 对最后一代种群排序
                    if last_gen_ranked:
                        best_overall_route_perm = list(population[last_gen_ranked[0][0]])
                        best_full_route = self._get_full_route(best_overall_route_perm)
                        best_overall_distance = 1 / last_gen_ranked[0][1] # 更新距离
                    else: # 种群为空或排序失败
                        best_overall_route_perm = self._create_individual() # 创建一个随机个体
                        best_full_route = self._get_full_route(best_overall_route_perm)
                        best_overall_distance = 1 / self._calculate_fitness(best_overall_route_perm)
                else: # 种群为空，创建一个随机个体
                    best_overall_route_perm = self._create_individual()
                    best_full_route = self._get_full_route(best_overall_route_perm)
                    best_overall_distance = 1 / self._calculate_fitness(best_overall_route_perm)

            elif self.num_cities_to_permute == 0: # 只有起点和终点，此情况应已在开始时处理
                best_full_route = [self.start_city_idx, self.end_city_idx]
                best_overall_distance = total_distance(best_full_route, self.cities_coords)
        
        if not best_full_route or len(best_full_route) != self.num_cities:
             # 后备：如果 best_full_route 由于某种原因格式错误
            if self.num_cities_to_permute == 0: # 只有起点和终点
                best_full_route = [self.start_city_idx, self.end_city_idx]
            else: # 尝试从 best_overall_route_perm 重建或创建一个新的
                if best_overall_route_perm and len(best_overall_route_perm) == self.num_cities_to_permute:
                    best_full_route = self._get_full_route(best_overall_route_perm)
                else: # 最后手段
                    best_overall_route_perm = self._create_individual()
                    best_full_route = self._get_full_route(best_overall_route_perm)
            # 如果路径被重新构建，则重新计算距离
            best_overall_distance = total_distance(best_full_route, self.cities_coords)


        if self.num_cities == 0: # 此情况应由 __init__ 捕获，但作为安全措施
            return [], 0, [], []

        return best_full_route, best_overall_distance, convergence_progress, average_distances_progress