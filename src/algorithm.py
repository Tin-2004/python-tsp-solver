import sys
import os
import logging # 导入 logging 模块

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
    def __init__(self, cities_coords, population_size, elite_rate, mutation_rate, generations, crossover_rate=0.9, start_city_idx=0, end_city_idx=None, tournament_size=1): # Add tournament_size
        self.cities_coords = cities_coords
        self.num_cities = len(cities_coords)
        self.population_size = population_size
        self.elite_rate = elite_rate # 新增：精英比例
        # 根据精英比例计算精英数量，确保至少为0
        self.elite_size = max(0, int(self.population_size * self.elite_rate))
        # 如果设置了精英比例但计算出的精英数量为0（由于种群太小或比例太低），并且种群大于0，则至少保留一个精英
        if self.elite_rate > 0 and self.elite_size == 0 and self.population_size > 0:
            self.elite_size = 1
            logging.info(f"提示: 由于种群大小 ({self.population_size}) 和精英比例 ({self.elite_rate}) 导致计算精英数量为0, 已自动设置为保留1个精英。")


        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.generations = generations
        self.start_city_idx = start_city_idx
        # 如果未指定 end_city_idx，则默认为最后一个城市
        self.end_city_idx = end_city_idx if end_city_idx is not None else self.num_cities - 1
        self.tournament_size = tournament_size # Store tournament size

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

    def _selection_for_breeding(self, ranked_population_with_indices, current_population, num_to_select):
        """
        选择操作（锦标赛选择），用于选择父代进行繁殖。
        ranked_population_with_indices: list of (original_index, fitness)
        current_population: the actual list of individuals
        num_to_select: how many parent individuals to select
        Returns a list of selected parent individuals (not indices).
        """
        selected_parents = []
        population_size = len(current_population)

        if population_size == 0:
            return [] # Cannot select from an empty population

        # Determine the actual tournament size, ensuring it's valid
        actual_tournament_size = min(self.tournament_size, population_size)
        if actual_tournament_size <= 0: # If self.tournament_size was 0 or negative, or pop_size is 0
            if population_size > 0:
                actual_tournament_size = 1 # Fallback to selecting the best of 1 (random pick effectively)
            else:
                return [] # Still cannot select if population is truly empty

        if not ranked_population_with_indices: # Should not happen if current_population is not empty
            # Fallback: if ranked_population_with_indices is empty for some reason, resort to random choice
            for _ in range(num_to_select):
                if current_population:
                    selected_parents.append(list(random.choice(current_population)))
            return selected_parents

        for _ in range(num_to_select):
            # Randomly select 'actual_tournament_size' contenders from the ranked population list
            # Each contender is a tuple (original_index, fitness_value)
            try:
                # Ensure we can sample 'actual_tournament_size' items
                # If ranked_population_with_indices has fewer items than actual_tournament_size (e.g. pop_size < tournament_size)
                # then sample all available items.
                num_candidates_to_sample = min(actual_tournament_size, len(ranked_population_with_indices))
                if num_candidates_to_sample == 0 and len(ranked_population_with_indices) > 0: # e.g. actual_tournament_size was forced to 0 but list not empty
                    num_candidates_to_sample = 1 # sample at least one if possible
                elif num_candidates_to_sample == 0: # no candidates to sample
                    if current_population: # fallback to random choice from current_pop
                         selected_parents.append(list(random.choice(current_population)))
                    continue


                tournament_contenders = random.sample(ranked_population_with_indices, num_candidates_to_sample)
            except ValueError: 
                # This can happen if ranked_population_with_indices is empty or smaller than num_candidates_to_sample
                # (though num_candidates_to_sample logic should prevent sampling more than available)
                # Fallback to picking a random individual from the current_population
                if current_population:
                    selected_parents.append(list(random.choice(current_population)))
                continue # Move to select the next parent

            winner_info = None # Stores (original_index, fitness) of the winner
            best_fitness_in_tournament = -float('inf')

            for original_idx, fitness_val in tournament_contenders:
                if fitness_val > best_fitness_in_tournament:
                    best_fitness_in_tournament = fitness_val
                    winner_info = (original_idx, fitness_val)
            
            if winner_info:
                # winner_info[0] is the original index in current_population
                selected_parents.append(list(current_population[winner_info[0]]))
            else:
                # Fallback if no winner was determined (e.g., all fitnesses were -inf, or empty contenders list)
                if current_population: # Ensure current_population is not empty
                    selected_parents.append(list(random.choice(current_population)))
        
        return selected_parents

    def _mating_pool(self, population, selection_results):
        """创建交配池"""
        matingpool = []
        for i in range(len(selection_results)):
            index = selection_results[i]
            matingpool.append(population[index])
        return matingpool

    def _crossover(self, parent1, parent2):
        """交叉操作（顺序交叉 OX1）"""
        # Ensure parents are not empty and are lists
        p1 = list(parent1) if parent1 is not None else []
        p2 = list(parent2) if parent2 is not None else []

        if not p1 or not p2 or len(p1) != len(p2) or not p1: 
            # If parents are unsuitable for crossover (e.g. different lengths, or empty after permute),
            # return one of them randomly, or an empty list if both are bad.
            if p1: return list(p1)
            if p2: return list(p2)
            return []

        # gene_a and gene_b are indices for the segment to be copied from parent1
        # Ensure len(p1)-1 is not negative if len(p1) is 0. Handled by initial check.
        gene_a = random.randint(0, len(p1) - 1)
        gene_b = random.randint(0, len(p1) - 1)
        
        start_gene = min(gene_a, gene_b)
        end_gene = max(gene_a, gene_b)

        child = [None] * len(p1)
        child_p1_segment_list = [] # Store the segment from p1

        # Copy the segment from parent1 to child and to child_p1_segment_list
        for i in range(start_gene, end_gene + 1): 
            child[i] = p1[i]
            child_p1_segment_list.append(p1[i])        
        
        # Convert the segment from p1 to a set for efficient lookup
        child_p1_segment_set = set(child_p1_segment_list)
            
        # Elements from p2 that are not in the copied segment from p1 (using the set for efficiency)
        p2_remaining_elements = [gene for gene in p2 if gene not in child_p1_segment_set]
        
        # Fill in the Nones in child with elements from p2_remaining_elements
        fill_idx = 0
        for i in range(len(child)):
            if child[i] is None:
                if fill_idx < len(p2_remaining_elements):
                    child[i] = p2_remaining_elements[fill_idx]
                    fill_idx += 1
                else:
                    # This should not happen if p1 and p2 are valid permutations of the same set of cities
                    # and p2_remaining_elements was correctly populated.
                    # If it does, child might contain Nones, which needs to be handled.
                    break # Break if no more elements to fill from p2_remaining
        
        # Final check for Nones, can happen if p2_remaining_elements was not enough
        # or if num_cities_to_permute is very small and crossover points are at ends.
        if None in child:
            current_child_genes = [g for g in child if g is not None]
            all_possible_genes = list(dict.fromkeys(p1 + p2)) # Unique genes from parents
            missing_genes = [g for g in all_possible_genes if g not in current_child_genes]
            random.shuffle(missing_genes) 
            
            fill_missing_idx = 0
            for i in range(len(child)):
                if child[i] is None:
                    if fill_missing_idx < len(missing_genes):
                        child[i] = missing_genes[fill_missing_idx]
                        fill_missing_idx += 1
                    else:
                        # Fallback to returning a copy of a parent if child is still malformed.
                        return list(p1) if random.random() < 0.5 else list(p2)
        return child


    def _mutate(self, individual_perm):
        """变异操作（反转变异）"""
        if not individual_perm or len(individual_perm) < 2: # Cannot mutate if less than 2 elements
            return list(individual_perm) # Return a copy
        
        mutated_individual = list(individual_perm) # Work on a copy
        if random.random() < self.mutation_rate:
            # Select two distinct random indices for the segment
            idx1, idx2 = sorted(random.sample(range(len(mutated_individual)), 2))
            
            # The segment to reverse is individual[idx1...idx2]
            segment_to_reverse = mutated_individual[idx1:idx2+1] # Slice includes idx2
            segment_to_reverse.reverse()
            mutated_individual[idx1:idx2+1] = segment_to_reverse
            
        return mutated_individual

    def run(self):
        population = self._create_initial_population()
        best_overall_route_perm = None
        best_overall_distance = float('inf')
        convergence_progress = []
        average_distances_progress = []

        if self.num_cities_to_permute == 0: # Only start and end city
            # No permutation possible, route is fixed
            fixed_route_perm = [] # Empty perm for intermediate cities
            best_overall_route_perm = fixed_route_perm
            best_full_route = self._get_full_route(fixed_route_perm)
            best_overall_distance = total_distance(best_full_route, self.cities_coords)
            convergence_progress = [best_overall_distance] * self.generations
            average_distances_progress = [best_overall_distance] * self.generations
            logging.info(f"只有起点和终点，路径固定，距离: {best_overall_distance:.2f}")
            return best_full_route, best_overall_distance, convergence_progress, average_distances_progress
        
        # Handle case where num_cities_to_permute is 1 (only one intermediate city)
        # Crossover and mutation might behave unusually.
        # GA might not be very effective here, but let it run.
        if self.num_cities_to_permute == 1:
            logging.warning("警告: 只有一个中间城市需要排列，遗传算法可能不是最优选择或行为可能受限。")


        logging.info(f"开始遗传算法，总共 {self.generations} 代，种群大小 {self.population_size}, 精英数量: {self.elite_size}")

        for generation in range(self.generations):
            current_generation_population = population 

            ranked_pop_with_indices = self._rank_routes(current_generation_population) 
            
            current_gen_best_fitness = ranked_pop_with_indices[0][1]
            current_gen_best_distance = 1 / current_gen_best_fitness
            
            current_gen_total_distance = 0
            if len(current_generation_population) > 0:
                for individual_perm in current_generation_population:
                    full_route = self._get_full_route(individual_perm)
                    current_gen_total_distance += total_distance(full_route, self.cities_coords)
                current_gen_avg_distance = current_gen_total_distance / len(current_generation_population)
            else:
                current_gen_avg_distance = float('inf') # Should not happen
            average_distances_progress.append(current_gen_avg_distance)

            if current_gen_best_distance < best_overall_distance:
                best_overall_distance = current_gen_best_distance
                best_overall_route_perm = list(current_generation_population[ranked_pop_with_indices[0][0]])
            
            convergence_progress.append(best_overall_distance)

            if generation % 10 == 0 or generation == self.generations - 1:
                logging.info(f"代 {generation+1}/{self.generations} - 当前代最优: {current_gen_best_distance:.2f}, 全局最优: {best_overall_distance:.2f}, 当前代平均: {current_gen_avg_distance:.2f}")

            next_generation_population = []

            # 1. 直接保留精英个体
            for i in range(self.elite_size):
                elite_individual_index = ranked_pop_with_indices[i][0]
                next_generation_population.append(list(current_generation_population[elite_individual_index]))

            # 2. 生成剩余的非精英子代
            num_offspring_to_generate = self.population_size - self.elite_size
            offspring_generated_count = 0
            
            # Create a pool of parents for breeding using selection
            # We need enough parents to generate num_offspring_to_generate.
            # If each crossover produces one child, we need num_offspring_to_generate * 2 selections if pairing.
            # Or select num_offspring_to_generate parents if each is used once with another.
            # Let's select a pool of potential parents.
            # The number of parents to select for the mating pool can be num_offspring_to_generate,
            # and then we pair them up.
            if num_offspring_to_generate > 0:
                # Select a pool of parents for breeding.
                # The size of this pool can be num_offspring_to_generate.
                # We will then pick pairs from this pool.
                # Ensure ranked_pop_with_indices is passed correctly.
                parent_pool = self._selection_for_breeding(ranked_pop_with_indices, current_generation_population, num_offspring_to_generate * 2) # Get enough for pairs

                if not parent_pool or len(parent_pool) < 2 and num_offspring_to_generate > 0 :
                    # Fallback: if parent pool is too small, fill with random individuals from current pop
                    # This might happen if selection_for_breeding has issues or pop is tiny.
                    # logging.info("警告: 父代选择池过小或为空，使用随机父代填充。")
                    temp_fill_needed = num_offspring_to_generate - offspring_generated_count
                    for _ in range(temp_fill_needed):
                        if current_generation_population: # Check if current_generation_population is not empty
                             random_parent_for_child = list(random.choice(current_generation_population))
                             mutated_random_child = self._mutate(random_parent_for_child) # Mutate it
                             next_generation_population.append(mutated_random_child)
                             offspring_generated_count +=1
                        else: # Cannot proceed if current_generation_population is empty
                            break 
                    # Ensure population size is maintained if we had to use this fallback for all offspring
                    while len(next_generation_population) < self.population_size and current_generation_population:
                        next_generation_population.append(list(random.choice(current_generation_population)))


                idx = 0
                while offspring_generated_count < num_offspring_to_generate and idx + 1 < len(parent_pool):
                    parent1 = parent_pool[idx]
                    parent2 = parent_pool[idx+1]
                    
                    child = []
                    if random.random() < self.crossover_rate and self.num_cities_to_permute >= 2 : # Crossover only if enough cities to permute
                        child = self._crossover(parent1, parent2)
                        if not child or len(child) != self.num_cities_to_permute: # Crossover failed or produced wrong size
                            child = list(parent1) if random.random() < 0.5 else list(parent2) # Fallback
                    else:
                        child = list(parent1) if random.random() < 0.5 else list(parent2) # Copy one parent

                    # Ensure child is a valid permutation before mutation
                    if not child or len(child) != self.num_cities_to_permute:
                        # If child is still invalid, use a fresh random individual for this slot
                        # This is a fallback for problematic crossovers/copies with very small num_cities_to_permute
                        child = self._create_individual()


                    mutated_child = self._mutate(child)
                    next_generation_population.append(mutated_child)
                    offspring_generated_count += 1
                    idx += 2 # Move to next pair of parents

                # If not enough offspring were generated (e.g. parent_pool was odd or too small)
                # Fill remaining spots by mutating copies of existing elites or selected parents
                # This is to ensure the population size is maintained.
                while offspring_generated_count < num_offspring_to_generate:
                    fallback_parent = []
                    if next_generation_population : # Prefer to use already selected elites if available
                        fallback_parent = list(random.choice(next_generation_population[:self.elite_size])) \
                                           if self.elite_size > 0 and next_generation_population \
                                           else (list(random.choice(parent_pool)) if parent_pool else self._create_individual())
                    elif parent_pool:
                        fallback_parent = list(random.choice(parent_pool))
                    else: # Absolute fallback
                        fallback_parent = self._create_individual()

                    mutated_fallback_child = self._mutate(fallback_parent)
                    next_generation_population.append(mutated_fallback_child)
                    offspring_generated_count += 1
            
            population = next_generation_population

        best_full_route = self._get_full_route(best_overall_route_perm) # Removed problematic conditional here
                                                                    # best_overall_route_perm should be valid if GA ran.
        
        # Final checks and error handling for result consistency
        if best_overall_route_perm is None:
            if self.num_cities_to_permute > 0 : # If there were cities to permute but no best_overall_route_perm
                logging.warning("警告: GA未能确定最优个体排列，将尝试使用最后一代的最优个体。")
                if population: # Check if population is not empty
                    last_gen_ranked = self._rank_routes(population)
                    if last_gen_ranked:
                        best_overall_route_perm = list(population[last_gen_ranked[0][0]])
                        best_full_route = self._get_full_route(best_overall_route_perm)
                        best_overall_distance = 1 / last_gen_ranked[0][1] # Update distance
                    else: # Population was empty or ranking failed
                        best_overall_route_perm = self._create_individual() # Create a random one
                        best_full_route = self._get_full_route(best_overall_route_perm)
                        best_overall_distance = 1 / self._calculate_fitness(best_overall_route_perm)
                else: # Population is empty, create a random one
                    best_overall_route_perm = self._create_individual()
                    best_full_route = self._get_full_route(best_overall_route_perm)
                    best_overall_distance = 1 / self._calculate_fitness(best_overall_route_perm)

            elif self.num_cities_to_permute == 0: # Only start and end, should have been handled
                best_full_route = [self.start_city_idx, self.end_city_idx]
                best_overall_distance = total_distance(best_full_route, self.cities_coords)
        
        if not best_full_route or len(best_full_route) != self.num_cities:
             # Fallback if best_full_route is somehow malformed
            if self.num_cities_to_permute == 0:
                best_full_route = [self.start_city_idx, self.end_city_idx]
            else: # Try to reconstruct from best_overall_route_perm or make a new one
                if best_overall_route_perm and len(best_overall_route_perm) == self.num_cities_to_permute:
                    best_full_route = self._get_full_route(best_overall_route_perm)
                else: # Last resort
                    best_overall_route_perm = self._create_individual()
                    best_full_route = self._get_full_route(best_overall_route_perm)
            # Recalculate distance if route was re-formed
            best_overall_distance = total_distance(best_full_route, self.cities_coords)


        if self.num_cities == 0: # Should be caught by __init__ but as a safeguard
            return [], 0, [], []

        return best_full_route, best_overall_distance, convergence_progress, average_distances_progress