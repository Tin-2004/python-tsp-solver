import numpy as np
import random

class AntColonyOptimizationTSP:
    def __init__(self, cities_coords, n_ants, n_iterations, alpha, beta, evaporation_rate, q=1.0, pheromone_init=0.1, start_city_idx=0, end_city_idx=None):
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

    def _update_pheromones(self, all_paths, best_path_distance_current_iter):
        # Evaporation
        self.pheromone *= (1 - self.evaporation_rate)

        # Deposition
        for path, distance in all_paths:
            pheromone_deposit = self.q / distance 
            for i in range(len(path) - 1):
                self.pheromone[path[i], path[i+1]] += pheromone_deposit
                self.pheromone[path[i+1], path[i]] += pheromone_deposit # Symmetric TSP

    def run(self):
        best_overall_path = None
        best_overall_distance = float('inf')
        convergence_progress = []

        print(f"初始信息素水平: {self.pheromone_init}")
        print(f"启发式信息 (部分样本 from city 0): {self.heuristic[0, :5]}")


        for iteration in range(self.n_iterations):
            all_paths_this_iteration = []
            current_iter_best_distance = float('inf')
            current_iter_best_path = None

            for ant in range(self.n_ants):
                path = self._construct_solution(ant)
                distance = self._calculate_path_distance(path)
                
                all_paths_this_iteration.append((path, distance))

                if distance < current_iter_best_distance:
                    current_iter_best_distance = distance
                    # current_iter_best_path = path # Not strictly needed here

                if distance < best_overall_distance:
                    best_overall_distance = distance
                    best_overall_path = path
            
            self._update_pheromones(all_paths_this_iteration, current_iter_best_distance)
            
            convergence_progress.append(best_overall_distance)
            if (iteration + 1) % 10 == 0 or iteration == 0:
                print(f"第 {iteration + 1} 代: 当前代最佳距离 = {current_iter_best_distance:.2f}, 全局最佳距离 = {best_overall_distance:.2f}")

        if best_overall_path is None and self.n_cities > 0 : # Handle case where no path was found (e.g. single city)
            if self.n_cities == 1:
                best_overall_path = [self.start_city_idx]
                best_overall_distance = 0.0
            else: # Should not happen in a connected graph with n_cities > 1
                 raise Exception("No path found, check ACO logic or graph connectivity.")
        elif self.n_cities == 0:
            return [], 0, []


        return best_overall_path, best_overall_distance, convergence_progress

