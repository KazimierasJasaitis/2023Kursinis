import random
import numpy as np
import matplotlib.pyplot as plt

class AntColonyOptimizer:
    def __init__(self, num_ants, alpha, beta, rho, q, pheromone_callback=None):
        self.num_ants = num_ants
        self.alpha = alpha
        self.beta = beta
        self.rho = rho
        self.q = q
        self.pheromone_callback = pheromone_callback
        self.iteration = 0

    def _initialize_pheromones(self, cities_count):
        return np.ones((cities_count, cities_count))

    def _initialize_distances(self, cities):
        cities_count = len(cities)
        distances = np.zeros((cities_count, cities_count))
        for i in range(cities_count):
            for j in range(i + 1, cities_count):
                dist = np.sqrt((cities[i][0] - cities[j][0]) ** 2 + (cities[i][1] - cities[j][1]) ** 2)
                distances[i][j] = dist
                distances[j][i] = dist
        return distances

    def _calculate_probabilities(self, city, unvisited, pheromones, distances):
        pheromone_trails = np.array([pheromones[city][i] for i in unvisited])
        distances_to_unvisited = np.array([distances[city][i] for i in unvisited])
        attractiveness = 1 / distances_to_unvisited

        probabilities = (pheromone_trails ** self.alpha) * (attractiveness ** self.beta)
        return probabilities / probabilities.sum()

    def _construct_solutions(self, cities_count, pheromones, distances):
        solutions = []
        for _ in range(self.num_ants):
                #random first visited city
            visited = [random.randint(0, cities_count - 1)]
            unvisited = list(set(range(cities_count)) - set(visited))

            for _ in range(cities_count - 1):
                current_city = visited[-1]
                probabilities = self._calculate_probabilities(current_city, unvisited, pheromones, distances)
                    #when selecting a random index, you can provide a list of probabilities for each index (p=)
                next_city = unvisited[np.random.choice(range(len(unvisited)), p=probabilities)]
                visited.append(next_city)
                unvisited.remove(next_city)

            solutions.append((visited, self._calculate_tour_length(visited, distances)))
        return solutions

    def _calculate_tour_length(self, tour, distances):
        tour_length = sum([distances[tour[i]][tour[i + 1]] for i in range(len(tour) - 1)])
        tour_length += distances[tour[-1]][tour[0]]
        return tour_length

    def _update_pheromones(self, pheromones, solutions):
        best_solution = min(solutions, key=lambda x: x[1])
        for i in range(len(best_solution[0]) - 1):
            pheromones[best_solution[0][i]][best_solution[0][i + 1]] += self.q / best_solution[1]
            pheromones[best_solution[0][i + 1]][best_solution[0][i]] += self.q / best_solution[1]

    def optimize(self, cities):
        cities_count = len(cities)
        pheromones = self._initialize_pheromones(cities_count)
        distances = self._initialize_distances(cities)

        best_solution = None
        done = 0
        done_thr = 0
        while done != 1:
            solutions = self._construct_solutions(cities_count, pheromones, distances)
                #Takes the decond value in solutions, which is the tour length and finds the lowest value
            current_best_solution = min(solutions, key=lambda x: x[1])
            if best_solution is None or current_best_solution[1] < best_solution[1]:
                best_solution = current_best_solution
                done_thr = 0
            else:
                done_thr += 1
                if done_thr == 20:
                    done = 1
            self.iteration += 1
            pheromones = (1 - self.rho) * pheromones
            self._update_pheromones(pheromones, solutions)

            # Call the pheromone_callback function, if provided
            if self.pheromone_callback:
                self.pheromone_callback(pheromones, best_solution)

        return best_solution

def plot_solution(cities, best_tour, iteration):
    plt.clf()

    cities_x = [city[0] for city in cities]
    cities_y = [city[1] for city in cities]

    tour_x = [cities_x[best_tour[i]] for i in range(len(best_tour))]
    tour_y = [cities_y[best_tour[i]] for i in range(len(best_tour))]

    plt.scatter(cities_x, cities_y, color='blue')
    plt.plot(tour_x + [tour_x[0]], tour_y + [tour_y[0]], color='red', linestyle='-', linewidth=1, marker='o')

    for i, city in enumerate(cities):
        plt.text(city[0] + 0.1, city[1] + 0.1, f"{i}", fontsize=12, color='black')

    plt.title(f"TSP Solution using Ant Colony Optimization (Iteration {iteration})")
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")
    plt.grid()
    plt.draw()
    plt.pause(0.001)

def graph_callback_with_iteration(pheromones, best_solution):
    if not hasattr(graph_callback_with_iteration, "iteration"):
        graph_callback_with_iteration.iteration = 1
    n = 5
    if graph_callback_with_iteration.iteration % n == 0:
        plot_solution(cities, best_solution[0], graph_callback_with_iteration.iteration)

    graph_callback_with_iteration.iteration += 1

def solve_once():
    # Initialize the ACO and run the optimization
    aco = AntColonyOptimizer(
        num_ants, alpha, beta, rho, q, pheromone_callback=graph_callback_with_iteration
    )
    best_tour, best_tour_length = aco.optimize(cities)

    # Plot the solution
    plot_solution(cities, best_tour, aco.iteration)

    print("Best tour:", best_tour)
    print("Best tour length:", best_tour_length)
    print("Number of iterations:", aco.iteration)

    # Plot the solution
    plot_solution(cities, best_tour, aco.iteration)
    plt.show()

def solve_multiple(n):
    for i in range(n):
        aco = AntColonyOptimizer(
        num_ants, alpha, beta, rho, q)
        best_tour, best_tour_length = aco.optimize(cities)
        print("Best tour length:", best_tour_length)



if __name__ == "__main__":
    # Define the coordinates of the cities
    cities = [
    (0, 0),
    (1, 1),
    (3, 1),
    (4, 0),
    (3, -1),
    (1, -1),
    (13, 0),
    (1, 7),
    (3, 8),
    (4, 11),
    (3, -16),
    (1, -12),
    (11, 0),
    (12, 1),
    (3, 16),
    (-4, 0),
    (-3, -1),
    (-11, -1),
    (-13, 0),
    (1, -7),
    (3, 5),
    (4, -11),
    (3, -6),
    (1, -2),
    (7, 15),
    (15, 6),
    (-6, -12),
    (9, -7),
    (10, 8),
    (14, -3)
]


    # Define the ACO parameters
    num_ants = 100
    alpha = 0.5
    beta = 5
    rho = 0.1
    q = 100

    ntimes = 5
    solve_multiple(ntimes)

    

    