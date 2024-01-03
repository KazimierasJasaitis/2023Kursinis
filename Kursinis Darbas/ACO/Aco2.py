import random
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


class AntColonyOptimizer:
    def __init__(self, num_ants, num_iterations, alpha, beta, rho, q, pheromone_callback=None):
        self.num_ants = num_ants
        self.num_iterations = num_iterations
        self.alpha = alpha
        self.beta = beta
        self.rho = rho
        self.q = q
        self.pheromone_callback = pheromone_callback
        self.best_tour = None  

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
            visited = [random.randint(0, cities_count - 1)]
            unvisited = list(set(range(cities_count)) - set(visited))

            for _ in range(cities_count - 1):
                current_city = visited[-1]
                probabilities = self._calculate_probabilities(current_city, unvisited, pheromones, distances)
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

    def optimize_step(self):
        if self.best_tour is None:
            cities_count = len(self.cities)
            self.pheromones = self._initialize_pheromones(cities_count)
            self.distances = self._initialize_distances(self.cities)
            self.best_solution = None

        solutions = self._construct_solutions(self.cities_count, self.pheromones, self.distances)
        current_best_solution = min(solutions, key=lambda x: x[1])
        if self.best_solution is None or current_best_solution[1] < self.best_solution[1]:
            self.best_solution = current_best_solution
            self.best_tour = self.best_solution[0]

        self.pheromones = (1 - self.rho) * self.pheromones
        self._update_pheromones(self.pheromones, solutions)

        # Call the pheromone_callback function, if provided
        if self.pheromone_callback:
            self.pheromone_callback(self.pheromones)

    def optimize(self, cities):
        self.cities = cities
        self.cities_count = len(cities)
        for _ in range(self.num_iterations):
            self.optimize_step()

        return self.best_tour, self._calculate_tour_length(self.best_tour, self.distances), self.pheromones



import matplotlib.pyplot as plt

def plot_solution(ax, cities, best_tour, pheromones, title="TSP Solution and Pheromone Trails"):
    ax.clear()
    cities_x = [city[0] for city in cities]
    cities_y = [city[1] for city in cities]

    tour_x = [cities_x[best_tour[i]] for i in range(len(best_tour))]
    tour_y = [cities_y[best_tour[i]] for i in range(len(best_tour))]

    ax.scatter(cities_x, cities_y, color='blue')
    ax.plot(tour_x + [tour_x[0]], tour_y + [tour_y[0]], color='red', linestyle='-', linewidth=1, marker='o')

    for i, city in enumerate(cities):
        ax.text(city[0] + 0.1, city[1] + 0.1, f"{i}", fontsize=12, color='black')

    for i in range(len(cities)):
        for j in range(i + 1, len(cities)):
            x_values = [cities_x[i], cities_x[j]]
            y_values = [cities_y[i], cities_y[j]]
            edge_weight = pheromones[i, j]
            ax.plot(x_values, y_values, color='gray', linestyle='--', linewidth=edge_weight, alpha=0.5)

    ax.set_title(title)
    ax.set_xlabel("X-axis")
    ax.set_ylabel("Y-axis")
    ax.grid()

















def update_plot(iteration, aco):
    global aco
    aco.optimize_step()
    plot_solution(ax, cities, aco.best_tour, aco.pheromones, title=f"TSP Solution and Pheromone Trails at Iteration {iteration}")




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
        (1, -12)
    ]

    # Define the ACO parameters
    num_ants = 10
    num_iterations = 100
    alpha = 1
    beta = 5
    rho = 0.5
    q = 100

    fig, ax = plt.subplots()
    aco = AntColonyOptimizer(num_ants, num_iterations, alpha, beta, rho, q)
    ani = FuncAnimation(fig, update_plot, frames=num_iterations, fargs=(aco,), repeat=False)
    plt.show()
