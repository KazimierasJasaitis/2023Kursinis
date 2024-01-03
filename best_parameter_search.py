import hso
import pso
import csv

def run_hso(parameters):
    # Setup the constant parameters
    paper_size = (10, 10)
    image_sizes = [[500, 500], [500, 500], [500, 500], [500, 500]]
    dimensions = 3 * len(image_sizes)
    desired_fitness = 0

    # Initialize HarmonySearch with both constant and variable parameters
    hs = hso.HarmonySearch(HM_size=parameters['HM_size'], 
                           dimensions=dimensions, 
                           image_sizes=image_sizes, 
                           paper_size=paper_size, 
                           desired_fitness=desired_fitness, 
                           memory_consideration_rate=parameters['memory_consideration_rate'], 
                           pitch_adjustment_rate=parameters['pitch_adjustment_rate'])
    best_position = hs.run()
        # Open CSV file in append mode
    with open("parameterTestResults.csv", 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([parameters['HM_size'], parameters['memory_consideration_rate'], parameters['pitch_adjustment_rate'], hs.best_fitness])

    return hs.best_fitness

def run_pso(population_size, dimensions, image_sizes, paper_size, desired_fitness, w, c1, c2):
    # Initialize the PSO instance with the given parameters
    pso = pso.PSO(population_size=population_size, dimensions=dimensions, image_sizes=image_sizes, paper_size=paper_size, desired_fitness=desired_fitness, w=w, c1=c1, c2=c2)

    # Run the PSO algorithm
    best_position = pso.run()
    print("\nBest Fitness:", pso.gbest_fitness)

    # Process and print the results
    best_position_2d = best_position.reshape(-1, 3)
    for i, (x, y, scale) in enumerate(best_position_2d):
        print(f"Image {i+1}: x = {round(x)}, y = {round(y)}, scale = {round(scale, 2)}")

    # Open CSV file in append mode
    with open("parameterTestResults.csv", 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([population_size, w, c1, c2, pso.gbest_fitness, pso.iterations])
    # Return the best position and other relevant information
    return best_position, pso.gbest_fitness, pso.iterations


def find_best_parameters(algorithm):
    best_parameters = None
    best_performance = float('inf')

    # Define bounds and steps for each parameter
    if algorithm == 'hso':
        HM_size_range = range(10, 100, 10)  # Example: 50 to 150 with step 50
        memory_consideration_rate_range = np.arange(0.7, 0.95, 0.05)  # Example: 0.5 to 0.8 with step 0.1
        pitch_adjustment_rate_range = np.arange(0.1, 0.5, 0.1)  # Example: 0.2 to 0.5 with step 0.1

        for HM_size in HM_size_range:
            for memory_consideration_rate in memory_consideration_rate_range:
                for pitch_adjustment_rate in pitch_adjustment_rate_range:
                    params = {
                        'HM_size': HM_size,
                        'memory_consideration_rate': memory_consideration_rate,
                        'pitch_adjustment_rate': pitch_adjustment_rate,
                    }
                    
                    performance = run_hso(params)
                    if performance < best_performance:
                        best_performance = performance
                        best_parameters = params

    elif algorithm == 'pso':
        population_size_range = range(20, 50, 5)  # Example: 30 to 100 with step 10
        w_range = np.arange(0.9, 1.2, 0.05)  # Example: 0.4 to 0.8 with step 0.1
        c1_range = np.arange(1.5, 2.0, 0.1)  # Example: 1.0 to 2.0 with step 0.5
        c2_range = np.arange(1.5, 2.0, 0.1)  # Example: 1.0 to 2.0 with step 0.5

        for population_size in population_size_range:
            for w in w_range:
                for c1 in c1_range:
                    for c2 in c2_range:
                        params = {
                            'population_size': population_size,
                            'w': w,
                            'c1': c1,
                            'c2': c2,
                        }
                        performance = run_pso(params)
                        if performance < best_performance:
                            best_performance = performance
                            best_parameters = params

    return best_parameters, best_performance

# Example usage
if __name__ == "__main__":
    import numpy as np

    best_hso_params, best_hso_perf = find_best_parameters('hso')
    print("Best HSO Parameters:", best_hso_params, "Performance:", best_hso_perf)

    best_pso_params, best_pso_perf = find_best_parameters('pso')
    print("Best PSO Parameters:", best_pso_params, "Performance:", best_pso_perf)
