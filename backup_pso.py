import numpy as np

class Particle:
    
    def __init__(self, dimensions, image_sizes, paper_size):
        # Number of images
        position_count = dimensions // 3
        x_coordinates = np.empty(position_count)
        y_coordinates = np.empty(position_count)
        # Calculate total x and y lengths of all images
        sum_x_lengths = np.sum([size[0] for size in image_sizes])
        sum_y_lengths = np.sum([size[1] for size in image_sizes])
        # Calculate a single scale value for all images
        scale_value = min(paper_height / sum_x_lengths, paper_width / sum_y_lengths)

        for i in range(position_count):
            # Calculate x_max and y_max for each image
            x_max = paper_height - image_sizes[i][0]
            y_max = paper_width - image_sizes[i][1]

            # Randomly initialize x and y coordinates
            x_coordinates[i] = np.random.uniform(0, x_max / scale_value)
            y_coordinates[i] = np.random.uniform(0, y_max / scale_value)

        # Merge x_coordinates, y_coordinates, and scale_value into self.position
        self.position = np.empty(3 * position_count)
        self.position[0::3] = [0,50,0,50]
        self.position[1::3] = [0,0,50,50]
        self.position[2::3] = 0.1

        # Initialize velocity with random values between 0 and 1
        self.velocity = np.random.uniform(0, 1, dimensions)
        self.pbest_position = self.position
        self.pbest_fitness = float('inf')
        
    def compute_fitness(self, image_sizes, paper_size, scaling_penalty_factor=1, boundary_penalty_factor=1, overlap_penalty_factor=1, uncovered_area_penalty_factor=1):
        global printn
        paper_height, paper_width = paper_size
        total_area = paper_height * paper_width
        sum_image_areas = 0
        total_resizing_deviation = 0
        overlapping_area = 0
        boundary_penalty = 0
        overlapping_area_penalty = 0
        covered_area = 0
        covered_matrix = np.zeros(paper_size, dtype=bool)
        biggest_possible_overlap = 0

        positions = self.position.reshape(-1, 3)
        avg_scale = np.mean([scale for _, _, scale in positions])

        for i, (x, y, scale) in enumerate(positions):
            x=round(x)
            y=round(y)
            # Penalize negative or 0 scales
            if scale <= 0:
                fitness = float('inf')
                return fitness
            
            original_width, original_height = image_sizes[i]
            # Calculate the new dimensions of the image after resizing
            new_width = round(original_width * scale)
            new_height = round(original_height * scale)

            if new_width <= 0 or new_height <= 0: 
                fitness = float('inf') 
                return fitness 

            # Add to the sum of image areas
            image_area = new_width * new_height
            sum_image_areas += image_area

            # Check for overlaps with other images
            for j in range(i + 1, len(positions)):
                x2, y2, scale2 = positions[j]
                x2=round(x2)
                y2=round(y2)
                original_width2, original_height2 = image_sizes[j]
                new_width2 = round(original_width2 * scale2)
                new_height2 = round(original_height2 * scale2)
                if new_width2 <= 0 or new_height2 <= 0: 
                    fitness = float('inf') 
                    return fitness 

                overlap_height = min((y + new_height),y2 + new_height2)-max(y,y2)
                overlap_width = min((x+new_width),x2 + new_width2)-max(x,x2)

                overlapping_area += max(0,overlap_height * overlap_width)

                biggest_overlap_height = min(new_height,new_height2)
                biggest_overlap_width = min(new_width,new_width2)
                biggest_possible_overlap += biggest_overlap_height * biggest_overlap_width     

            # Check for out of boundary
            if (x + new_width > paper_height or y + new_height > paper_width or x < 0 or y < 0):
                in_bound_height = min((y+new_height),paper_height)-max(y,0)
                in_bound_width = min((x+new_width),paper_width)-max(x,0)
                # Calculate area inside the bounds
                in_bounds_area = in_bound_height * in_bound_width
                # Calculate total out-of-bound area
                out_of_bounds_area = image_area - in_bounds_area
                boundary_penalty += max(0,out_of_bounds_area)
                print(f"x:{x} x+new_width:{x+new_width} out of bounds:{out_of_bounds_area}")

            # Calculate the resizing deviation
            total_resizing_deviation += round(abs(avg_scale - scale) / (1/new_width)) # For each pixel that is out of place from the average scale scenario
            total_resizing_deviation += round(abs(avg_scale - scale) / (1/new_height)) # Same for width

            # Calculate uncovered area
            overlap = covered_matrix[y:y + new_height, x:x + new_width] # Check if the current image overlaps with already covered area
            covered_area += np.sum(~overlap) # Calculate the uncovered area for the current image
            covered_matrix[y:y + new_height, x:x + new_width] = True # Mark the newly covered area by the current image as True in the matrix

            uncovered_area = total_area - covered_area
            
        
        # Normalizing penalty weights
        total_resizing_deviation = total_resizing_deviation / 100
        boundary_penalty = boundary_penalty / ( total_area * len(image_sizes))
        uncovered_area_penalty = uncovered_area / total_area
        overlapping_area_penalty = overlapping_area / biggest_possible_overlap
        # Compute the penalties
        fitness =   total_resizing_deviation * scaling_penalty_factor + \
                    boundary_penalty * boundary_penalty_factor + \
                    uncovered_area_penalty * uncovered_area_penalty_factor + \
                    overlapping_area_penalty * overlap_penalty_factor
        #print("area: ",sum_image_areas,", overlapping: ", overlapping_area, "boundary: ", boundary_penalty )
        #if fitness < 0: print(f"Area: {sum_image_areas}, Overlapping: {overlapping_area}, Boundary: {boundary_penalty}")
        if printn: 
            print(f"1: {total_resizing_deviation}, 2: {boundary_penalty}, 3: {uncovered_area_penalty}, 4:{overlapping_area_penalty}") 
            print(x)
            printn = False

        return fitness
    
    def update_position(self):
        self.position = self.position + self.velocity

    def update_velocity(self, gbest_position, w, c1, c2):
        r1 = np.random.uniform(0, 1, len(self.position))
        r2 = np.random.uniform(0, 1, len(self.position))
        cognitive_velocity = c1 * r1 * (self.pbest_position - self.position)
        social_velocity = c2 * r2 * (gbest_position - self.position)
        self.velocity = w * self.velocity + cognitive_velocity + social_velocity


class PSO:
    def __init__(self, population_size, dimensions, image_sizes, paper_size, desired_fitness, w, c1, c2):
        
        self.population_size = population_size
        self.dimensions = dimensions
        self.image_sizes = image_sizes
        self.paper_size = paper_size
        self.desired_fitness = desired_fitness
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.gbest_position = np.zeros(dimensions)
        self.gbest_fitness = float('inf')
        self.population = [Particle(dimensions, image_sizes, paper_size) for _ in range(population_size)]
        self.iterations = 0
        self.iterations_without_improvement = 0

    def run(self):
        while self.gbest_fitness > self.desired_fitness:
            if self.iterations_without_improvement >= 1:
                break
            self.iterations_without_improvement += 1
            self.iterations += 1
            for particle in self.population:
                fitness = particle.compute_fitness(self.image_sizes, self.paper_size)
                if fitness < particle.pbest_fitness:
                    particle.pbest_fitness = fitness
                    particle.pbest_position = particle.position
                    
                    if fitness < self.gbest_fitness:
                        self.gbest_fitness = fitness
                        self.gbest_position = particle.position
                        self.iterations_without_improvement = 0

            #print(self.population[2].velocity[11],self.population[2].pbest_fitness,self.population[2].position)
            for particle in self.population:
                particle.update_velocity(self.gbest_position, self.w, self.c1, self.c2)
                particle.update_position()
        return self.gbest_position

# Example usage:
if __name__ == "__main__":
    printn = 1
    paper_width = 100
    paper_height = 100
    paper_size = (paper_width, paper_height)

    w = 0.7
    c1 = 1.5
    c2 = 1.5

    image_sizes = [[500,500],[500,500],[500,500],[500,500]]

    N = len(image_sizes)
    dimensions = 3 * N

    pso = PSO(population_size=1, dimensions=dimensions, image_sizes=image_sizes, paper_size=paper_size, desired_fitness=0.0001, w=w, c1=c1, c2=c2)

    best_position = pso.run()
    print(pso.gbest_fitness)
    best_position_2d = best_position.reshape(-1, 3)

    # Print each image's position and scale factor
    for i, (x, y, scale) in enumerate(best_position_2d):
        print(f"Image {i+1}: x = {round(x)}, y = {round(y)}, scale = {scale}")
    print(pso.iterations)