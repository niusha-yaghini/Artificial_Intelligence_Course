import numpy as np


class Particle:
    def __init__(
        self,
        dimension,
        bounds,
    ):
        # Step 1: Initialize Position
        self.position = np.array(
            [
                np.random.uniform(
                    low,
                    high
                )
                for low, high in bounds
            ]
        )

        # Step 2: Initialize Velocity
        self.velocity = np.zeros(
            dimension
        )

        # Step 3: Current Fitness
        self.fitness = np.inf

        # Step 4: Personal Best
        self.best_position = self.position.copy()
        self.best_fitness = np.inf
        
        
class Swarm:
    def __init__(
        self,
        population_size,
        dimension,
        bounds,
        fitness_function,
    ):
        self.bounds = bounds
        self.particles = []
        self.global_best_position = None
        self.global_best_fitness = np.inf

        # Step 1: Create Particles
        for _ in range(
            population_size
        ):
            particle = Particle(
                dimension,
                bounds
            )
            self.particles.append(
                particle
            )

        # Step 2: Initial Evaluation
        self.evaluate(
            fitness_function
        )
        
    def evaluate(
        self,
        fitness_function,
    ):
        for particle in self.particles:
            # Current Fitness
            fitness = fitness_function(
                particle.position
            )
            particle.fitness = fitness

            # Update Personal Best
            if fitness < particle.best_fitness:
                particle.best_fitness = fitness
                particle.best_position = (
                    particle.position.copy()
                )

            # Update Global Best
            if fitness < self.global_best_fitness:
                self.global_best_fitness = fitness
                self.global_best_position = (
                    particle.position.copy()
                )
                
    def update_velocity(
        self,
        w=0.7,
        c1=2,
        c2=2,
    ):
        for particle in self.particles:
            # Random values
            r1 = np.random.random(
                len(particle.position)
            )
            r2 = np.random.random(
                len(particle.position)
            )

            # Inertia
            inertia = (
                w
                *
                particle.velocity
            )

            # Cognitive Component
            cognitive = (
                c1
                *
                r1
                *
                (
                    particle.best_position
                    -
                    particle.position
                )
            )

            # Social Component
            social = (
                c2
                *
                r2
                *
                (
                    self.global_best_position
                    -
                    particle.position
                )
            )

            # New Velocity
            particle.velocity = (
                inertia
                +
                cognitive
                +
                social
            )
    
    def update_position(
        self,
    ):
        for particle in self.particles:
            # Step 1: Move Particle
            particle.position = (
                particle.position
                +
                particle.velocity
            )

            # Step 2: Boundary Handling
            for i, (low, high) in enumerate(
                self.bounds
            ):
                if particle.position[i] < low:
                    particle.position[i] = low
                elif particle.position[i] > high:
                    particle.position[i] = high