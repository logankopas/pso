"""
Particle Swarm Optimization algorithm for finding the minimum of a cost function
More information can be found here: https://en.wikipedia.org/wiki/Particle_swarm_optimization
"""

import typing
from dataclasses import dataclass
import operator
from numpy import random
import numpy as np
import rastrigin
import plot


class ParticleSwarm:
    """Initializes and manages Swarm"""
    def __init__(self,
                 n_particles: int,
                 cost_function: typing.Callable,
                 x_bounds: tuple[float, float],
                 y_bounds: tuple[float, float],
                 random_seed: typing.Optional[float] = None,
                 plot: bool = False):
        self.n_particles = n_particles
        self.cost_function = cost_function
        self.x_bounds = x_bounds
        self.y_bounds = y_bounds
        self.global_best_z_value = np.inf
        self.global_best_position = (np.inf, np.inf)
        self.plot = plot
        self._particles = []
        self._iteration = 0
        self._tick_distance = 0.12 # for plotting

        # hyperparameters
        # Values set based on Clerc and Kennedy, more info here: 
        # https://towardsdatascience.com/particle-swarm-optimization-visually-explained-46289eeb2e14
        self._inertia = 0.8
        self._cognitive_coefficient = 2.05
        self._social_coefficient = 2.05


        if random_seed is not None:
            random.seed(random_seed)

    def initialize_swarm(self):
        """Initialize all particles and global values"""
        for _ in range(self.n_particles):
            position = tuple(random.uniform(*self.x_bounds, 2))
            velocity = tuple(random.uniform(*self.x_bounds, 2))
            z_value = self.cost_function(*position)

            self._particles.append(Particle(
                position=position,
                velocity=velocity,
                best_position=position,
                best_z_value=z_value
            ))
            if z_value < self.global_best_z_value:
                self.global_best_z_value = z_value
                self.global_best_position = position
        
        if self.plot:
            plot.plot_function(self.cost_function, self.x_bounds, self.y_bounds, 
                               self._tick_distance, filename='0.png')

    def step_all_particles(self):
        """Step all particles through the next iteration"""
        # This could be sped up through vectorization
        
        self._iteration += 1
        bounds = [self.x_bounds, self.y_bounds]
        for particle in self._particles:
            new_velocity = [np.inf, np.inf]
            new_position = [np.inf, np.inf]

            # Determine new velocity (stochastic) and update position accordingly
            for _dimension in range(2):
                r_cognitive, r_social = random.uniform(0, 1, 2) 
                new_velocity[_dimension] = (
                    (self._inertia * particle.velocity[_dimension])
                    + (self._cognitive_coefficient * r_cognitive * 
                       (particle.best_position[_dimension] - particle.position[_dimension]))
                    + (self._social_coefficient * r_social * 
                       (self.global_best_position[_dimension] - particle.position[_dimension]))
                )
                new_position[_dimension] = particle.position[_dimension] + new_velocity[_dimension]
                
                # keep it within the bounds
                lower_bound, upper_bound = bounds[_dimension]
                new_position[_dimension] = max(new_position[_dimension], lower_bound)
                new_position[_dimension] = min(new_position[_dimension], upper_bound)
                

            particle.position = new_position
            particle.velocity = new_velocity

            # Update bests for each particle

            z_value = self.cost_function(*particle.position)
            if z_value < particle.best_z_value:
                particle.best_z_value = z_value
                particle.best_position = particle.position
        
        # Update global best
        best_particle = sorted(self._particles, key=lambda x: x.best_z_value)[0]
        if best_particle.best_z_value < self.global_best_z_value:
            self.global_best_z_value = best_particle.best_z_value
            self.global_best_position = best_particle.best_position

        if plot:
            plot.plot_function(
                self.cost_function, self.x_bounds, self.y_bounds, self._tick_distance,
                [[*p.position, self.cost_function(*p.position)] for p in self._particles], f'{self._iteration}.png'
            )

    def search(self, n_iterations: int):
        for _ in range(n_iterations):
            self.step_all_particles()
        
        if self.plot: 
            plot.create_animation()

        return self.global_best_z_value, self.global_best_position


@dataclass
class Particle:
    """Holds data pertaining to a particle"""
    position: tuple[float, float]
    velocity: tuple[float, float]
    best_z_value: float
    best_position: tuple[float, float]


if __name__ == '__main__':
    # Development testing
    ps = ParticleSwarm(
        n_particles=10,
        cost_function=rastrigin.rastrigin_2d,
        x_bounds=(-5.12, 5.12),
        y_bounds=(-5.12, 5.12),
        random_seed=17,
        plot=True
    )
    ps.initialize_swarm()
    print(ps.global_best_z_value)
    print(ps.search(20))