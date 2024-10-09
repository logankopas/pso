#!/usr/bin/env python
import argparse
import pso
import rastrigin

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='pso',
        description='Particle Swarm Optimizer for finding optimal solution to the Rastrigin function'
    )
    parser.add_argument(
        '--particles', default=10, type=int, help='The number of particles used in the search space')
    parser.add_argument(
        '--iterations', default=20, type=int, help='The number of iterations to find solution')
    parser.add_argument(
        '--plot', default=False, type=bool, action=argparse.BooleanOptionalAction,
        help='''Whether or not to produce display plots. Plots will be stored in the tmp/ directory. 
        This will slow down the program immensely.''')
    parser.add_argument(
        '--seed', default=None, type=int, help='Value to seed the random number generator with')
    
    args = parser.parse_args()

    swarm = pso.ParticleSwarm(
        n_particles = args.particles,
        cost_function = rastrigin.rastrigin_2d,
        x_bounds=(-5.12, 5.12),
        y_bounds=(-5.12, 5.12),
        random_seed=args.seed,
        plot=args.plot
    )

    swarm.initialize_swarm()

    best_value, best_position = swarm.search(args.iterations)
    expected_best_value = 0
    expected_best_position = (0, 0)

    print(f'True optimal value: {expected_best_value} at position {expected_best_position}')
    print(f'Optimal value found: {best_value} at position {best_position}')
    print(f'Actual error: {best_value - expected_best_value}')
    if args.plot:
        print('Plots produced in tmp/ folder')