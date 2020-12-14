import numpy as np
from numpy.linalg import norm
import json
import matplotlib.pyplot as plt
from operator import add
import math
import datetime
from iterative_hessian_sketch import iterative_hessian_sketch, get_optimal_step_size_and_momentum_parameters
from test_matrix_generator import create_fast_solver_test_matrix
import traceback


def generate_chart_config(A, b, true_x, m, solver_config, number_of_iterations, number_of_trials, epsilon=None, max_time=None, figure=1):
    chart_config = {}
    for solver in solver_config:
        # print(m, solver['name'])
        # --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # Initialize variables
        # --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        avg_residue, avg_error, avg_minimizer_error = [], [], []
        total_time, sketch_time, factor_time = 0, 0, 0
        true_residue = norm(np.matmul(A, true_x) - b, ord=2)**2
        # --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # Run solver
        # --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        for trial in range(number_of_trials):
            print('FIGURE', figure, m, solver['name'], trial)
            # For each trial initialize a new starting point. We want E[x_0] = 0.
            x_0 = np.random.normal(0, 1, size=(A.shape[1], 1))

            # Run 1 iteration of vanilla iterated hessian sketch to get another point
            x_list, H_t_pinv, total_time, sketch_time, factor_time = iterative_hessian_sketch(np.copy(A), np.copy(b), m, np.copy(x_0), 1, step_size=solver['step_size'],
                                                                                              sketch=solver['sketch'], fixed=solver['fixed'], return_h=solver['fixed'],
                                                                                              epsilon=epsilon, max_iter=1)

            x_list, _, total_time, sketch_time, factor_time = iterative_hessian_sketch(np.copy(A), np.copy(b), m, np.copy(x_list[1]), number_of_iterations - 1, step_size=solver['step_size'],
                                                                                       sketch=solver['sketch'], fixed=solver['fixed'],
                                                                                       x_0=np.copy(x_list[0]), iteration_type='heavy_ball', x_list=x_list[:], h_t_pinv=np.copy(H_t_pinv),
                                                                                       total_time=total_time, sketch_time=sketch_time, factor_time=factor_time, epsilon=epsilon, max_time=max_time)

            # Residue
            residue = x_list[:]
            residue = [norm(np.matmul(A, residue[i]) - np.copy(b), ord=2) ** 2 for i in range(len(residue))]

            # Get prediction error ||A(x_t - x*)||_2^2 for each iteration
            prediction_error = x_list[:]
            prediction_error = [norm(np.matmul(A, prediction_error[i] - np.copy(true_x)), ord=2) ** 2 for i in range(len(prediction_error))]

            # Get minimizer error ||x-x*||_2^2 for each iteration
            minimizer_error = x_list[:]
            minimizer_error = [norm(minimizer_error[i] - np.copy(true_x), ord=2) ** 2 for i in range(len(minimizer_error))]

            # For every trial, add the error at every iteration.
            if avg_error:
                avg_error = list(map(add, avg_error, prediction_error[:]))
            else:
                avg_error = prediction_error[:]

            if avg_minimizer_error:
                avg_minimizer_error = list(map(add, avg_minimizer_error, minimizer_error[:]))
            else:
                avg_minimizer_error = minimizer_error[:]

            if avg_residue:
                avg_residue = list(map(add, avg_residue, residue[:]))
            else:
                avg_residue = residue[:]

        # For every solver, divide by the (number of trials * n) to get the normalized error.
        chart_config[solver['name']] = {'avg_error': [i / (number_of_trials * A.shape[0]) for i in avg_error],
                                        'avg_minimizer_error': [i / number_of_trials for i in avg_minimizer_error],
                                        'true_residue': true_residue,
                                        'avg_residue': [(i / (number_of_trials * true_residue)) for i in avg_residue],
                                        'total_time': total_time / number_of_trials, 'sketch_time': sketch_time / number_of_trials, 'factor_time': factor_time / number_of_trials}
    return chart_config


def generate_figure_1(n, d, alpha, sketch, solver_list, working_directory, number_of_iterations, number_of_trials, sparse=False):
    # 1. Get test matrices
    A, true_x, b = create_fast_solver_test_matrix(n, d, working_directory, sparse=sparse)
    overall_chart_config = []
    m = alpha * d

    # 2. Alpha is the control parameter that determines the size of the sketch
    step_size_list = [0.001, 0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 0.999]

    for solver in solver_list:
        _, opt_step_size = get_optimal_step_size_and_momentum_parameters(sketch, solver.split(' ')[0] == 'Fixed', True, n, d, m)
        opt_step_size_list = sorted(set([opt_step_size] + step_size_list))
        solver_config = [{'name': '{} {}'.format(solver, step_size), 'sketch': sketch, 'fixed': True, 'step_size': step_size} for step_size in opt_step_size_list]

        # Generate chart_config. Chart config has also the details necessary to make a chart.
        chart_config = generate_chart_config(np.copy(A), np.copy(b), np.copy(true_x), m, solver_config, number_of_iterations, number_of_trials)

        # Save chart_config for figure generation
        overall_chart_config += [{'opt_step_size': opt_step_size, 'solver': solver, 'chart_config': chart_config}]
        open(working_directory + 'Chart_Config_Figure_1_Step_size.json', 'w').write(json.dumps({'overall_chart_config': overall_chart_config}))


def generate_figure_2_and_4(n, d, solver_config, working_directory, number_of_iterations, number_of_trials, sparse=False):
    # 1. Get test matrices
    A, true_x, b = create_fast_solver_test_matrix(n, d, working_directory, sparse=sparse)

    # 2. Alpha is the control parameter that determines the size of the sketch
    overall_chart_config = []
    for alpha in range(4, 44, 4):
        m = alpha * d

        # Generate chart_config. Chart config has also the details necessary to make a chart.
        chart_config = generate_chart_config(np.copy(A), np.copy(b), np.copy(true_x), m, solver_config, number_of_iterations, number_of_trials)

        # Save chart_config for figure generation
        overall_chart_config += [{'alpha': alpha, 'chart_config': chart_config}]
        open(working_directory + 'Chart_Config_Figure_2_and_4.json', 'w').write(json.dumps({'overall_chart_config': overall_chart_config}))

    # 3. Create figure 1
    def create_sub_figures_1(metric, x_axis, image_name, figure_title=None, y_label=None, x_label='Control Parameter α', x_line=False, y_line=False, tol=None):
        # Close any open plots
        plt.clf()

        if x_line:
            plt.axhline(y=tol, ls='--', color='grey')

        max_idx = 0
        for solver in list(overall_chart_config[0]['chart_config'].keys()):
            avg_error = []
            for chart in overall_chart_config:
                avg_error.append(chart['chart_config'][solver][metric][-1])

            if tol is not None:
                min_sketch_size = [i for i in avg_error if i < tol]
                if min_sketch_size:
                    new_value = (avg_error.index(min_sketch_size[0]) + 1) * 4
                    max_idx = max(max_idx, new_value)

            plt.plot(x_axis, avg_error, label=solver)

        if y_line:
            plt.axvline(x=max_idx, ls='--', color='grey')

        if figure_title is not None:
            plt.title(figure_title)
        if x_label is not None:
            plt.xlabel(x_label)
        if y_label is not None:
            plt.ylabel(y_label)

        plt.yscale("log")
        plt.legend(ncol=2, loc='upper right')

        plt.savefig(working_directory + '{}.png'.format(image_name), dpi=1000)

    x_axis = [i['alpha'] for i in overall_chart_config]
    create_sub_figures_1('avg_error', x_axis[:], 'Figure_2_a_Prediction_Error_vs_Sketch_Size', y_label='Prediction Error', x_line=True, y_line=True, tol=1e-6)
    create_sub_figures_1('avg_minimizer_error', x_axis[:], 'Figure_2_b_Error_vs_Sketch_Size', y_label='Error', x_line=True, y_line=True, tol=1e-3)
    create_sub_figures_1('avg_residue', x_axis[:], 'Figure_2_c_Cost_Approximation_vs_Sketch_Size', y_label='Epsilon ε')

    # 4. Create figure 3
    def create_time_profile_figures(solver, x_axis, image_name):
        # Close any open plots
        plt.clf()

        y_1, y_2, y_3, y_4 = [], [], [], []
        for chart in overall_chart_config:
            y_1.append(chart['chart_config'][solver]['sketch_time'])
            y_2.append(chart['chart_config'][solver]['factor_time'])
            y_3.append(chart['chart_config'][solver]['total_time'] - chart['chart_config'][solver]['factor_time'] - chart['chart_config'][solver]['sketch_time'])
            y_4.append(chart['chart_config'][solver]['total_time'])

        plt.plot(x_axis, y_1, label='Sketch')
        plt.plot(x_axis, y_2, label='Factor')
        plt.plot(x_axis, y_3, label='Iterate')
        plt.plot(x_axis, y_4, label='Total')

        plt.title('{} Time Profile'.format(solver))
        plt.xlabel('Control Parameter α')
        plt.ylabel('Time (in seconds)')

        plt.legend(loc='center right', ncol=2)
        plt.savefig(working_directory + '{}.png'.format(image_name), dpi=1000)

    solver_list = list(overall_chart_config[0]['chart_config'].keys())
    for solver in solver_list:
        create_time_profile_figures(solver, x_axis[:], 'Figure_4_{}_Time_Profile'.format(solver.replace(' ', '_')))


def generate_figure_3(d, working_directory, solver_config, number_of_trials, alpha, number_of_iterations, sparse=False):
    m = alpha * d
    for n in [128, 256, 512, 1024, 2048, 4096, 8192, 16384]:
        # print('-'*40)
        # print('n = {}'.format(n))
        # print('-'*40)
        # 1. Get test matrices
        A, true_x, b = create_fast_solver_test_matrix(n, d, working_directory, sparse=sparse)

        # Generate chart_config. Chart config has also the details necessary to make a chart.
        chart_config = generate_chart_config(np.copy(A), np.copy(b), np.copy(true_x), m, solver_config, number_of_iterations, number_of_trials, figure=3)

        # Save chart_config for figure generation
        open(working_directory + 'Figure_3_Config_n_{}.json'.format(n), 'w').write(json.dumps({'n': n, 'chart_config': chart_config}))

    # Specify x-axis and solver list
    x_list = [2 ** (7 + n_idx) for n_idx in range(8)]
    solver_list = [i['name'] for i in solver_config]

    def create_sub_figures(metric, y_label, image_name):
        # Close any open plots
        plt.clf()

        # For each solver get error level after ceil(1 + log(n)) iterations
        for solver in solver_list:
            y_list = []
            for n_idx in range(8):
                n = 2 ** (7 + n_idx)
                with open(working_directory + 'Figure_3_Config_n_{}.json'.format(n)) as f:
                    chart_config = json.load(f)['chart_config']
                no_of_iter = math.ceil(1 + np.log(n))
                y_list.append(chart_config[solver][metric][no_of_iter - 1])

            plt.plot(x_list, y_list, label=solver)

        plt.yscale("log")
        plt.legend(loc='lower left', ncol=2)
        plt.ylabel(y_label)
        plt.xlabel('Row dimension of data matrix n')
        plt.savefig(working_directory + '{}.png'.format(image_name), dpi=1000)

    create_sub_figures('avg_error', 'Prediction_Error', 'Figure_3_a_Prediction_Error_vs_n')
    create_sub_figures('avg_minimizer_error', 'Error', 'Figure_3_b_Error_vs_n')


def generate_figure_5(n, d, alpha, solver_config, working_directory, number_of_iterations, number_of_trials, sparse=False):
    # 1. Get test matrices
    A, true_x, b = create_fast_solver_test_matrix(n, d, working_directory, sparse=sparse)

    # 2. Alpha is the control parameter that determines the size of the sketch
    m = alpha * d

    # Generate chart_config. Chart config has also the details necessary to make a chart.
    overall_chart_config = generate_chart_config(np.copy(A), np.copy(b), np.copy(true_x), m, solver_config, number_of_iterations, number_of_trials, epsilon=1e-12, max_time=3, figure=5)

    # Save chart_config for figure generation
    open(working_directory + 'Chart_Config_Figure_5.json', 'w').write(json.dumps({'overall_chart_config': overall_chart_config}))

    def create_sub_figures_4(metric, image_name, x_label, y_label):
        # Close any open plots
        plt.clf()

        for solver in list(overall_chart_config.keys()):
            y_values = overall_chart_config[solver][metric][:300]
            plt.plot(range(1, len(y_values) + 1), y_values, label=solver)

        if x_label is not None:
            plt.xlabel(x_label)
        if y_label is not None:
            plt.ylabel(y_label)

        plt.yscale("log")

        plt.legend()

        plt.savefig(working_directory + '{}.png'.format(image_name), dpi=1000)

    create_sub_figures_4('avg_error', 'Figure_5_a_Error_vs_Iterations', x_label='Iteration Count', y_label='Prediction Error')
    create_sub_figures_4('avg_minimizer_error', 'Figure_5_b_Error_vs_Iterations', x_label='Iteration Count', y_label='Error')


def generate_figure_6(working_directory, solver_config, number_of_trials, n, d, alpha, number_of_iterations, sparse=False):
    m = alpha * d
    # 128, 256, 512, 1024, 2048,
    for r in range(10):
        # print('-'*40)
        # print('r = {}'.format(r))
        # print('-'*40)
        # 1. Get test matrices
        A, true_x, b = create_fast_solver_test_matrix(n, d, working_directory, condition_number=10**r, sparse=sparse)

        # Generate chart_config. Chart config has also the details necessary to make a chart.
        chart_config = generate_chart_config(np.copy(A), np.copy(b), np.copy(true_x), m, solver_config, number_of_iterations, number_of_trials, epsilon=1e-16)

        # Save chart_config for figure generation
        open(working_directory + 'Figure_6_Config_r_{}.json'.format(r), 'w').write(json.dumps({'r': r, 'chart_config': chart_config}))


def generate_figures(n, d, alpha, working_directory, number_of_iterations, number_of_trials, sparse=False):
    if sparse:
        # -------------------------------------------------------------------------------------------------------------------------------------------------
        # Sparse JL for sparse matrices
        # -------------------------------------------------------------------------------------------------------------------------------------------------
        solver_config = [
            {'name': 'Refreshed Sparse JL', 'sketch': 'Sparse JL', 'fixed': False, 'step_size': None},
            {'name': 'Fixed Sparse JL', 'sketch': 'Sparse JL', 'fixed': True, 'step_size': None},
        ]
        generate_figure_1(n, d, alpha, 'Sparse JL', ['Refreshed Sparse JL', 'Fixed Sparse JL'], working_directory, number_of_iterations, number_of_trials, sparse=True)
    else:
        # -------------------------------------------------------------------------------------------------------------------------------------------------
        # Gaussian and SRHT Sketches for sparse matrices
        # -------------------------------------------------------------------------------------------------------------------------------------------------
        solver_config = [
            {'name': 'RG', 'sketch': 'Gaussian', 'fixed': False, 'step_size': None},
            {'name': 'FG', 'sketch': 'Gaussian', 'fixed': True, 'step_size': None},
            {'name': 'FS (b)', 'sketch': 'SRHT', 'fixed': True, 'step_size': None},
            {'name': 'RS (b)', 'sketch': 'SRHT', 'fixed': False, 'step_size': None},
            {'name': 'FS (p)', 'sketch': 'SRHT', 'fixed': True, 'step_size': 0.7},
            {'name': 'RS (p)', 'sketch': 'SRHT', 'fixed': False, 'step_size': 0.7},
        ]
        generate_figure_1(n, d, alpha, 'SRHT', ['Refreshed SRHT', 'Fixed SRHT'], working_directory, number_of_iterations, number_of_trials)
    #     generate_figure_6(working_directory, solver_config, number_of_trials, n, d, alpha, number_of_iterations)

    generate_figure_2_and_4(n, d, solver_config, working_directory, number_of_iterations, number_of_trials, sparse=sparse)
    generate_figure_3(d, working_directory, solver_config, number_of_trials, alpha=alpha, number_of_iterations=number_of_iterations, sparse=sparse)
    generate_figure_5(n, d, alpha, solver_config, working_directory, number_of_iterations, number_of_trials, sparse=sparse)


generate_figures(n=4096, d=10, alpha=12, working_directory='C:/Users/amrit/Desktop/Scripts/', number_of_iterations=20, number_of_trials=50, sparse=False)
