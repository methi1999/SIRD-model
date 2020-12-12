import yaml
import pickle
import random
import os
from timeit import default_timer as timer

import numpy as np
from matplotlib.lines import Line2D

from graph import watts_strogatz_graph as ws_graph
from plot import plot_compare_evolution, plot_simple_evolution, graph_animate, get_plot_title

# manually set seed to reproduce results
np.random.seed(42)
random.seed(42)


def SIRD_graph(graph_obj, config):
    """
    Simulates a SIRD model on the input graph
    Parameters
    ----------
    graph_obj: Input networkX graph object
    config: config file

    Returns a TXN matrix where element (i, j) is the state of node j at time step i
    -------
    """

    N = len(graph_obj.nodes)
    T = config['horizon']
    si, ir, id = config['infection_rate'], config['recovery_rate'], config['death_rate']
    sv, iv, vi = config['vaccination_rate'], config['vaccination_rate'], config['reinfection_rate']

    # ensure that sum of probabilities don't exceed 1
    assert ir + id + iv < 1
    assert si + sv < 1

    # initial state
    I0, R0, D0, V0 = config['init_state']['I'], config['init_state']['R'], config['init_state']['D'], config['init_state']['V']
    S0 = N - I0 - R0 - D0
    assert S0 > 0

    # shuffle node list and assign initial state
    shuffled_nodes = np.array(range(N))
    np.random.shuffle(shuffled_nodes)

    # state array is a Nx1 vector with array[i] denoting state of ith person
    # states for S,I,R,D,V -> (0,1,2,3,4)
    category_array = np.zeros(N, dtype=np.uint8)
    category_array[shuffled_nodes[:I0]] = 1
    category_array[shuffled_nodes[I0: I0+R0]] = 2
    category_array[shuffled_nodes[I0+R0: I0+R0+D0]] = 3
    category_array[shuffled_nodes[I0+R0+D0: I0+R0+D0+V0]] = 4

    # state array [i][j] = category of node j after time step i
    state_array = np.zeros((T + 1, N))
    state_array[0] = np.copy(category_array)

    for t in range(1, T + 1):
        if t % (T // 3) == 0:
            print('Time step:', t, '/', T)

        cur_categories = np.copy(state_array[t - 1])

        for node in graph_obj.nodes:
            # initialise new state with old state
            new_state = cur_categories[node]
            neighbours = graph_obj.adj[node]
            i_neighbours = np.sum(cur_categories[neighbours] == 1)

            if cur_categories[node] == 0:  # susceptible
                prob_i = si * (i_neighbours / len(neighbours))
                # S can either be vaccinated or become infected
                next_possible_states = [1, 4, 0]
                next_possible_probs = [prob_i, sv, 1 - prob_i - sv]
                new_state = np.random.choice(a=next_possible_states, size=1, p=next_possible_probs)

            elif cur_categories[node] == 1:  # infectious
                # I can either recover or die
                next_possible_states = [2, 3, 4, 1]
                next_possible_probs = [ir, id, iv, 1 - ir - id - iv]
                new_state = np.random.choice(a=next_possible_states, size=1, p=next_possible_probs)

            elif cur_categories[node] == 4:  # vaccinated
                prob_r = vi * (i_neighbours / len(neighbours))
                next_possible_states = [1, 4]
                next_possible_probs = [prob_r, 1 - prob_r]
                new_state = np.random.choice(a=next_possible_states, size=1, p=next_possible_probs)

            # do nothing for recovered and dead

            # assign new state to current time vector
            cur_categories[node] = new_state

        # store new time step results in final array
        state_array[t] = cur_categories

    return state_array


def simulate_and_simple_plot(config):
    """
    Simulates and plots by calling the 2 functions
    Parameters
    ----------
    config: config file

    Returns nothing. Plots final simulation results as a function of time
    -------

    """
    # define network
    N, k, rewire_p = config['population_size'], config['ws']['neighbours'], config['ws']['rewire_p']
    num_runs = config['num_runs']
    G = ws_graph(N, k, rewire_p)

    # load pickle
    load_previous = config['load_pickle']
    sim_name = 'simple_run_' + str(num_runs)
    pkl_path = os.path.join(config['dir']['pickle'], sim_name + '.pkl')
    img_path = os.path.join(config['dir']['img'], sim_name + '.png')
    found_previous = False

    states = []

    if load_previous and os.path.exists(pkl_path):
        states, dumped_config = pickle.load(open(pkl_path, 'rb'))
        # ensure that configs are the same
        if dumped_config == config:
            print("Loaded pickle dump")
            found_previous = True

    # if did not find pickle dump, simulate
    if not found_previous:
        res = []
        for i in range(num_runs):
            print("Simulation Num:", i + 1, '/', num_runs)
            res.append(SIRD_graph(G, config))
        states = np.array(res)
        # dump pickle
        pickle.dump((states, config), open(pkl_path, 'wb'))

    # plot time evolution
    plot_simple_evolution(config, states, plot_average=True, save_pth=img_path)

    # animate graph
    graph_animate(config, G, states[0], save_pth=img_path[:-3]+'gif')


def simulate_and_compare_plot(config, parameter_name, parameter_values, marker_list: list):
    """
    Plots simulation curves on the same plot so that values can be compared
    Parameters
    ----------
    config: main config file
    parameter_name: the parameter to be varied
    parameter_values: values of parameters
    marker_list: list of markers for each parameter value e.g. ['x', 'o']
    -------

    Returns nothing. Plots final simulation results as a function of time
    """
    # detect typos
    assert parameter_name in config.keys()

    # define graph if parameters are not to be varied
    if parameter_name != 'population_size':
        N, k, rewire_p = config['population_size'], config['ws']['neighbours'], config['ws']['rewire_p']
        G = ws_graph(N, k, rewire_p)

    config_list = []
    for v in parameter_values:
        config_copy = dict(config)
        config_copy[parameter_name] = v
        config_list.append(config_copy)

    assert len(marker_list) == len(config_list)
    # print(config_list)

    num_runs = config['num_runs']

    # generating the title
    title = get_plot_title(config)
    title += '\nVarying ' + parameter_name + " - "
    marker_des = [Line2D.markers[x] for x in marker_list]
    for i, marker in enumerate(marker_des):
        title += marker + ':' + str(parameter_values[i]) + '; '

    load_previous = config['load_pickle']
    sim_name = 'compare_' + parameter_name + '_' + str(num_runs)
    pkl_path = os.path.join(config['dir']['pickle'], sim_name + '.pkl')
    img_path = os.path.join(config['dir']['img'], sim_name + '.png')
    found_previous = False

    if load_previous and os.path.exists(pkl_path):
        state_list, dumped_config_list = pickle.load(open(pkl_path, 'rb'))
        # ensure that configs are the same
        if dumped_config_list == config_list:
            print("Loaded pickle dump")
            found_previous = True

    # if did not find pickle dump, simulate
    if not found_previous:
        state_list = []
        for cur_c in config_list:
            print("Setting parameter {} to {}".format(parameter_name, cur_c[parameter_name]))

            if parameter_name == 'population_size':
                N, k, rewire_p = cur_c['population_size'], cur_c['ws']['neighbours'], cur_c['ws']['rewire_p']
                G = ws_graph(N, k, rewire_p)

            runs = []
            start = timer()
            for i in range(num_runs):
                print("Simulation Num:", i + 1, '/', num_runs)
                runs.append(SIRD_graph(G, cur_c))
            end = timer()
            print((end-start)/num_runs)
            state_list.append(np.array(runs))
        # dump pickle
        pickle.dump((state_list, config_list), open(pkl_path, 'wb'))

    # plot
    plot_compare_evolution(state_list, marker_list, config_list, title, plot_average=True, save_pth=img_path)

    # animate graph
    graph_animate(config, G, state_list[0][0], save_pth=img_path[:-3]+'gif')


if __name__ == '__main__':
    # load config file
    config = yaml.load(open('config.yaml', 'r'), Loader=yaml.FullLoader)

    # basic
    simulate_and_simple_plot(config)

    # compare
    # simulate_and_compare_plot(config, 'infection_rate', [0.5, 0.6, 0.7], ['x', 'o', 'v'])
    # simulate_and_compare_plot(config, 'recovery_rate', [0.1, 0.15, 0.2], ['x', 'o', 'v'])
    # simulate_and_compare_plot(config, 'vaccination_rate', [0, 0.0001, 0.0005], ['x', 'o', 'v'])
    # simulate_and_compare_plot(config, 'population_size', [200, 500, 1000, 5000], ['x', 'o', 's', 'v'])