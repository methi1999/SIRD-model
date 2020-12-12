import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib import animation


def plot_simple_evolution(config, states, plot_average, save_pth=None):
    """
    Plot the time evolution of the SIRD states
    Parameters
    ----------
    config
    states: TXN array where element (i,j) denotes state of node j at time i
    plot_average: average out the values if True
    save_pth: path for saving figure. None by default
    -------

    """
    N = config['population_size']
    # if only one run
    if states.ndim == 2:
        states = states[None, :, :]

    S = np.sum(states == 0, axis=2)
    I = np.sum(states == 1, axis=2)
    R = np.sum(states == 2, axis=2)
    D = np.sum(states == 3, axis=2)
    V = np.sum(states == 4, axis=2)

    # Define time grid
    t = list(range(states.shape[1]))

    fig = plt.figure(facecolor='w', figsize=(8, 6))
    ax = fig.add_subplot(111, facecolor='#dddddd', axisbelow=True)

    lines, labels, color_codes = get_legend()
    ax.legend(lines, labels)

    if plot_average:
        S, I, R, D, V = np.mean(S, axis=0), np.mean(I, axis=0), np.mean(R, axis=0), np.mean(D, axis=0), np.mean(V, axis=0)
        ax.plot(t, S / N, color_codes['S'], alpha=0.5, lw=2)
        ax.plot(t, I / N, color_codes['I'], alpha=0.5, lw=2)
        ax.plot(t, R / N, color_codes['R'], alpha=0.5, lw=2)
        ax.plot(t, D / N, color_codes['D'], alpha=0.5, lw=2)
        ax.plot(t, V / N, color_codes['V'], alpha=0.5, lw=2)
    else:
        for i in range(states.shape[0]):
            ax.plot(t, S[i] / N, color_codes['S'], alpha=0.5, lw=2)
            ax.plot(t, I[i] / N, color_codes['I'], alpha=0.5, lw=2)
            ax.plot(t, R[i] / N, color_codes['R'], alpha=0.5, lw=2)
            ax.plot(t, D[i] / N, color_codes['D'], alpha=0.5, lw=2)
            ax.plot(t, V[i] / N, color_codes['V'], alpha=0.5, lw=2)

    ax.set_xlabel('Time step')
    ax.set_ylabel('Fraction')

    plt.title(get_plot_title(config))

    ax.set_ylim(0, 1.2)
    ax.yaxis.set_tick_params(length=0)
    ax.xaxis.set_tick_params(length=0)
    ax.grid(b=True, which='major', c='w', lw=2, ls='-')

    for spine in ('top', 'right', 'bottom', 'left'):
        ax.spines[spine].set_visible(False)
    plt.show()

    if save_pth is not None:
        fig.savefig(save_pth)


def plot_compare_evolution(state_list, marker_list, config_list, title, plot_average, save_pth=None):
    """
    Plot SIRD evolution of 2 configurations which differ in one parameter value

    Parameters
    ----------
    state_list: list of TxN state vectors
    marker_list: list of markers to distinguish configurations
    config_list: list of configs. Should differ at the parameter value
    title: title of the plot. Marker list should be used to generate the plot
    plot_average: whether to plot average of curves. Mostly true since plot will become messy
    save_pth: if not None, save final plot

    Returns nothing. Simply plots the 2 configurations
    -------

    """
    fig = plt.figure(facecolor='w', figsize=(8, 6))
    ax = fig.add_subplot(111, facecolor='#dddddd', axisbelow=True)

    lines, labels, color_codes = get_legend()
    ax.legend(lines, labels)

    # iterate over various configurations
    for i, states in enumerate(state_list):
        marker = marker_list[i]
        config = config_list[i]

        N = config['population_size']
        # if only one run
        if states.ndim == 2:
            states = states[None, :, :]

        S = np.sum(states == 0, axis=2)
        I = np.sum(states == 1, axis=2)
        R = np.sum(states == 2, axis=2)
        D = np.sum(states == 3, axis=2)
        V = np.sum(states == 4, axis=2)

        # time grid
        t = list(range(states.shape[1]))
        # marker spacing
        total_marks = 20
        marker_spacing = S.shape[1] // total_marks

        if plot_average:
            S, I, R, D, V = np.mean(S, axis=0), np.mean(I, axis=0), np.mean(R, axis=0), np.mean(D, axis=0), np.mean(V, axis=0)
            ax.plot(t, S / N, color_codes['S'], alpha=0.5, lw=2, marker=marker, markevery=marker_spacing)
            ax.plot(t, I / N, color_codes['I'], alpha=0.5, lw=2, marker=marker, markevery=marker_spacing)
            ax.plot(t, R / N, color_codes['R'], alpha=0.5, lw=2, marker=marker, markevery=marker_spacing)
            ax.plot(t, D / N, color_codes['D'], alpha=0.5, lw=2, marker=marker, markevery=marker_spacing)
            ax.plot(t, V / N, color_codes['V'], alpha=0.5, lw=2, marker=marker, markevery=marker_spacing)
        else:
            for k in range(states.shape[0]):
                ax.plot(t, S[k] / N, color_codes['S'], alpha=0.5, lw=2, marker=marker, markevery=marker_spacing)
                ax.plot(t, I[k] / N, color_codes['I'], alpha=0.5, lw=2, marker=marker, markevery=marker_spacing)
                ax.plot(t, R[k] / N, color_codes['R'], alpha=0.5, lw=2, marker=marker, markevery=marker_spacing)
                ax.plot(t, D[k] / N, color_codes['D'], alpha=0.5, lw=2, marker=marker, markevery=marker_spacing)
                ax.plot(t, V[k] / N, color_codes['V'], alpha=0.5, lw=2, marker=marker, markevery=marker_spacing)

    ax.set_xlabel('Time step')
    ax.set_ylabel('Fraction')
    ax.set_yscale('log')

    plt.title(title)

    # ax.set_ylim(0, 1.2)
    ax.yaxis.set_tick_params(length=0)
    ax.xaxis.set_tick_params(length=0)
    ax.grid(b=True, which='major', c='w', lw=2, ls='-')

    for spine in ('top', 'right', 'bottom', 'left'):
        ax.spines[spine].set_visible(False)
    plt.show()

    if save_pth is not None:
        fig.savefig(save_pth)


def graph_single_update(num, layout, G, node_states, ax, title):
    ax.clear()

    # Draw the graph with specified node colors
    # Scheme: S - blue,  I - red, R - green, D - black,
    colors = np.zeros((len(G.nodes), 3))
    colors[node_states[num] == 0] = (0, 0, 1)
    colors[node_states[num] == 1] = (1, 0, 0)
    colors[node_states[num] == 2] = (0, 1, 0)
    colors[node_states[num] == 3] = (0, 0, 0)
    colors[node_states[num] == 4] = (1, 1, 0)

    nx.draw(G, pos=layout, node_color=colors, ax=ax)

    # Set the title
    ax.set_title(title + "\nTime step {}".format(num))


def graph_animate(config, G, node_states, save_pth):
    # Build plot
    fig, ax = plt.subplots(figsize=(10, 8))

    layout = nx.spring_layout(G)
    T = node_states.shape[0]
    title = get_plot_title(config)

    ani = animation.FuncAnimation(fig, graph_single_update, frames=T, fargs=(layout, G, node_states, ax, title))
    if save_pth is not None:
        ani.save(save_pth, writer='imagemagick')

    plt.show()


def get_plot_title(config, prefix='', suffix=''):
    N = config['population_size']
    num_runs = config['num_runs']
    si, ir, id = config['infection_rate'], config['recovery_rate'], config['death_rate']
    sv, iv, vi = config['vaccination_rate'], config['vaccination_rate'], config['reinfection_rate']

    I0, R0, D0, V0 = config['init_state']['I'], config['init_state']['R'], config['init_state']['D'], config['init_state']['V']
    S0 = N - I0 - R0 - D0 - V0

    title = prefix
    title += "N={}, si={}, ir={},id={}, sv={}, iv={}, vi={}\n".format(N, si, ir, id, sv, iv, vi)
    title += "# runs = {}, S0 = {}, I0 = {}, R0 = {}, D0 = {}, V0 = {}".format(num_runs, S0, I0, R0, D0, V0)
    title += suffix
    return title


def get_legend():
    colors = ['blue', 'red', 'green', 'black', 'orange']
    lines = [Line2D([0], [0], color=c, linewidth=3) for c in colors]
    labels = ['Susceptible', 'Infected', 'Recovered with immunity', 'Deaths', 'Vaccinated']
    color_codes = {'S': 'b', 'I': 'r', 'R': 'g', 'D': 'k', 'V': 'y'}

    return lines, labels, color_codes


def something_vs_simtime(parameter_name, parameter_values, sim_times):
    plt.plot(parameter_values, sim_times, marker='x')
    plt.xlabel(parameter_name)
    plt.ylabel("Average simulation time per run in seconds")
    plt.title("{} vs Simulation time".format(parameter_name))
    plt.grid()
    plt.show()
