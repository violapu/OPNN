import pandas as pd
import numpy as np

from matplotlib import pyplot as plt
from matplotlib import rc


def drop_row(df, iterations):
    if len(df) > (iterations + 1):
        new_df = df.drop([iterations + 1, len(df) - 1])
        return new_df
    else:
        return df


def losses_against_iterations(df, path):
    plt.style.use('default')
    plt.rcParams.update({'font.size': 16})
    rc('text', usetex=True)
    rc('font', **{'family': 'serif', 'serif': ['Computer Modern Roman']})

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6))

    ax1.plot(df['iterations'], df['int_losses'], label='Training loss', color='royalblue')
    ax1.plot(df['iterations'], df['int_validation_losses'], label='Validation loss', color='darkorange')
    ax1.set_title('Interior Losses')
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Loss')
    ax1.set_yscale('log')
    ax1.legend()

    ax2.plot(df['iterations'], df['boundary_losses'], label='Training loss', color='royalblue')
    ax2.plot(df['iterations'], df['boundary_validation_losses'], label='Validation loss', color='darkorange')
    ax2.set_title('Boundary Losses')
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Loss')
    ax2.set_yscale('log')
    ax2.legend()

    plt.tight_layout()
    plt.savefig(path)


def avg_losses_against_iterations(df, path, window=200):
    df['avg_int_losses'] = df['int_losses'].rolling(window=window).mean()
    df['avg_int_validation_losses'] = df['int_validation_losses'].rolling(window=window).mean()
    df['avg_boundary_losses'] = df['boundary_losses'].rolling(window=window).mean()
    df['avg_boundary_validation_losses'] = df['boundary_validation_losses'].rolling(window=window).mean()

    plt.style.use('default')
    plt.rcParams.update({'font.size': 16})
    rc('text', usetex=True)
    rc('font', **{'family': 'serif', 'serif': ['Computer Modern Roman']})

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6))

    ax1.plot(df['iterations'], df['avg_int_losses'], label='Interior Losses', color='royalblue')
    ax1.plot(df['iterations'], df['avg_int_validation_losses'], label='Validation loss', color='darkorange')
    ax1.set_title('Interior Losses')
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Loss')
    ax1.set_yscale('log')
    ax1.legend()

    ax2.plot(df['iterations'], df['avg_boundary_losses'], label='Boundary Losses', color='royalblue')
    ax2.plot(df['iterations'], df['avg_boundary_validation_losses'], label='Validation loss', color='darkorange')
    ax2.set_title('Boundary Losses')
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Loss')
    ax2.set_yscale('log')
    ax2.legend()

    plt.tight_layout()
    plt.savefig(path)


def all_losses(iterations, file_name, path):
    default_df = pd.read_csv(f'histories/history_DefaultAdaptive_{file_name}.csv')
    optimal_df = pd.read_csv(f'histories/history_OptimalAdaptive_{file_name}.csv')
    magnitude_df = pd.read_csv(f'histories/history_MagnitudeAdaptive_{file_name}.csv')

    new_default_df = drop_row(default_df, iterations)
    new_optimal_df = drop_row(optimal_df, iterations)
    new_magnitude_df = drop_row(magnitude_df, iterations)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6))

    ax1.plot(new_default_df['iterations'], new_default_df['int_losses'], label='Default')
    ax1.plot(new_default_df['iterations'], new_optimal_df['int_losses'], label='Optimal')
    ax1.plot(new_default_df['iterations'], new_magnitude_df['int_losses'], label='Magnitude')
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Loss')
    ax1.set_title('Interior Losses')
    ax1.legend(loc='upper right')
    ax1.set_yscale('log')

    ax2.plot(new_default_df['iterations'], new_default_df['boundary_losses'], label='Default')
    ax2.plot(new_default_df['iterations'], new_optimal_df['boundary_losses'], label='Optimal')
    ax2.plot(new_default_df['iterations'], new_magnitude_df['boundary_losses'], label='Magnitude')
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Loss')
    ax2.set_title('Boundary Losses')
    ax2.legend(loc='upper right')
    ax2.set_yscale('log')

    plt.tight_layout()
    plt.savefig(path)


def all_average_losses(iterations, file_name, path, window=500):
    default_df = pd.read_csv(f'histories/history_DefaultAdaptive_{file_name}.csv')
    optimal_df = pd.read_csv(f'histories/history_OptimalAdaptive_{file_name}.csv')
    magnitude_df = pd.read_csv(f'histories/history_MagnitudeAdaptive_{file_name}.csv')

    new_default_df = drop_row(default_df, iterations)
    new_optimal_df = drop_row(optimal_df, iterations)
    new_magnitude_df = drop_row(magnitude_df, iterations)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6))

    ax1.plot(new_default_df['iterations'], new_default_df['int_losses'].rolling(window=window).mean(),
             label='Default')
    ax1.plot(new_default_df['iterations'], new_optimal_df['int_losses'].rolling(window=window).mean(),
             label='Optimal')
    ax1.plot(new_default_df['iterations'], new_magnitude_df['int_losses'].rolling(window=window).mean(),
             label='Magnitude')
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Loss')
    ax1.set_title('Interior Losses')
    ax1.legend(loc='upper right')
    ax1.set_yscale('log')

    ax2.plot(new_default_df['iterations'], new_default_df['boundary_losses'].rolling(window=window).mean(),
             label='Default')
    ax2.plot(new_default_df['iterations'], new_optimal_df['boundary_losses'].rolling(window=window).mean(),
             label='Optimal')
    ax2.plot(new_default_df['iterations'], new_magnitude_df['boundary_losses'].rolling(window=window).mean(),
             label='Magnitude')
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Loss')
    ax2.set_title('Boundary Losses')
    ax2.legend(loc='upper right')
    ax2.set_yscale('log')

    plt.tight_layout()
    plt.savefig(path)


def surface_plot(pde, nn, domain, x_label, y_label, title, path, elevation=None, angle=None,
                 boundary_plot=False):
    plt.style.use('default')
    plt.rcParams.update({'font.size': 11.5})
    rc('text', usetex=True)
    rc('font', **{'family': 'serif', 'serif': ['Computer Modern Roman']})

    fig = plt.figure(figsize=(6, 4), tight_layout=True)
    ax = fig.gca(projection='3d')
    x, y, z = pde.get_interior_plot_data(point_count=5000, tensor=nn.y_int, x=domain)
    ax.plot_surface(x, y, z, cmap='viridis')
    if boundary_plot:
        color = next(ax._get_lines.prop_cycler)['color']
        x, y, z = pde.get_boundary_plot_data(point_count=1000, tensor=nn.boundary_condition, x=domain)
        ax.plot3D(x[0], y[0], z[0], color=color, label='Boundary Condition')
        for i in range(1, len(x)):
            if np.any(y[i] != 0):
                ax.plot3D(x[i], y[i], z[i], color=color)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_zlabel('$V$')
    ax.set_title(title, fontsize=13)
    ax.view_init(elev=elevation, azim=angle)

    plt.savefig(path)


def implied_vol_surface_plot(pde, nn, domain, x_label, y_label, title, path):
    plt.style.use('default')
    plt.rcParams.update({'font.size': 10})
    rc('text', usetex=True)
    rc('font', **{'family': 'serif', 'serif': ['Computer Modern Roman']})

    fig = plt.figure(figsize=(6, 4), tight_layout=True)
    ax = fig.gca(projection='3d')
    x, y, z = pde.get_interior_implied_vol_plot_data2d(point_count=5000, tensor=nn.y_int, x=domain)
    ax.plot_surface(x, y, z, cmap='viridis')

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_zlabel('$vol$')
    ax.set_title(title)

    plt.savefig(path)


def single_plot(pde, domain, x_label, title, path1, path2):
    plt.style.use('default')
    plt.rcParams.update({'font.size': 14})
    rc('text', usetex=True)
    rc('font', **{'family': 'serif', 'serif': ['Computer Modern Roman']})

    plt.figure()
    fig1, ax1 = plt.subplots(figsize=(6, 4), tight_layout=True)
    fig2, ax2 = plt.subplots(figsize=(6, 4), tight_layout=True)

    x, y = pde.get_predicted_plot_data(domain)
    analytical_x, analytical_y = pde.get_analytical_plot_data(domain)
    ax1.plot(x, y, label='Approximated Solution', color='royalblue')
    ax1.plot(analytical_x, analytical_y, label='Analytical Solution', linestyle='dashed', color='darkorange')
    ax1.plot(x, np.maximum(pde.strike_price([])-x, 0), color='dimgrey', label='Payoff', linestyle='dotted')
    ax1.set_title(title)
    ax1.set_xlabel(x_label)
    ax1.set_ylabel("$V$")
    ax1.grid(True)
    ax1.legend()

    ax2.plot(analytical_x, abs(analytical_y - y), color='royalblue')
    ax2.set_title('\nDifferences in Prices')
    ax2.set_xlabel(x_label)
    ax2.set_ylabel("$V$")
    ax2.grid(True)

    fig1.savefig(path1)
    fig2.savefig(path2)


def single_plot_with_generator(data, index, nn, numerical_prices, title, x_label, legend_label, path1, path2):
    fig1, ax1 = plt.subplots(figsize=(6, 4), tight_layout=True)
    fig2, ax2 = plt.subplots(figsize=(6, 4), tight_layout=True)
    plt.style.use('default')
    plt.rcParams.update({'font.size': 14})
    rc('text', usetex=True)
    rc('font', **{'family': 'serif', 'serif': ['Computer Modern Roman']})

    y = nn.predict(data)
    x = data[index].transpose()[0]
    ax1.plot(x, y, label='Approximated Solution', color='royalblue')
    ax1.plot(x, numerical_prices, label=legend_label, linestyle='dashed', color='darkorange')
    ax1.plot(x, np.maximum(x-20, 0), color='dimgrey', label='Payoff', linestyle='dotted')
    ax1.set_title(title)
    ax1.set_xlabel(x_label)
    ax1.set_ylabel('$V$')
    ax1.grid(True)
    ax1.legend()

    ax2.plot(x, abs(y.transpose()[0] - numerical_prices), color='royalblue')
    ax2.set_title('\nDifferences in Prices')
    ax2.set_xlabel(x_label)
    ax2.grid(True)
    ax2.set_ylabel('$V$')

    fig1.savefig(path1)
    fig2.savefig(path2)
