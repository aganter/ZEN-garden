

import pickle
import os
import numpy as np
import matplotlib.pyplot as plt
from skopt.learning import GaussianProcessRegressor
from skopt.plots import plot_objective
from skopt.plots import plot_convergence
import numpy as np
import matplotlib.pyplot as plt
from skopt.acquisition import gaussian_acquisition_1D
import numpy as np
np.random.seed(237)
import matplotlib.pyplot as plt
from skopt.plots import plot_gaussian_process
from itertools import product
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import numpy as np
import matplotlib.pyplot as plt
from skopt.learning import GaussianProcessRegressor
from skopt.space import Space



opt1 = r"C:\Users\bekaj\Documents\ETH\Master\Masterarbeit\euler_calcs\check_with_info\optimizer_objects\70\0\opt_DK-DE.biomethane_transport.0.pkl"
opt2 = r"C:\Users\bekaj\Documents\ETH\Master\Masterarbeit\euler_calcs\finetuning_picklesave_aggts_60\optimizer_objects\0\opt_DE-PL.biomethane_transport.1.pkl"

with open(opt1, 'rb') as f:
    optimizer = pickle.load(f)

    # Generating random data for plotting
    x = [opt[0] for opt in optimizer.Xi]
    y = [opt[1] for opt in optimizer.Xi]
    z = [opt for opt in optimizer.yi]

    # Creating a 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plotting the data
    ax.scatter(x, y, z)

    # Adding labels
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')

    # Display the plot
    plt.show()


    # Define your search space
    space = optimizer.space

    for i in range(len(optimizer.models)):

        if i%5 == 0:

            X = np.array([res for res in optimizer.Xi])[:20+i]
            Y = np.array([res for res in optimizer.yi])[:20+i]

            # Assuming 'optimizer' is your Optimizer instance
            gp = optimizer.models[i]

            # Generate a grid of points (100x100 in this example)
            x = np.linspace(min(X[:, 0]), max(X[:, 0]), 100)
            y = np.linspace(min(X[:, 1]), max(X[:, 1]), 100)
            xx, yy = np.meshgrid(x, y)
            grid = np.vstack((xx.ravel(), yy.ravel())).T

            # Transform grid points to the warped space if necessary
            grid_transformed = space.transform(grid.tolist())

            # Predict the mean and standard deviation of the objective function over the grid
            mean_prediction, std_prediction = gp.predict(grid_transformed, return_std=True)

            # Reshape for plotting
            mean_prediction = mean_prediction.reshape(xx.shape)
            std_prediction = std_prediction.reshape(xx.shape)

            # Plot the predicted mean as a contour plot
            plt.figure(figsize=(14, 7))

            plt.subplot(1, 2, 1)
            contour = plt.contourf(xx, yy, mean_prediction, levels=50, cmap='viridis')
            plt.colorbar(contour)
            plt.scatter([x[0] for x in optimizer.Xi][:20+i], [x[1] for x in optimizer.Xi][:20+i], c='r', s=50, label='Observations')
            plt.title('GP Mean Prediction')
            plt.xlabel('x')
            plt.ylabel('y')

            # Plot the predicted standard deviation as a contour plot
            plt.subplot(1, 2, 2)
            contour = plt.contourf(xx, yy, std_prediction, levels=50, cmap='viridis')
            plt.colorbar(contour)
            plt.scatter([x[0] for x in optimizer.Xi][:20+i], [x[1] for x in optimizer.Xi][:20+i], c='r', s=50, label='Observations')
            plt.title('GP Standard Deviation')
            plt.xlabel('x')
            plt.ylabel('y')

            plt.tight_layout()
            plt.show()




    # Assuming 'optimizer' is your Optimizer instance
    gp = optimizer.models[-1]

    # Generate a grid of points (100x100 in this example)
    x = np.linspace(-5, 10, 100)
    y = np.linspace(-5, 10, 100)
    xx, yy = np.meshgrid(x, y)
    grid = np.vstack((xx.ravel(), yy.ravel())).T

    # Transform grid points to the warped space if necessary
    grid_transformed = space.transform(grid.tolist())

    # Assuming 'optimizer' is your Optimizer object from skopt
    X = np.array([res for res in optimizer.Xi])
    Y = np.array([res for res in optimizer.yi])

    # Create a meshgrid for the input space
    x = np.linspace(min(X[:, 0]), max(X[:, 0]), 100)
    y = np.linspace(min(X[:, 1]), max(X[:, 1]), 100)
    x, y = np.meshgrid(x, y)
    X_grid = np.vstack([x.ravel(), y.ravel()]).T

    # Use the GP model to predict the objective function's value over the grid
    Y_pred, sigma = optimizer.models[-1].predict(X_grid, return_std=True)
    Y_pred = Y_pred.reshape(x.shape)
    sigma = sigma.reshape(x.shape)

    # Plot the GP model approximation in 3D
    fig = plt.figure(figsize=(14, 6))

    # Plotting the approximation of the GP model
    ax = fig.add_subplot(1, 2, 1, projection='3d')
    ax.plot_surface(x, y, Y_pred, cmap='viridis', alpha=0.6)
    ax.plot_wireframe(x, y, Y_pred + sigma, color='green', alpha=0.5)
    ax.plot_wireframe(x, y, Y_pred - sigma, color='red', alpha=0.5)
    ax.set_title('GP model approximation with uncertainty')
    ax.set_xlabel('X1')
    ax.set_ylabel('X2')
    ax.set_zlabel('Objective')

    # Plotting the acquisition function
    acq_func_values = optimizer.acq_func(X_grid)
    acq_func_values = acq_func_values.reshape(x.shape)
    ax2 = fig.add_subplot(1, 2, 2)
    contour = ax2.contourf(x, y, acq_func_values, levels=50, cmap='viridis')
    fig.colorbar(contour, ax=ax2, orientation='vertical')
    ax2.set_title('Acquisition function')
    ax2.set_xlabel('X1')
    ax2.set_ylabel('X2')

    plt.tight_layout()
    plt.show()

    q = 9

    #
    #
    # X = np.array(optimizer.Xi)
    # y = np.array(optimizer.yi)
    #
    # x1 = np.linspace(X[:,0].min(), X[:,0].max())
    # x2 = np.linspace(X[:,1].min(), X[:,1].max())
    # x = (np.array([x1, x2])).T
    #
    # gp = optimizer.models[-1]
    #
    # # Predict mean and std deviation for each point in X
    # x1x2 = np.array(list(product(x1, x2)))
    # y_pred, MSE = gp.predict(x1x2, return_std=True)
    #
    # X0p, X1p = x1x2[:, 0].reshape(50, 50), x1x2[:, 1].reshape(50, 50)
    # Zp = np.reshape(y_pred, (50, 50))
    #
    # # alternative way to generate equivalent X0p, X1p, Zp
    # # X0p, X1p = np.meshgrid(x1, x2)
    # # Zp = [gp.predict([(X0p[i, j], X1p[i, j]) for i in range(X0p.shape[0])]) for j in range(X0p.shape[1])]
    # # Zp = np.array(Zp).T
    #
    # fig = plt.figure(figsize=(10, 8))
    # ax = fig.add_subplot(111)
    # ax.pcolormesh(X0p, X1p, Zp)
    #
    # plt.show()
    #
    # ax = fig.add_subplot(111, projection='3d')
    # surf = ax.plot_surface(X0p, X1p, Zp, rstride=1, cstride=1, cmap='jet', linewidth=0, antialiased=False)
    #
    #
    #
    # # Assuming `optimizer` is your Optimizer object
    # x = np.linspace(optimizer.space.bounds[0][0], optimizer.space.bounds[0][1], 100)
    # y = np.linspace(optimizer.space.bounds[1][0], optimizer.space.bounds[1][1], 100)
    # X, Y = np.meshgrid(x, y)
    # Z = np.array(
    #     [[optimizer.acquisition_function(np.array([[xx, yy]])) for xx, yy in zip(x_row, y_row)] for x_row, y_row in
    #      zip(X, Y)])
    #
    # plt.contourf(X, Y, Z)
    # plt.colorbar()
    # plt.xlabel('x')
    # plt.ylabel('y')
    # plt.title('Acquisition Function')
    # plt.show()
    #
    #
    #
    #
    # o = 0
    # # Assuming `optimizer` is your Optimizer object
    # # x = np.linspace(optimizer.space.bounds[0][0], optimizer.space.bounds[0][1], 100)
    # # y = np.linspace(optimizer.space.bounds[1][0], optimizer.space.bounds[1][1], 100)
    # # X, Y = np.meshgrid(x, y)
    # # Z = np.array(
    # #     [[optimizer.acquisition_function(np.array([[xx, yy]])) for xx, yy in zip(x_row, y_row)] for x_row, y_row in
    # #      zip(X, Y)])
    # #
    # # plt.contourf(X, Y, Z)
    # # plt.colorbar()
    # # plt.xlabel('x')
    # # plt.ylabel('y')
    # # plt.title('Acquisition Function')
    # # plt.show()
    #
    # # Generate points for evaluation
    # X = np.linspace(optimizer.space.bounds[0][0], optimizer.space.bounds[0][1], 100)
    # Y = np.linspace(optimizer.space.bounds[1][0], optimizer.space.bounds[1][1], 100)
    # X, Y = np.meshgrid(X, Y)
    # Z_mean, Z_std = optimizer.models[-1].predict(np.stack([X.ravel(), Y.ravel()], axis=1), return_std=True)
    # Z_mean = Z_mean.reshape(X.shape)
    # Z_std = Z_std.reshape(X.shape)
    #
    # # Plot mean prediction
    # plt.figure(figsize=(14, 6))
    # plt.subplot(1, 2, 1)
    # contour = plt.contourf(X, Y, Z_mean, levels=50)
    # plt.colorbar(contour)
    # plt.title('GP mean prediction')
    # plt.xlabel('x')
    # plt.ylabel('y')
    #
    # # Plot std deviation
    # plt.subplot(1, 2, 2)
    # contour = plt.contourf(X, Y, Z_std, levels=50)
    # plt.colorbar(contour)
    # plt.title('GP uncertainty (std. dev.)')
    # plt.xlabel('x')
    # plt.ylabel('y')
    #
    # plt.tight_layout()
    # plt.show()
    #
    # # # Assuming a 1D problem for simplicity
    # # # X = np.linspace(optimizer.space.bounds[0][0], optimizer.space.bounds[0][1], 15).reshape(-1, 1)
    # #
    # # x1_range = np.linspace(optimizer.space.bounds[0][0], optimizer.space.bounds[0][1], 100)  # Range for the first dimension
    # # x2_range = np.linspace(optimizer.space.bounds[1][0], optimizer.space.bounds[1][1], 100)  # Range for the second dimension
    # #
    # # # Create a meshgrid for the 2D space
    # # X1, X2 = np.meshgrid(x1_range, x2_range)
    # #
    # # # Reshape the grid points into a (n_samples, 2) array for predictions
    # # X = np.vstack([X1.ravel(), X2.ravel()]).T
    # # gp = optimizer.models[-1]  # Get the latest GP model
    # #
    # # # Predict mean and std deviation for each point in X
    # # y_mean, y_std = gp.predict(X, return_std=True)
    # #
    # # plt.figure(figsize=(12, 6))
    # # plt.plot(X, y_mean, 'r-', label='GP mean')
    # # plt.fill_between(X[:, 0], y_mean - y_std, y_mean + y_std, alpha=0.2, label='Confidence interval (±2σ)')
    # # # plt.scatter(optimizer.Xi, optimizer.yi, c='b', s=50, zorder=10, label='Observations')
    # # plt.title("Gaussian Process Approximation and Uncertainty")
    # # plt.legend()
    # # plt.show()
    #
    # y = 0
    #
    # # Create a mesh grid for visualization
    # x = np.linspace(-2, 2, 400)
    # y = np.linspace(-2, 2, 400)
    # X, Y = np.meshgrid(x, y)
    # Z = np.array([optimizer([xx, yy]) for xx, yy in zip(np.ravel(X), np.ravel(Y))])
    # Z = Z.reshape(X.shape)
    #
    # # Plot the objective function
    # plt.figure(figsize=(10, 8))
    # contour = plt.contourf(X, Y, Z, levels=50, cmap="viridis")
    # plt.colorbar(contour)
    # plt.xlabel('x')
    # plt.ylabel('y')
    # plt.title('Objective Function')
    #
    # # Optionally, plot the observations
    # plt.scatter(result.x_iters[:, 0], result.x_iters[:, 1], c='red')
    #
    # plt.show()
    #
    #
    # plot_objective(result)


