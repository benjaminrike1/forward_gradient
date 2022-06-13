"""Collection of helper functions used to study the behavior of the forward gradient on 
optimization test functions. """
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import numpy as np

def plot_loss(losses, steps):
    "Plots the function value per step"
    fig, ax = plt.subplots(1,1, figsize=(15, 8))
    X = [i for i in range(steps)]
    ax.set_title("f(x) by steps")
    ax.set_xlabel('Steps')
    ax.set_ylabel('f(x)')
    sns.lineplot(x=X, y=losses, ax=ax)


def plot_contour2(loss, params, func, xlim, ylim):
    """Plots a countour plot and 3D plot of a function, its losses and the parameter history.
    Loss should be a list of losses per iteration, params a list of tuples of parameters per iteration,
    and func a callable function."""
    x0, x1 = xlim
    y0, y1 = ylim
    w0 = [params[i] for i in range(0, len(params), 2)]
    w1 = [params[i] for i in range(1, len(params), 2)]
    theta_0, theta_1 = w0, w1

    w0 = np.linspace(x0, x1, 1000)
    w1 = np.linspace(y0, y1, 1000)

    T0, T1 = np.meshgrid(w0, w1)
    f_s = func((T0, T1)).numpy()

    #Reshaping the cost values    
    Z = f_s.reshape(T0.shape)


    #Angles needed for quiver plot
    anglesx = np.array(theta_0)[1:] - np.array(theta_0)[:-1]
    anglesy = np.array(theta_1)[1:] - np.array(theta_1)[:-1]

    fig = plt.figure(figsize = (16,8))

    #Surface plot
    ax = fig.add_subplot(1, 2, 1, projection='3d')
    ax.plot_surface(T0, T1, Z, rstride = 5, cstride = 5, cmap = 'jet', alpha=0.5)
    ax.plot(theta_0,theta_1,loss, marker = '*', color = 'r', alpha = .4, label = 'Gradient descent')

    ax.set_xlabel('theta 0')
    ax.set_ylabel('theta 1')
    ax.set_zlabel('Cost function')
    ax.set_title('Gradient descent: Root at {}'.format([theta_0[-1], theta_1[-1]]))
    ax.view_init(45, 45*5)


    #Contour plot
    ax = fig.add_subplot(1, 2, 2)
    ax.contour(T0, T1, Z, 1000, cmap = 'jet')
    ax.quiver(theta_0[:-1], theta_1[:-1], anglesx, anglesy, scale_units = 'xy', angles = 'xy', scale = 1, color = 'r', alpha = .9)

    plt.show()

