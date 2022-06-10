import matplotlib.pyplot as plt
import seaborn as sns

def plot_loss(losses, steps):
    fig, ax = plt.subplots(1,1, figsize=(15, 8))
    X = [i for i in range(steps)]
    ax.set_title("f(x) by steps")
    ax.set_xlabel('Steps')
    ax.set_ylabel('f(x)')
    print(losses)
    sns.lineplot(x=X, y=losses, ax=ax)

def plot_countour(loss, params, func, xlim, ylim):
  x_0, x_1 = xlim
  y_0, y_1 = ylim

  w0 = [params[i] for i in range(0, len(params), 2)]
  w1 = [params[i] for i in range(1, len(params), 2)]


  old_w = np.asarray([(e_0, e_1) for e_0, e_1 in zip(w0, w1)])

  w0 = torch.linspace(x_0, x_1, 1000)
  w1 = torch.linspace(y_0, y_1, 1000)

  W0, W1 = torch.meshgrid(w0, w1)
  f_s = func((W0, W1))

  plt.figure(figsize=(8, 8), dpi=80)
  plt.contourf(w0, w1, f_s,alpha=.7)
  plt.axhline(0, color='black', alpha=.5, dashes=[2, 4],linewidth=1)
  plt.axvline(0, color='black', alpha=0.5, dashes=[2, 4],linewidth=1)
  for i in range(len(old_w) - 1):
      plt.annotate('', xy=old_w[i + 1, :], xytext=old_w[i, :],
                  arrowprops={'arrowstyle': '->', 'color': 'r', 'lw': 1},
                  va='center', ha='center')
  
  CS = plt.contour(w0, w1, f_s, linewidths=1,colors='black')
  plt.clabel(CS, inline=1, fontsize=8)
  plt.title("Contour Plot of Gradient Descent")
  plt.xlim(x_0, x_1)
  plt.ylim(y_0, y_1)
  plt.xlabel("w0")
  plt.ylabel("w1")
  plt.show()


def plot_contour2(loss, params, func, xlim, ylim):
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

