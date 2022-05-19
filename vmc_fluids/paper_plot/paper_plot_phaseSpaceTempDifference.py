import numpy as np
import h5py
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from cycler import cycler
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

folder = "./"
save_figs_not_in_paper = False

# phase space plot
wdir_Wiener = "./data_phaseSpace/Wiener/Nsamples10000_T10.0/"
wdir_INN = "./data_phaseSpace/INN/NsamplesTDVP1000_NsamplesObs10000_T10/"
wdir_Wiener_diffTemp = "./data_phaseSpace/Wiener/Nsamples10000_Tdifferent/"
wdir_INN_diffTemp = "./data_phaseSpace/INN/NsamplesTDVP10000_NsamplesObs10000_Tdifferent/"


data_Wiener = h5py.File(wdir_Wiener + 'infos.hdf5', 'r')
data_INN = h5py.File(wdir_INN + 'infos.hdf5', 'r')

data_Wiener_diffTemp = h5py.File(wdir_Wiener_diffTemp + 'infos.hdf5', 'r')
data_INN_diffTemp = h5py.File(wdir_INN_diffTemp + 'infos.hdf5', 'r')

# data_Wiener = h5py.File(wdir_Wiener_diffTemp + 'infos.hdf5', 'r')
# data_INN = h5py.File(wdir_INN_diffTemp + 'infos.hdf5', 'r')


# Moments of the distribution
fig_all, axes = plt.subplots(figsize=(12, 3), ncols=4)
fig, ax = plt.subplots(figsize=(4, 4))

for idx, (data, name, ls) in enumerate(zip([data_Wiener_diffTemp, data_INN_diffTemp], ["Wiener", "INN"], ["--", "-"])):
    means = np.array(data["x1"])
    covars = np.array(data["covar"])
    times = np.array(data["times"])

    if idx == 0:
        colors = ['blue', 'green'] * 3
    else:
        colors = ['red', 'orange'] * 3
    for mean_idx, (mean, covar, alpha, color) in enumerate(zip(means.T, covars.T, [0.3, 0.3, 0.4, 0.4, 0.5, 0.5], colors)):
        if idx == 0:
            ax.plot(times, mean, linestyle=ls, alpha=alpha, color=color, zorder=12)
            axes[0].plot(times, mean, linestyle=ls, alpha=alpha, color=color, zorder=12)
            # ax.plot(times, mean, linestyle=ls, alpha=alpha, color=color, zorder=12)
            axes[1].plot(times, covar[mean_idx], linestyle=ls, alpha=alpha, color=color, zorder=12)
        else:
            ax.plot(times, mean, linestyle=ls, alpha=alpha, color=color, zorder=11)
            axes[0].plot(times, mean, linestyle=ls, alpha=alpha, color=color, zorder=11)
            axes[1].plot(times, covar[mean_idx], linestyle=ls, alpha=alpha, color=color, zorder=11)

# ax.hlines(0, times[0], times[-1], linestyle='--', color='black', zorder=10)
# axes[0].hlines(0, times[0], times[-1], linestyle='--', color='black', zorder=10)

ax.grid()
ax.set_xlabel(r'$\omega t$')
ax.set_ylabel(r'$\langle O \rangle$')
axes[0].grid()
axes[0].set_xlabel(r'$\omega t$')
axes[0].set_title(r'$\langle O\rangle$')
axes[1].grid()
axes[1].set_xlabel(r'$\omega t$')
axes[1].set_title(r'$\langle O^2\rangle - \langle O \rangle^2 $')
# axes[0].set_title(r'$\langle O \rangle$')

line1 = Line2D([0, 1], [0, 1], linestyle='--', color='blue')
line2 = Line2D([0, 1], [0, 1], linestyle='-', color='red')
line3 = Line2D([0, 1], [0, 1], linestyle='--', color='green')
line4 = Line2D([0, 1], [0, 1], linestyle='-', color='orange')
line5 = Line2D([0, 1], [0, 1], linestyle='--', color='black')

ax.legend([line1, line2, line3, line4, line5], [r"$\langle x \rangle$ - Wiener", r"$\langle x \rangle$ - INN", r"$\langle p \rangle$ - Wiener", r"$\langle p \rangle$ - INN", "Steady State"])
# axes[0].legend([line1, line2, line3, line4, line5], [r"$\langle x \rangle$ - Wiener", r"$\langle x \rangle$ - INN", r"$\langle p \rangle$ - Wiener", r"$\langle p \rangle$ - INN", "Steady State"])
axes[1].legend([line1, line2, line3, line4], [r"$x$ - Wiener", r"$x$ - INN", r"$p$ - Wiener", r"$p$ - INN"], ncol=1, loc='upper right')

fig.tight_layout()
if save_figs_not_in_paper:
    fig.savefig(folder + 'phaseSpace_means.pdf')
    fig.savefig(folder + 'phaseSpace_means.png')

entropy = True
if entropy:
    # Entropy
    fig, ax = plt.subplots(figsize=(4, 4))
    entropy = np.array(data_INN["entropy"])
    times = np.array(data_INN["times"])
    std = 0.5 * np.log(2 * np.pi * np.exp(1) * 10) * 6
    ax.plot(times, entropy)
    ax.hlines(std, times[0], times[-1], color='black', linestyle='--')
    ax.grid()
    ax.set_xlabel(r'$\omega t$')
    ax.set_ylabel(r'Entropy')

    axes[3].plot(times, entropy, color='blue', alpha=.8)
    axes[3].hlines(std, times[0], times[-1], color='black', linestyle='--')
    axes[3].grid()
    axes[3].set_xlabel(r'$\omega t$')
    axes[3].set_title(r'Entropy')
    # axes[3].set_ylabel(r'Entropy')

    line1 = Line2D([0, 1], [0, 1], linestyle='-', color='blue')
    line2 = Line2D([0, 1], [0, 1], linestyle='--', color='black')
    axes[3].legend([line1, line2, line3], ["INN", "Steady State"])

    plt.tight_layout()
    if save_figs_not_in_paper:
        plt.savefig(folder + 'phaseSpace_entropy.pdf')
        plt.savefig(folder + 'phaseSpace_entropy.png')
else:
    pass

# Integrals
fig, ax = plt.subplots(figsize=(4, 4))
ax.plot(np.array(data_INN["times"]), np.array(data_INN["integral_1sigma"]), label=r"$\sigma - INN$", linestyle='-', color='blue')
ax.plot(np.array(data_INN["times"]), np.array(data_INN["integral_0.5sigma"]), label=r"$0.5\sigma - INN$", linestyle='-', color='blue')
ax.plot(np.array(data_INN["times"]), np.array(data_INN["integral_0.1sigma"]), label=r"$0.1\sigma - INN$", linestyle='-', color='blue')
axes[2].plot(np.array(data_INN["times"]), np.array(data_INN["integral_1sigma"]), label=r"$\sigma - INN$", linestyle='-', color='blue', alpha=.8)
axes[2].plot(np.array(data_INN["times"]), np.array(data_INN["integral_0.5sigma"]), label=r"$0.5\sigma - INN$", linestyle='-', color='blue', alpha=.8)
axes[2].plot(np.array(data_INN["times"]), np.array(data_INN["integral_0.1sigma"]), label=r"$0.1\sigma - INN$", linestyle='-', color='blue', alpha=.8)

nth = 20
ax.scatter(np.array(data_Wiener["times"])[::nth], np.array(data_Wiener["integral_1sigma"])[::nth], label=r"$\sigma - Wiener$", linestyle='--', color='red', marker='x', alpha=0.2)
ax.scatter(np.array(data_Wiener["times"])[::nth], np.array(data_Wiener["integral_0.5sigma"])[::nth], label=r"$0.5\sigma - Wiener$", linestyle='--', color='red', marker='x', alpha=0.2)
ax.scatter(np.array(data_Wiener["times"])[::nth], np.array(data_Wiener["integral_0.1sigma"])[::nth], label=r"$0.1\sigma - Wiener$", linestyle='--', color='red', marker='x', alpha=0.2)
# axes[2].scatter(np.array(data_Wiener["times"])[::nth], np.array(data_Wiener["integral_1sigma"])[::nth], label=r"$\sigma - Wiener$", linestyle='None', color='red', marker='x', alpha=0.2)
# axes[2].scatter(np.array(data_Wiener["times"])[::nth], np.array(data_Wiener["integral_0.5sigma"])[::nth], label=r"$0.5\sigma - Wiener$", linestyle='None', color='red', marker='x', alpha=0.2)
# axes[2].scatter(np.array(data_Wiener["times"])[::nth], np.array(data_Wiener["integral_0.1sigma"])[::nth], label=r"$0.1\sigma - Wiener$", linestyle='None', color='red', marker='x', alpha=0.2)

int_1sigma = 0.0143877
int_05sigma = 0.000296478
int_01sigma = 2.07554 * 1e-8
# Obtained with Wolfram:
# integrate 1 / sqrt(2 * pi * s^2)**d * 2 * pi**(d/2) / Gamma(d/2) * exp(-0.5 * x^2 / s^2) * x^(d-1) with x=0..k where s=1, k=1, d=2


ax.hlines(int_1sigma, times[0], times[-1], color='black', linestyle='--', zorder=0)
ax.hlines(int_05sigma, times[0], times[-1], color='black', linestyle='--', zorder=0)
ax.hlines(int_01sigma, times[0], times[-1], color='black', linestyle='--', zorder=0)
ax.grid()
ax.set_xlabel(r'$\omega t$')
ax.set_ylabel(r'Integral')
ax.set_yscale('log')
axes[2].hlines(int_1sigma, times[0], times[-1], color='black', linestyle='--', zorder=10)
axes[2].hlines(int_05sigma, times[0], times[-1], color='black', linestyle='--', zorder=10)
axes[2].hlines(int_01sigma, times[0], times[-1], color='black', linestyle='--', zorder=10)
axes[2].grid()
axes[2].set_xlabel(r'$\omega t$')
# axes[2].set_ylabel(r'Integral')
axes[2].set_title(r'Integral')
axes[2].set_yscale('log')

line1 = plt.scatter([0], [0], marker='x', color='red', alpha=0.2)
line2 = Line2D([0, 1], [0, 1], linestyle='-', color='blue')
line3 = Line2D([0, 1], [0, 1], linestyle='--', color='black')
ax.legend([line1, line2, line3], ["Wiener", "INN", "Steady State"])
# axes[2].legend([line1, line2, line3], ["Wiener", "INN", "Steady State"], loc='lower right')
# axes[2].legend([line1, line2, line3], ["Wiener", "INN", "Steady State"], loc=(0.42, 0.08))

ax.text(2.4, 4e-2, r'$r=\sigma$')
ax.text(3, 1.6e-3, r'$r=0.5\sigma$')
ax.text(4.3, 1.5e-7, r'$r=0.1\sigma$')
axes[2].text(1.6, 8e-2, r'$r=\sigma$')
axes[2].text(1.6, 1.6e-3, r'$r=0.5\sigma$')
axes[2].text(1.6, 1.5e-7, r'$r=0.1\sigma$')

for (ax_i, letter, pos) in zip(axes, ["a", "b", "c", "d"], [(0, 0.95), (0, 0.95), (0, 0.95), (0, 0.95)]):
    ax_i.text(pos[0], pos[1] + 0.09, f"({letter})", transform=ax_i.transAxes, fontdict={"fontsize": 13, "weight": 'bold'})

fig.tight_layout()
if save_figs_not_in_paper:
    fig.savefig(folder + 'phaseSpace_integrals.pdf')
    fig.savefig(folder + 'phaseSpace_integrals.png')

fig_all.tight_layout()
fig_all.savefig(folder + 'phaseSpace.pdf', bbox_inches='tight')
fig_all.savefig(folder + 'phaseSpace.png')


plt.show()
