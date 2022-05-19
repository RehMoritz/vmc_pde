import numpy as np
import h5py
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from cycler import cycler
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


folder = "./"
save_figs_not_in_paper = False

# Diffusion Plot
wdir_grid_studentT = "./data_diffusion/StudentT_gridbased/"
wdir_INN_studentT = "./data_diffusion/dim8_StudentT_nu2_NsamplesTDVP10000_NsamplesObs10000/"
wdir_INN_Gauss = "./data_diffusion/dim8_Gauss_NsamplesTDVP10000_NsamplesObs10000/"

data_grid_studentT = h5py.File(wdir_grid_studentT + 'infos_step4e-3_dt1e-6_centergrid_slimmed.hdf5', 'r')
data_INN_studentT = h5py.File(wdir_INN_studentT + 'infos.hdf5', 'r')
data_INN_Gauss = h5py.File(wdir_INN_Gauss + 'infos.hdf5', 'r')

fig, ax = plt.subplots(figsize=(4, 4))
fig_all, axes = plt.subplots(figsize=(8, 4), ncols=2)
fig_inset, ax_inset = plt.subplots(figsize=(5, 3))
ax.plot(data_INN_studentT["times"], data_INN_studentT["entropy"], color='blue', alpha=0.8, label=r'Student-$t$ ($\nu_0=2$) - INN')
ax.plot(data_INN_Gauss["times"], data_INN_Gauss["entropy"], color="red", alpha=0.8, label='Gauss - INN')
ax.plot(data_INN_Gauss["times"], 0.5 * np.array(data_INN_Gauss["x1"]).shape[-1] * np.log(2 * np.pi * np.exp(1) * (1 + 2 * np.array(data_INN_Gauss["times"]))), color='black', linestyle='--', alpha=0.8, label='Gauss - Analytical')
ax_inset.plot(data_INN_studentT["times"], data_INN_studentT["entropy"], color='blue', alpha=0.8, label=r'Student-$t$ ($\nu_0=2$) - INN')
ax_inset.plot(data_grid_studentT["t"], data_grid_studentT["entropy"], color='black', linestyle='--', alpha=0.8, label=r'Student-$t$ ($\nu_0=2$) - Grid')
ax_inset.plot(data_INN_Gauss["times"], data_INN_Gauss["entropy"], color="red", alpha=0.8, label='Gauss - INN')
ax_inset.plot(data_INN_Gauss["times"], 0.5 * np.array(data_INN_Gauss["x1"]).shape[-1] * np.log(2 * np.pi * np.exp(1) * (1 + 2 * np.array(data_INN_Gauss["times"]))), color='black', linestyle='--', alpha=0.8, label='Gauss - Analytical')
axes[0].plot(data_grid_studentT["t"], data_grid_studentT["entropy"], color='black', alpha=0.8, label=r'Student-$t$ ($\nu_0=2$) - Grid')
axes[0].plot(data_INN_studentT["times"], data_INN_studentT["entropy"], color='blue', alpha=0.8, label=r'Student-$t$ ($\nu_0=2$) - INN')
axes[0].plot(data_INN_Gauss["times"], data_INN_Gauss["entropy"], color="red", alpha=0.8, label='Gauss - INN')
axes[0].plot(data_INN_Gauss["times"], 0.5 * np.array(data_INN_Gauss["x1"]).shape[-1] * np.log(2 * np.pi * np.exp(1) * (1 + 2 * np.array(data_INN_Gauss["times"]))), color='black', linestyle='--', alpha=0.8, label='Gauss - Analytical')

line1 = Line2D([0, 1], [0, 1], linestyle='-', color='blue')
line2 = Line2D([0, 1], [0, 1], linestyle='--', color='black')
ax.legend()
ax.grid()
ax.set_ylabel(r'Entropy')
ax.set_xlabel(r'$Dt$')
ax_inset.legend(loc='upper left')
ax_inset.grid()
ax_inset.set_ylabel(r'Entropy')
ax_inset.set_xlabel(r'$Dt$')
axes[0].legend()
axes[0].grid()
axes[0].set_ylabel(r'Entropy')
axes[0].set_xlabel(r'$Dt$')

fig.tight_layout()
if save_figs_not_in_paper:
    fig.savefig(folder + 'diffusion_entropy.pdf')
    fig.savefig(folder + 'diffusion_entropy.png')

fig, ax = plt.subplots(figsize=(4, 4))
ax.plot(data_INN_studentT["times"], np.exp(data_INN_studentT["dist_params"]) + 1, color='blue', alpha=0.8, label=r'Student-$t$ ($\nu_0=2$) - INN')
ax.set_ylabel(r'$\nu$')
ax.set_xlabel(r'$Dt$')
ax.grid()
ax.legend()
axes[1].plot(data_INN_studentT["times"], np.exp(data_INN_studentT["dist_params"]) + 1, color='blue', alpha=0.8, label=r'Student-$t$ ($\nu_0=2$) - INN')
axes[1].set_ylabel(r'$\nu$')
axes[1].set_xlabel(r'$Dt$')
axes[1].grid()
axes[1].legend()
axins = ax_inset.inset_axes([0.57, 0.15, 0.39, 0.5])
axins.plot(data_INN_studentT["times"], np.exp(data_INN_studentT["dist_params"]) + 1, color='blue', alpha=0.8, label=r'Student-$t$ ($\nu_0=2$) - INN')
axins.set_ylabel(r'$\nu$')
# axins.set_xlabel(r'$\omega t$')
axins.grid()
# axins.legend()

fig.tight_layout()
fig_all.tight_layout()
fig_inset.tight_layout()
if save_figs_not_in_paper:
    fig.savefig(folder + 'diffusion_distparam.pdf')
    fig.savefig(folder + 'diffusion_distparam.png')
    fig_all.savefig(folder + 'diffusion.pdf')
    fig_all.savefig(folder + 'diffusion.png')
fig_inset.savefig(folder + 'diffusion_inset.pdf')
fig_inset.savefig(folder + 'diffusion_inset.png')


plt.show()
