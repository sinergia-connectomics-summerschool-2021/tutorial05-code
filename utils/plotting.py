import os
import numpy as np
import matplotlib.pyplot as plt
import scipy
import nilearn
from nilearn import plotting
import nibabel as nb
from sklearn.utils import Bunch
import pandas as pd
import pygsp
import tqdm


def plot_shaded(x, y, axis, color, label, ax, ls='-'):
    ax.plot(x, np.median(y, axis=axis), color=color, lw=2, label=label, ls=ls)
    ax.fill_between(x, *np.percentile(y, (5, 95), axis=axis), color=color,
                    alpha=.4)
    return ax


def plot_shaded_norm(x, y, axis, color, label, ax, ls='-'):
    ax.plot(x, np.mean(y, axis=axis), color=color, lw=2, label=label, ls=ls)
    ax.fill_between(x, *[[1], [-1]]*np.std(y, axis=axis) +
                    np.mean(y, axis=axis), color=color, alpha=.4)
    return ax


def my_box_plot(data, ax, colors, labels):
    bplot = ax.boxplot(data, notch=False, vert=True, patch_artist=True,
                       labels=labels, showfliers=False)

    for i, color in enumerate(colors):
        plt.setp(bplot['boxes'][i],color=color, facecolor=color, alpha=.5,
                 lw=2)
        plt.setp(bplot['whiskers'][i * 2:i * 2 + 2], color=color,
                 alpha=1, lw=2)
        plt.setp(bplot['caps'][i * 2:i * 2 + 2], color=color, alpha=1, lw=2)
        plt.setp(bplot['medians'][i], color=color, alpha=1, lw=2)
        ymin, ymax = ax.get_ylim()

        x = np.random.normal(i + 1, 0.1, size=np.size(data[i]))
        if labels[i] == 'NH-sur.':
            alpha = .005
        else:
            alpha = .3

        ax.scatter(x, data[i], 40, np.tile(np.array(color).reshape(-1, 1),
                   len(x))[:3].T, alpha=alpha, edgecolors='gray')
        ax.set_ylim([ymin, ymax])
    return ax


def plot_rois(roi_values, scale, center_at_zero=False, label='brain',
                  cmap='coolwarm', vmin=None, vmax=None, fmt='png'):

    annots = [os.path.join('data','label','rh.lausanne2008.scale{}.annot'.format(scale)),
              os.path.join('data','label','lh.lausanne2008.scale{}.annot'.format(scale))]

    annot_right = nb.freesurfer.read_annot(annots[0])
    annot_left = nb.freesurfer.read_annot(annots[1])

    labels_right = [elem.decode('utf-8') for elem in annot_right[2]]
    labels_left = [elem.decode('utf-8') for elem in annot_left[2]]

    desikan_atlas = Bunch(map_left=annot_left[0],
                          map_right=annot_right[0])

    parcellation_right = desikan_atlas['map_right']
    roi_vect_right = np.zeros_like(parcellation_right, dtype=float) * np.nan

    parcellation_left = desikan_atlas['map_left']
    roi_vect_left = np.zeros_like(parcellation_left, dtype=float) * np.nan

    roifname = os.path.join('data','label','roi_info.xlsx')
    roidata = pd.read_excel(roifname, sheet_name='SCALE {}'.format(scale))
    cort = np.where(roidata['Structure'] == 'cort')[0]

    right_rois = ([roidata['Label Lausanne2008'][i] for i in
                   range(len(roidata)) if ((roidata['Hemisphere'][i] == 'rh') &
                   (roidata['Structure'][i] == 'cort'))])
    left_rois = ([roidata['Label Lausanne2008'][i] for i in
                  range(len(roidata)) if ((roidata['Hemisphere'][i] == 'lh') &
                  (roidata['Structure'][i] == 'cort'))])

    for i in range(len(right_rois)):
        label_id = labels_right.index(right_rois[i])
        ids_roi = np.where(parcellation_right == label_id)[0]
        roi_vect_right[ids_roi] = roi_values[i]

    for i in range(len(left_rois)):
        label_id = labels_left.index(left_rois[i])
        ids_roi = np.where(parcellation_left == label_id)[0]
        roi_vect_left[ids_roi] = roi_values[len(right_rois) + i]

    fsaverage = nilearn.datasets.fetch_surf_fsaverage(mesh='fsaverage')

    if vmin is None:
        vmin = min([0, min(roi_values)])
    if vmax is None:
        vmax = max(roi_values)

    if center_at_zero:
        max_val = max([abs(vmin), vmax])
        vmax = max_val
        vmin = -max_val
    
    plt.ioff()
    fig, axs = plt.subplots(1, 3, figsize=(9, 2),
                                subplot_kw={'projection': '3d'})

    plotting.plot_surf_roi(fsaverage['pial_right'], roi_map=roi_vect_right,
                           hemi='right', view='lateral',
                           bg_map=fsaverage['sulc_right'], bg_on_data=True,
                           darkness=.5, cmap=cmap,
                           vmin=vmin, vmax=vmax,
                           figure=fig, axes=axs[0])

    plotting.plot_surf_roi(fsaverage['pial_right'], roi_map=roi_vect_right,
                           hemi='right', view='dorsal',
                           bg_map=fsaverage['sulc_right'], bg_on_data=True,
                           darkness=.5, cmap=cmap,
                           vmin=vmin, vmax=vmax,
                           figure=fig, axes=axs[1])

    plotting.plot_surf_roi(fsaverage['pial_left'], roi_map=roi_vect_left,
                           hemi='left', view='dorsal',
                           bg_map=fsaverage['sulc_left'], bg_on_data=True,
                           darkness=.5, cmap=cmap,
                           vmin=vmin, vmax=vmax,
                           figure=fig, axes=axs[1])

    plotting.plot_surf_roi(fsaverage['pial_left'], roi_map=roi_vect_left,
                           hemi='left', view='lateral',
                           bg_map=fsaverage['sulc_left'], bg_on_data=True,
                           darkness=.5, cmap=cmap,
                           vmin=vmin, vmax=vmax,
                           figure=fig, axes=axs[2])


    axs[1].view_init(elev=90, azim=270)
    
    for i in range(3):
        if i in [1]:
            axs[i].dist = 5.7
        else:
            axs[i].dist = 6

    fig.tight_layout()
    
    save_fname = f'data/output/{label}.{fmt}'
    fig.savefig(save_fname)
    plt.close(fig)
    
    return save_fname
