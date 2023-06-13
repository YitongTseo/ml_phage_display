import pathlib
import sys
HOME_DIRECTORY = pathlib.Path().absolute()
sys.path.append(str(HOME_DIRECTORY) + '/src')

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import pyplot as plt
import plotly.graph_objects as go
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.patches as mpatches
import pdb
import matplotlib
import copy
from matplotlib.pyplot import figure


TRUE_POSITIVE = (0, 1, 0, 0.5)
FALSE_POSITIVE = (0, 0, 1, 0.5)
FALSE_NEGATIVE = (1, 0, 0, 0.5)
TRUE_NEGATIVE = (0.8, 0.8, 0.8, 0.03)

def plot_relations_in_plotly(x_idx, y_idx, peptides, datapoints, vals=["Pval", "FC", "ER"]):
    fig = go.Figure(
        [
            go.Scatter(
                x=datapoints[:, x_idx],
                y=datapoints[:, y_idx],
                text=peptides,
                mode="markers",
            )
        ]
    )
    fig.show()

# def apply_lines(h, top_k_datapoints,x_idx, y_idx, extent, cmap, gridsize, style='around_border'):
#     if style == 'around_border':

    
    # if style == 'in_border':
    #     def hexLines(a=None,i=None,off=[0,0]):
    #         '''regular hexagon segment lines as `(xy1,xy2)` in clockwise 
    #         order with points in line sorted top to bottom
    #         for irregular hexagon pass both `a` (vertical) and `i` (horizontal)'''
    #         if a is None: a = 2 / np.sqrt(3) * i;    
    #         if i is None: i = np.sqrt(3) / 2 * a;     
    #         h  = a / 2 
    #         xy = np.array([ [ [ 0, a], [ i, h] ], 
    #                         [ [ i, h], [ i,-h] ], 
    #                         [ [ i,-h], [ 0,-a] ], 
    #                         [ [-i,-h], [ 0,-a] ], #flipped
    #                         [ [-i, h], [-i,-h] ], #flipped
    #                         [ [ 0, a], [-i, h] ]  #flipped
    #                     ])  
    #         return xy+off


    #     #get hexagon centers that should be highlighted
    #     verts = h.get_offsets()
    #     pdb.set_trace()
    #     # cnts  = h.get_array()
    #     cnts = top_k_datapoints
    #     # highl = verts[cnts > .5*cnts.max()]
    #     highl = verts[cnts > 0]

    #     #create hexagon lines
    #     a = ((verts[0,1]-verts[1,1])/3).round(6)
    #     i = ((verts[1:,0]-verts[:-1,0])/2).round(6)
    #     i = i[i>0][0]
    #     lines = np.concatenate([hexLines(a,i,off) for off in highl])

    #     #select contour lines and draw
    #     uls,c = np.unique(lines.round(4),axis=0,return_counts=True)
    #     for l in uls[c==1]: plt.plot(*l.transpose(),'w-',lw=2,scalex=False,scaley=False)


def plot_fancy_hexbin_relations(
    x_idx, 
    y_idx, 
    datapoints,
    ordering,
    all_positives, 
    vals=["-log(P-value)", "log(Fold Change)", "Enrichment Ratio"], 
    line_color='black',
    top_k=500,
    gridsize=30,
    title=None,
    plot_labels=True,
    line_weight=8,
    highlight_alpha=0.3
):
    # fig = plt.figure(figsize=(6,6))

    figure(figsize=(12, 12), dpi=80)
    font = {'family' : 'Arial',
        'weight' : 'normal',
        'size'   : 22}

    matplotlib.rc('font', **font)

    if plot_labels:
        plt.xlabel(vals[x_idx])
        plt.ylabel(vals[y_idx])
    if title is None:
        plt.title(vals[x_idx] + ' vs ' + vals[y_idx])
    else:
        plt.title(title)

    cmap = copy.copy(matplotlib.colormaps['Greys']) # copy the default cmap
    cmap.set_bad((1,1,1))


    #original plot
    extent = (
        datapoints[:, x_idx].min() - (0.1 * datapoints[:, x_idx].var()),
        datapoints[:, x_idx].max() + (0.3 * datapoints[:, x_idx].var()),
        datapoints[:, y_idx].min() - (0.1 * datapoints[:, y_idx].var()),
        datapoints[:, y_idx].max() + (0.3 * datapoints[:, y_idx].var()),
    )
    print(extent)
    cfg = dict(
        x=datapoints[:, x_idx], 
        y=datapoints[:, y_idx], 
        cmap=cmap, 
        norm=matplotlib.colors.LogNorm(clip=False), 
        gridsize=gridsize, 
        extent=extent
    )
    h  = plt.hexbin( ec="white",lw=1,zorder=-5,**cfg)

    if all_positives is not None:
        plt.scatter(
            x=datapoints[all_positives.astype(bool)][:, x_idx],
            y=datapoints[all_positives.astype(bool)][:, y_idx],
            c='forestgreen',
            marker='x',
            alpha=0.5
        )

    if ordering is not None:
        top_k_mask = ordering >= np.partition(ordering, kth=-top_k)[-top_k]
        top_k_datapoints = datapoints[top_k_mask]
        custom_cmap = matplotlib.colors.ListedColormap([line_color])
        # top_k_cfg = dict(
        #     x=top_k_datapoints[:, x_idx], 
        #     y=top_k_datapoints[:, y_idx], 
        #     cmap=custom_cmap,  
        #     # norm=matplotlib.colors.LogNorm(clip=False), 
        #     # cmap=cmap, 
        #     # norm=matplotlib.colors.LogNorm(clip=False), 
        #     gridsize=gridsize, 
        #     extent=extent
        # )

        plt.hexbin(
            x=top_k_datapoints[:, x_idx], 
            y=top_k_datapoints[:, y_idx], 
            cmap=cmap, 
            norm=matplotlib.colors.LogNorm(clip=False), 
            gridsize=gridsize, 
            extent=extent,

            # alpha=0.5,
            ec=line_color,
            lw=line_weight,
            zorder=-3,
            mincnt=1
        )
        plt.hexbin(
            x=top_k_datapoints[:, x_idx], 
            y=top_k_datapoints[:, y_idx], 
            cmap=cmap, 
            norm=matplotlib.colors.LogNorm(clip=False), 
            gridsize=gridsize, 
            extent=extent,
            
            ec="white", 
            lw=0, 
            zorder=-2, 
            mincnt=1
        )
        plt.hexbin(
            alpha=0.5,
            cmap=matplotlib.colors.ListedColormap([line_color]), 

            x=top_k_datapoints[:, x_idx], 
            y=top_k_datapoints[:, y_idx], 
            gridsize=gridsize, 
            extent=extent,
            
            ec="white", 
            lw=0, 
            zorder=-2, 
            mincnt=1
        )
        
        
        #draw thick white contours + overlay previous style
        # plt.hexbin( fc=line_color, ec=line_color, lw=line_weight, zorder=-2,mincnt=1,**top_k_cfg)
        # plt.hexbin(ec="black", lw=1, zorder=-2, mincnt=1, **top_k_cfg)
        # plt.hexbin(alpha=highlight_alpha, ec=line_color, zorder=-1, mincnt=1, **top_k_cfg)

        # plt.hexbin( alpha=0.5, ec=line_color, lw=line_weight, zorder=-3,mincnt=1,**top_k_cfg)
        # plt.hexbin(alpha=0.5, ec="white", lw=0, zorder=-2, mincnt=1, **top_k_cfg)
        # plt.hexbin( alpha=1.0, ec=line_color, fc=line_color,lw=10,zorder=-1,mincnt=1,**top_k_cfg)

        # plt.hexbin( ec=line_color, lw=line_weight, zorder=-2,mincnt=3,**top_k_cfg)
        # plt.hexbin( ec="white",lw=1,zorder=-1,mincnt=1,**top_k_cfg)
        # # plt.hexbin( alpha=highlight_alpha, fc=line_color,lw=1,zorder=-1,mincnt=1,**top_k_cfg)


        plt.xlim(extent[0], extent[1]) #required as second call of plt.hexbin()
        plt.ylim(extent[2], extent[3]) #strangely affects the limits ...

    # plt.colorbar()
    plt.show()


def plot_relations(
    x_idx, 
    y_idx, 
    datapoints,
    ordering,
    all_positives, 
    kind="hex", 
    vals=["Pval", "FC", "ER"], 
    top_k=500
):
    g = sns.jointplot(x=datapoints[:, x_idx], y=datapoints[:, y_idx], kind=kind)
    plt.xlabel(vals[x_idx])
    plt.ylabel(vals[y_idx])
    plt.title(vals[x_idx] + ' vs ' + vals[y_idx])
    
    if ordering is not None and all_positives is not None:
        g.ax_joint.cla()
        top_k_mask = ordering >= np.partition(ordering, kth=-top_k)[-top_k]
        colors = []
        for datapoint, is_true, is_in_top_k in zip(datapoints, all_positives, top_k_mask):
            if is_true and is_in_top_k:
                
                colors.append(TRUE_POSITIVE)

            if is_true and not is_in_top_k:
                colors.append(FALSE_NEGATIVE) 

            if not is_true and is_in_top_k:
                colors.append(FALSE_POSITIVE) 

            if not is_true and not is_in_top_k:
                colors.append(TRUE_NEGATIVE)

        plt.scatter(
            x=datapoints[:, x_idx],
            y=datapoints[:, y_idx],
            c=colors,
        )
        
        g.ax_joint.legend(handles=[
            mpatches.Patch(color=TRUE_POSITIVE, label='True Positive (peptide hit in top 500)'),
            mpatches.Patch(color=FALSE_POSITIVE, label='False Positive (peptide miss in top 500)'),
            mpatches.Patch(color=FALSE_NEGATIVE, label='False Negative (peptide hit not in top 500)'),
            mpatches.Patch(color=TRUE_NEGATIVE, label='True Negative (peptide miss not in top 500)'),
        ],loc='best')
    plt.show()



def plot_relations_in_3D(
    x_idx,
    y_idx,
    uncertainty,
    datapoints,
    ordering,
    all_positives,
    title="",
    vals=["Pval", "FC", "ER"],
    top_k=500,
):
    sns.set_style("whitegrid", {"axes.grid": False})
    fig = plt.figure(figsize=(6, 6))

    ax = Axes3D(fig)
    if ordering is not None and all_positives is not None:
        top_k_mask = ordering >= np.partition(ordering, kth=-top_k)[-top_k]
        colors = []
        for is_true, is_in_top_k in zip(all_positives, top_k_mask):
            if is_true and is_in_top_k:
                colors.append(TRUE_POSITIVE) 
            
            if is_true and not is_in_top_k:
                colors.append(FALSE_NEGATIVE) 

            if not is_true and is_in_top_k:
                colors.append(FALSE_POSITIVE) 

            if not is_true and not is_in_top_k:
                colors.append(TRUE_NEGATIVE)

        ax.legend(handles=[
            mpatches.Patch(color=TRUE_POSITIVE, label='True Positive (peptide hit in top 500)'),
            mpatches.Patch(color=FALSE_POSITIVE, label='False Positive (peptide miss in top 500)'),
            mpatches.Patch(color=FALSE_NEGATIVE, label='False Negative (peptide hit not in top 500)'),
            mpatches.Patch(color=TRUE_NEGATIVE, label='True Negative (peptide miss not in top 500)'),
        ])
    else:
        colors = uncertainty

    ax.scatter(
        xs=datapoints[:, x_idx],
        ys=datapoints[:, y_idx],
        zs=uncertainty,
        c=colors,
        marker="o",
    )
    ax.set_xlabel("Pred " + vals[x_idx])
    ax.set_ylabel("Pred " + vals[y_idx])
    ax.set_zlabel("Uncertainty")

    plt.title(title)
    # ax.view_init(elev=10.0, azim=90)
    plt.show()