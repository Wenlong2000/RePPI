import os
import sys
sys.path.insert(1, '../')
import numpy as np
import folktables
import pdb
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import seaborn as sns
import pandas as pd

import scipy
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from matplotlib.ticker import FormatStrFormatter
from matplotlib.ticker import LogLocator
from matplotlib.ticker import ScalarFormatter
from matplotlib.ticker import NullFormatter

def get_data(year,features,outcome, randperm=True):
    data_source = folktables.ACSDataSource(survey_year=year, horizon='1-Year', survey='person')
    acs_data = data_source.get_data(states=["CA"], download=True)
    income_features = acs_data[features].fillna(-1)
    income = acs_data[outcome].fillna(-1)
    # drop rows with missing values in income
    income_features = income_features[income > 0]
    income = income[income > 0]

    employed = np.isin(acs_data['COW'], np.array([1,2,3,4,5,6,7]))
    if randperm:
        shuffler = np.random.permutation(income.shape[0])
        income_features, income, employed = income_features.iloc[shuffler], income.iloc[shuffler], employed[shuffler]
    return income_features, income, employed

def transform_features(features, ft, enc=None):
    c_features = features.T[ft == "c"].T.astype(str)
    if enc is None:
        enc = OneHotEncoder(handle_unknown='ignore', drop='if_binary', sparse_output=False)
        enc.fit(c_features)
    c_features = enc.transform(c_features)
    features = scipy.sparse.csc_matrix(np.concatenate([features.T[ft == "q"].T.astype(float), c_features], axis=1))
    return features, enc

def lighten_color(color, amount=0.5):
    """
    Lightens the given color by multiplying (1-luminosity) by the given amount.
    Input can be matplotlib color string, hex string, or RGB tuple.

    Examples:
    >> lighten_color('g', 0.3)
    >> lighten_color('#F034A3', 0.6)
    >> lighten_color((.3,.55,.1), 0.5)
    """
    import matplotlib.colors as mc
    import colorsys
    try:
        c = mc.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], 1 - amount * (1 - c[1]), c[2])


def make_width_coverage_plot(df, estimand_title, filename, theta_true, alpha = 0.1, n_l = 0, n_u = np.inf, num_trials = 100, n_example_ind = 0, col = None):
    num_ints = 5
    inds = np.random.choice(num_trials, num_ints)
    ns = df["n"].unique()
    estimators = df["estimator"].unique()
    n_example = ns[n_example_ind]
    ints = [ [] for _ in range(len(estimators)) ]
    widths = np.zeros((len(estimators), len(ns)))

    # compute example intervals and average widths
    for i in range(len(estimators)):
        for j in range(len(ns)):
            widths[i,j] = df[(df.estimator == estimators[i]) & (df["n"] == ns[j])]['interval width'].mean()

        for j in range(num_ints):
            ind = inds[j]
            ints[i].append([df[(df.estimator == estimators[i]) & (df['n'] == n_example)].iloc[ind].lb, df[(df.estimator == estimators[i]) & (df['n'] == n_example)].iloc[ind].ub])

    n_l = n_l
    n_u = n_u
    inds_n = np.where((ns>n_l) & (ns<n_u))[0] # budget indices that will be plotted
    x_ticks = np.logspace(np.log10(min(df['n'][(df['n'] > n_l)])), np.log10(max(df['n'][(df['n'] < n_u)])), num=5) # adjust 'num' for more/less ticks
    x_ticks = [int(x) for x in x_ticks]
    x_ticks = np.linspace(min(df['labeled ratio']), max(df['labeled ratio']), num=5)
    y_ticks = np.logspace(np.log10(np.min(widths[:,inds_n[-1]])), np.log10(np.max(widths[:,inds_n[0]])), num=5) # adjust 'num' for more/less ticks

    # plotting params
    gap = 0.03
    start1 = 0.6
    start2 = 0.45
    start3 = 0.3
    start4 = 0.15
    linewidth_inner = 5
    linewidth_outer = 7
    if col == None:
        col = [sns.color_palette("pastel")[1], sns.color_palette("pastel")[2], sns.color_palette("pastel")[0], sns.color_palette("pastel")[3]]
    sns.set_theme(font_scale=1.5, style='white', palette=col, rc={'lines.linewidth': 3})
    fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(15,3.3))
    sns.lineplot(ax=axs[1],data=df[(df['n'] > n_l) & (df['n'] < n_u)], x='labeled ratio', y='interval width', hue='estimator', alpha=0.8)
    sns.lineplot(ax=axs[2],data=df[(df['n'] > n_l) & (df['n'] < n_u)], x='labeled ratio', y='coverage', hue='estimator', alpha=0.8, errorbar=None)

    axs[0].axvline(theta_true, color='gray', linestyle='dashed')
    for i in reversed(range(num_ints)):
        if i == 0:
            axs[0].plot([ints[0][i][0] , ints[0][i][1] ],[start1+i*gap,start1+i*gap], linewidth=linewidth_inner, color=lighten_color(col[0],0.6), path_effects=[pe.Stroke(linewidth=linewidth_outer, offset=(-1,0), foreground=col[0]), pe.Stroke(linewidth=linewidth_outer, offset=(1,0), foreground=col[0]), pe.Normal()],  solid_capstyle='butt')
            axs[0].plot([ints[1][i][0] , ints[1][i][1] ],[start2+i*gap, start2+i*gap], linewidth=linewidth_inner, color=lighten_color(col[1],0.6), path_effects=[pe.Stroke(linewidth=linewidth_outer, offset=(-1,0), foreground=col[1]), pe.Stroke(linewidth=linewidth_outer, offset=(1,0), foreground=col[1]), pe.Normal()],  solid_capstyle='butt')
            axs[0].plot([ints[2][i][0] , ints[2][i][1] ],[start3+i*gap, start3+i*gap], linewidth=linewidth_inner, color=lighten_color(col[2],0.6), path_effects=[pe.Stroke(linewidth=linewidth_outer, offset=(-1,0), foreground=col[2]), pe.Stroke(linewidth=linewidth_outer, offset=(1,0), foreground=col[2]), pe.Normal()],  solid_capstyle='butt')
            axs[0].plot([ints[3][i][0] , ints[3][i][1] ],[start4+i*gap, start4+i*gap], linewidth=linewidth_inner, color=lighten_color(col[3],0.6), path_effects=[pe.Stroke(linewidth=linewidth_outer, offset=(-1,0), foreground=col[3]), pe.Stroke(linewidth=linewidth_outer, offset=(1,0), foreground=col[3]), pe.Normal()],  solid_capstyle='butt')
        if i > 0:
            axs[0].plot([ints[0][i][0], ints[0][i][1]],[start1+i*gap,start1+i*gap], linewidth=linewidth_inner, color= lighten_color(col[0],0.6), path_effects=[pe.Stroke(linewidth=linewidth_outer, offset=(-1,0), foreground=col[0]), pe.Stroke(linewidth=linewidth_outer, offset=(1,0), foreground=col[0]), pe.Normal()], solid_capstyle='butt')
            axs[0].plot([ints[1][i][0] , ints[1][i][1]],[start2+i*gap, start2+i*gap], linewidth=linewidth_inner, color=lighten_color(col[1],0.6), path_effects=[pe.Stroke(linewidth=linewidth_outer, offset=(-1,0), foreground=col[1]), pe.Stroke(linewidth=linewidth_outer, offset=(1,0), foreground=col[1]), pe.Normal()], solid_capstyle='butt')
            axs[0].plot([ints[2][i][0] , ints[2][i][1]],[start3+i*gap, start3+i*gap], linewidth=linewidth_inner, color=lighten_color(col[2],0.6), path_effects=[pe.Stroke(linewidth=linewidth_outer, offset=(-1,0), foreground=col[2]), pe.Stroke(linewidth=linewidth_outer, offset=(1,0), foreground=col[2]), pe.Normal()], solid_capstyle='butt')
            axs[0].plot([ints[3][i][0] , ints[3][i][1]],[start4+i*gap, start4+i*gap], linewidth=linewidth_inner, color=lighten_color(col[3],0.6), path_effects=[pe.Stroke(linewidth=linewidth_outer, offset=(-1,0), foreground=col[3]), pe.Stroke(linewidth=linewidth_outer, offset=(1,0), foreground=col[3]), pe.Normal()], solid_capstyle='butt')
    axs[0].set_xlabel(estimand_title, fontsize=16)
    axs[0].set_yticks([])

    axs[1].get_legend().remove()
    axs[1].set(yscale='log')
    axs[1].set_xticks(x_ticks)
    axs[1].set_yticks(y_ticks)
    axs[1].xaxis.set_minor_formatter(NullFormatter())
    axs[1].yaxis.set_minor_formatter(NullFormatter())
    axs[1].get_xaxis().set_major_formatter(ScalarFormatter())
    axs[1].get_yaxis().set_major_formatter(ScalarFormatter())
    axs[1].yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
    axs[1].grid(True)

    axs[2].axhline(1-alpha, color="#888888", linestyle='dashed', zorder=1, alpha=0.8)
    handles, labels = axs[2].get_legend_handles_labels()
    # axs[2].legend(handles=handles, labels=labels, loc='lower right', bbox_to_anchor=(1.6, 0.43))
    axs[2].legend(handles=handles, labels=labels, loc='lower right')
    axs[2].set_ylim([0.6,1])
    x_ticks_coverage = np.linspace(np.min(ns[ns>n_l]), np.max(ns[ns<n_u]), 5)
    x_ticks_coverage = [int(x) for x in x_ticks_coverage]
    axs[2].set_xticks(x_ticks)
    # axs[2].set_xlim([np.min(ns[ns>n_l]), np.max(ns[ns<n_u])])
    axs[2].grid(True)

    sns.despine(top=True, right=True)
    sns.despine(left=True, ax = axs[0])
    plt.tight_layout()

    # save plot
    plt.savefig(filename)
    plt.show()


def make_length_table(df):
    ns = df["n"].unique()
    estimators = df["estimator"].unique()
    widths = np.zeros((len(estimators), len(ns)))

    # compute example intervals and average widths
    for i in range(len(estimators)):
        for j in range(len(ns)):
            widths[i, j] = df[(df.estimator == estimators[i]) & (df["n"] == ns[j])]['interval width'].mean()
    df_widths = pd.DataFrame(widths.T, columns=estimators)
    df_widths['n'] = ns
    return df_widths


def make_coverage_table(df):
    ns = df["n"].unique()
    estimators = df["estimator"].unique()
    coverages = np.zeros((len(estimators), len(ns)))

    # compute example intervals and average widths
    for i in range(len(estimators)):
        for j in range(len(ns)):
            coverages[i, j] = df[(df.estimator == estimators[i]) & (df["n"] == ns[j])]['coverage'].mean()
    df_coverages = pd.DataFrame(coverages.T, columns=estimators)
    df_coverages['n'] = ns
    return df_coverages


def make_error_table(df):
    ns = df["n"].unique()
    estimators = df["estimator"].unique()
    errors = np.zeros((len(estimators), len(ns)))

    # compute example intervals and average widths
    for i in range(len(estimators)):
        for j in range(len(ns)):
            errors[i, j] = df[(df.estimator == estimators[i]) & (df["n"] == ns[j])]['error'].mean()
    df_errors = pd.DataFrame(errors.T, columns=estimators)
    df_errors['n'] = ns
    return df_errors