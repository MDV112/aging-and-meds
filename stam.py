import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import pickle
from scipy import stats
from project_settings import ProSet
from project_settings import init_parser
import subprocess as sp
import os
from scipy import stats
import argparse


def select_rows(df, **kwargs):  # values have to be list Rearrange=0,State=0,Measure=0,nbeats=0,age=0
     chosen_df = df
     for key in kwargs.keys():
          if key != 0:
               chosen_df = chosen_df.loc[chosen_df.loc[:, key].isin(kwargs[key])]
     return chosen_df


def reindexing_df(df, chosen_nbeats=50):
     index = pd.MultiIndex.from_arrays([df['nbeats'], df['State']],
                                       names=('n', 'S'))
     df_nbeats_age_6 = pd.DataFrame(df.iloc[:, 4].values, index=index)
     stat_nbeats = df_nbeats_age_6.groupby(level=1).describe()
     df_ages = df.loc[df.loc[:, 'nbeats'] == chosen_nbeats, :]
     df_ages.drop(columns=['Rearrange', 'Measure', 'nbeats'], inplace=True)
     state = df_ages['State']
     df_ages = df_ages.T.loc['6':'24', :].values
     stat_ages = {'State': state, 'mean': df_ages.mean(axis=0), 'std': df_ages.std(axis=0),
                  'min': df_ages.min(axis=0), 'max': df_ages.max(axis=0), '25': np.percentile(df_ages, 25, axis=0),
                  '50': np.percentile(df_ages, 50, axis=0), '75': np.percentile(df_ages, 75, axis=0)}
     y_age_EER = df.loc[df.loc[:, 'nbeats'] == chosen_nbeats, :]
     df_ages = pd.DataFrame(df.iloc[:, 4:].values, index=index)
     return stat_nbeats, df_ages


def report_statistics(df):
     df_F1 = df.loc[(df.loc[:, 'Measure'] == 'F1') & (df.loc[:, 'nbeats'] <= 350), :]
     df_EER = df.loc[(df.loc[:, 'Measure'] == 'ERR') & (df.loc[:, 'nbeats'] <= 350), :]
     stat_nbeats_age_6_F1, df_ages_F1 = reindexing_df(df_F1)
     stat_nbeats_age_6_EER, df_ages_EER = reindexing_df(df_EER)
     a=1
     # index = pd.MultiIndex.from_arrays([df['nbeats'], df['State'], df['Measure']],
     #                                   names=('n', 'S', 'M'))
     # df_nbeats_age_6 = pd.DataFrame(df.iloc[:, 4].values, index=index)
     # df_ages = pd.DataFrame(df.iloc[:, 4:].values, index=index)
     # df_ages.groupby(level=1).describe()


def exclude_nbeats(df, mode, state='basal'):
     EER_df = df.loc[(df.loc[:, 'Measure'] == 'ERR') & (df.loc[:, 'State'] == state), :]
     tt = np.argwhere((EER_df['6'] > 0.499).values).squeeze()
     return EER_df['nbeats'].values[tt[0]]

def plot_res(df, mode, state='basal', nbeats=50, nbeats_exclude=600):
     plt.rcParams["font.family"] = "Times New Roman"
     fig = plt.subplots(figsize=(15, 15))
     # nbeats_exclude = exclude_nbeats(df, mode, state=state)
     y_F1 = df.loc[(df.loc[:, 'Measure'] == 'F1') & (df.loc[:, 'State'] == state) &
                   (df.loc[:, 'nbeats'] <= nbeats_exclude) & (df.loc[:, 'nbeats'] > 10), '6']
     y_EER = df.loc[(df.loc[:, 'Measure'] == 'ERR') & (df.loc[:, 'State'] == state) &
                    (df.loc[:, 'nbeats'] <= nbeats_exclude) & (df.loc[:, 'nbeats'] > 10), '6']
     df4plot = pd.DataFrame({'X': np.unique(df['nbeats'].loc[(df['nbeats'] <= nbeats_exclude ) & (df['nbeats'] > 10)]), 'Y_F1': y_F1.values, 'Y_EER': y_EER.values})
     # ax = sns.barplot(x=df4plot["X"], y=df4plot["Y_F1"], color='b', label='F1')
     ax = sns.barplot(x=df4plot["X"], y=df4plot["Y_EER"], color='r', label='EER')

     x_dashed = np.arange(len(df4plot["X"]) + 1)
     ax = sns.lineplot(x=x_dashed, y=np.ones_like(x_dashed)*(y_EER.min()), linestyle="dashed",
                       linewidth=6, color='g')
     ax.text(10, y_EER.min(), '{:.2f}'.format(y_EER.min()), fontsize=30, weight='bold')
     ax.set_xlabel(xlabel="# of beats", fontsize=24, weight='bold')
     ax.set_ylabel(ylabel="EER", fontsize=24, weight='bold')
     ax.set_xticklabels([str(i) for i in df4plot["X"]], fontsize=14, weight='bold')
     ax.set_yticklabels([str("{:.1f}".format(i)) for i in ax.get_yticks()], fontsize=14, weight='bold')
     for _, s in ax.spines.items():
          s.set_linewidth(5)
     plt.legend(fontsize=30)
     sns.despine(fig=None, ax=ax, top=True, right=True, left=False, bottom=False, offset=None, trim=False)

     save_fig_pth = '/home/smorandv/ac8_and_aging_NEW/ac8_and_aging/rr_datasets'

     mode2fig = {'equal': '_equal/', 'non_equal': '/', 'bas_on_int': '_equal/bas_on_int/', 'up2_21': '_equal_up2_21/',
                 'up2_21_bas_on_int': '_equal_up2_21/bas_on_int/'}
     save_fig_pth += mode2fig[mode]
     plt.savefig(save_fig_pth + 'nbeats_' + state + '.png')

     fig = plt.subplots(figsize=(15, 15))
     y_age_F1 = df.loc[(df.loc[:, 'Measure'] == 'F1') & (df.loc[:, 'State'] == state) & (df.loc[:, 'nbeats'] == nbeats),
                       ['6', '9', '12', '15', '18', '21', '24']].values.flatten()
     y_age_EER = df.loc[
                 (df.loc[:, 'Measure'] == 'ERR') & (df.loc[:, 'State'] == state) & (df.loc[:, 'nbeats'] == nbeats),
                 ['6', '9', '12', '15', '18', '21', '24']].values.flatten()
     df4plot_age = pd.DataFrame(
          {'X': np.arange(6, 27, 3), 'Y_F1': y_age_F1, 'Y_EER': y_age_EER})
     # ax = sns.barplot(x=df4plot_age["X"], y=df4plot_age["Y_F1"], color='b', label='F1')
     ax = sns.barplot(x=df4plot_age["X"], y=df4plot_age["Y_EER"], color='r', label='EER')
     x_dashed = np.arange(len(df4plot_age["X"]) + 1)
     ax = sns.lineplot(x=x_dashed, y=np.ones_like(x_dashed) * (y_age_EER.min()), linestyle="dashed",
                       linewidth=6, color='g')
     ax.text(len(x_dashed)-1, y_EER.min(), '{:.2f}'.format(y_age_EER.min()), fontsize=30, weight='bold')
     ax.set_xlabel(xlabel="Age [m]", fontsize=34, weight='bold')
     ax.set_ylabel(ylabel="EER", fontsize=34, weight='bold')
     ax.set_xticklabels([str(i) for i in df4plot_age["X"]], fontsize=26, weight='bold')
     ax.set_yticklabels([str("{:.1f}".format(i)) for i in ax.get_yticks()], fontsize=26, weight='bold')
     for _, s in ax.spines.items():
          s.set_linewidth(5)
     plt.legend(fontsize=30)
     sns.despine(fig=None, ax=ax, top=True, right=True, left=False, bottom=False, offset=None, trim=False)
     plt.savefig(save_fig_pth + 'Age_nbeats_' + str(nbeats) + '_' + state + '.png')
     a=1
     # plt.savefig('/home/smorandv/ac8_and_aging_NEW/ac8_and_aging/rr_datasets_equal_up2_21/' + 'Age_nbeats_' + str(nbeats)
     # + '_' + state + '.png')


def prepare_empty_df(idx, mode='equal'):
     d = {'Rearrange': 9*['F'] + 9*['T'],
          'State': 3*['basal'] + 3*['int'] + 3*['comb'] + 3*['basal'] + 3*['int'] + 3*['comb'],
          'Measure': 6*['ERR', 'ACC', 'F1']}
     df = pd.DataFrame(data=d)
     w = np.arange(100, 1050, 50)
     v1 = np.array([10, 25, 50, 75])
     aa = np.hstack([v1, w])
     k = np.repeat(aa, len(df))
     df_new = pd.concat(len(aa)*[df])
     df_new['nbeats'] = k
     df_new.loc[:, ['6', 'n_one_beats_6', '9', 'n_one_beats_9', '12', 'n_one_beats_12', '15', 'n_one_beats_15',
                    '18', 'n_one_beats_18', '21', 'n_one_beats_21', '24', 'n_one_beats_24']] = np.nan
     save_pkl_pth = '/home/smorandv/ac8_and_aging_NEW/ac8_and_aging/rr_datasets'

     mode2fig = {'equal': '_equal/', 'non_equal': '/', 'bas_on_int': '_equal/bas_on_int/', 'up2_21': '_equal_up2_21/',
                 'up2_21_bas_on_int': '_equal_up2_21/bas_on_int/'}
     save_pkl_pth += mode2fig[mode] + 'res_df_' + str(idx) + '.pkl'
     with open(save_pkl_pth, 'wb') as f:
          pickle.dump(df_new, f)



def plot_one_win(plot_df, mode, chosen_state):
     one_win_per_age = plot_df[['n_one_beats_6', 'n_one_beats_9', 'n_one_beats_12', 'n_one_beats_15', 'n_one_beats_18',
                                'n_one_beats_21', 'n_one_beats_24']]
     w = np.arange(100, 1050, 50)
     v1 = np.array([25, 50, 75])
     nbeats = np.hstack([v1, w])
     fig = plt.subplots(figsize=(15, 15))
     for ii in range(7):
          # plt.scatter(nbeats, one_win_per_age.values[1:, ii])
          ax = sns.scatterplot(x=nbeats, y=one_win_per_age.values[1:, ii], s=100)
     ax.set_xlabel(xlabel="# of beats", fontsize=34, weight='bold')
     ax.set_ylabel(ylabel="[%]", fontsize=34, weight='bold')
     ax.set_xticklabels(["{}".format(int(i)) for i in ax.get_xticks()], fontsize=26, weight='bold')
     ax.set_yticklabels([str("{:.1f}".format(i)) for i in ax.get_yticks()], fontsize=26, weight='bold')
     ax.legend(['6', '9', '12', '15', '18', '21', '24'], fontsize=30)
     # for _, s in ax.spines.items():
     #      s.set_linewidth(5)
     # plt.legend(fontsize=30)
     sns.despine(fig=None, ax=ax, top=True, right=True, left=False, bottom=False, offset=None, trim=False)
     save_fig_pth = '/home/smorandv/ac8_and_aging_NEW/ac8_and_aging/rr_datasets'

     mode2fig = {'equal': '_equal/', 'non_equal': '/', 'bas_on_int': '_equal/bas_on_int/', 'up2_21': '_equal_up2_21/',
                 'up2_21_bas_on_int': '_equal_up2_21/bas_on_int/'}
     save_fig_pth += mode2fig[mode]
     # save_fig_pth = '/home/smorandv/ac8_and_aging_NEW/ac8_and_aging/rr_datasets'
     # if mode == 'equal':
     #      save_fig_pth += '_' + mode + '/'
     # elif mode == 'non_equal':
     #      save_fig_pth += '/'
     # else:
     #      save_fig_pth += '_equal_' + mode + '/'
     plt.savefig(save_fig_pth + 'one_window_' + chosen_state + '.png')
     # plt.savefig(
     #      '/home/smorandv/ac8_and_aging_NEW/ac8_and_aging/rr_datasets_equal_up2_21/' + 'one_window_' + chosen_state + '.png')


def barplot(x,y,yerr, mode, state, nbeats, age_flag='nbeats', **extra_kwargs):
     fig, ax = plt.subplots(figsize=(20, 20))
     ax.bar(np.arange(len(y)), y, yerr=yerr, color='red')
     ax.set_xticks(np.arange(len(y)))  # make xticks to be with equal intervals (note that x in the bar is the same)
     ax.set_xticklabels(x, fontsize=26, weight='bold')  # rewrites the labels of ticks
     plt.yticks(fontsize=26, weight='bold')
     # x_dashed = np.arange(len(y) + 1)
     # sns.lineplot(x=x_dashed, y=np.ones_like(x_dashed) * (y.min()), linestyle="dashed", linewidth=6, color='g')
     # ax.text(len(x_dashed) - 1, y.min(), '{:.2f}'.format(y.min()), fontsize=30, weight='bold')
     if age_flag == 'age':
          ax.set_xlabel(xlabel="Age [m]", fontsize=38, weight='bold')
     elif (age_flag == 'nbeats') or (age_flag == 'cv'):
          ax.set_xlabel(xlabel="nbeats", fontsize=38, weight='bold')
     ax.set_ylabel(ylabel="EER", fontsize=38, weight='bold')
     if not(age_flag == 'cv'):
          ax.set_ylim(0, 0.5)
     for _, s in ax.spines.items():
          s.set_linewidth(5)
     sns.despine(fig=None, ax=ax, top=True, right=True, left=False, bottom=False, offset=None, trim=False)
     # plt.show()
     save_fig_pth = '/home/smorandv/ac8_and_aging_NEW/ac8_and_aging/rr_datasets'

     mode2fig = {'equal': '_equal/', 'non_equal': '/', 'bas_on_int': '_equal/bas_on_int/', 'up2_21': '_equal_up2_21/',
                 'up2_21_bas_on_int': '_equal_up2_21/bas_on_int/'}
     save_fig_pth += mode2fig[mode]
     if age_flag == 'age':
          plt.savefig(save_fig_pth + 'Age_nbeats_' + str(nbeats) + '_' + state + '.png')
     elif age_flag == 'nbeats':
          plt.savefig(save_fig_pth + 'nbeats_' + state + '.png')
     elif age_flag =='cv':
          plt.savefig(save_fig_pth + 'nbeats_cv_' + state + '.png')
     a=1

# def plot_cv_diff()

def plot_volcano(mat_volcano1, mat_volcano2, leg):
     ee = np.zeros(mat_volcano1.shape[1])
     for i in range(mat_volcano1.shape[1]):
          tmp = stats.mannwhitneyu(mat_volcano2[:, 0], mat_volcano2[:, i])
          ee[i] = tmp[1]
     x = np.log2(mat_volcano1[0, :] / mat_volcano1[0, 0])
     y = -np.log10(ee)
     fig, ax = plt.subplots(figsize=(10, 10))
     x_dashed = np.linspace(x.min(), x.max(), len(x))
     sns.lineplot(x=x_dashed, y=np.ones_like(x_dashed) * -np.log10(0.05), linestyle="dashed", linewidth=2, color='g')
     for g in np.unique(leg):
          i = np.where(leg == g)
          ax.scatter(x[i], y[i], label=g)
     ax.legend()
     # plt.show()


def avg_std_bootstrap_df(save_pkl_pth, mode2fig, mode, p, state, nbeats, data4volcano='nbeats'):
     ages = [6, 9, 12, 15, 18, 21, 24]
     mat_EER = np.zeros((p.bootstrap_total_folds, 14))
     mat_age_EER = np.zeros((p.bootstrap_total_folds, len(ages)))
     avg_std_EER = np.zeros((3, mat_EER.shape[1]))
     avg_std_age_EER = np.zeros((3, mat_age_EER.shape[1]))
     for j in range(1, p.bootstrap_total_folds + 1):
          pth = save_pkl_pth + mode2fig[mode] + 'res_df_' + str(j) + '.pkl'
          with open(pth, 'rb') as f:
               dd = pickle.load(f)
          chosen_df = select_rows(dd, **dict(Rearrange=[rearrange], State=['int', 'basal', 'comb'], Measure=['ERR', 'F1']))
          y_EER = chosen_df.loc[(chosen_df.loc[:, 'Measure'] == 'ERR') & (chosen_df.loc[:, 'State'] == state) &
                         (chosen_df.loc[:, 'nbeats'] <= 600) & (chosen_df.loc[:, 'nbeats'] > 10), '6']
          y_age_EER = chosen_df.loc[
               (chosen_df.loc[:, 'Measure'] == 'ERR') & (chosen_df.loc[:, 'State'] == state) & (chosen_df.loc[:, 'nbeats'] == nbeats),
               ['6', '9', '12', '15', '18', '21', '24']].values.flatten()
          mat_EER[j-1, :] = y_EER.values
          mat_age_EER[j-1, :] = y_age_EER
     avg_std_EER[0, :] = mat_EER.mean(axis=0)
     avg_std_EER[1, :] = mat_EER.std(axis=0)
     avg_std_EER[2, :] = avg_std_EER[0, :]/avg_std_EER[1, :]
     avg_std_EER[avg_std_EER == np.inf] = 0
     avg_std_age_EER[0, :] = mat_age_EER.mean(axis=0)
     avg_std_age_EER[1, :] = mat_age_EER.std(axis=0)
     avg_std_age_EER[2, :] = avg_std_age_EER[0, :]/avg_std_age_EER[1, :]
     avg_std_age_EER[avg_std_age_EER == np.inf] = 0
     x_EER = np.unique(chosen_df['nbeats'].loc[(chosen_df['nbeats'] <= 600) & (chosen_df['nbeats'] > 10)])
     barplot(x_EER, avg_std_EER[0, :], avg_std_EER[1, :], mode, state, nbeats, age_flag='nbeats')
     barplot(ages, avg_std_age_EER[0, :], avg_std_age_EER[1, :], mode, state, nbeats, age_flag='age')
     barplot(x_EER, avg_std_EER[2, :], np.zeros_like(avg_std_EER[2, :]), mode, state, nbeats, age_flag='cv')
     if data4volcano == 'nbeats':
          plot_volcano(avg_std_EER, mat_EER, x_EER)
     else:
          plot_volcano(avg_std_age_EER, mat_age_EER, ages)
     a=1
     return mat_EER, mat_age_EER


## up2_21 has EQUAL number of windows for both basal and intrinsic

prep_empty = False
rearrange = 'F'  # todo: Note that changing this overwrites the previous results

p = ProSet()
empty_parser = argparse.ArgumentParser()
parser = init_parser(parent=empty_parser)
p = parser.parse_args()


Chosen_states = ['basal', 'int', 'comb']
# Modes = ['equal', 'bas_on_int', 'up2_21', 'up2_21_bas_on_int']
Modes = ['up2_21', 'up2_21_bas_on_int']  # 'up2_21'
# for j in range(p.bootstrap_total_folds + 1):  # todo: activate this for prep_empty=True for different number of folds
res_dict ={}
for mode in Modes:
     res_dict[mode] = {}
     for chosen_state in Chosen_states:
          if prep_empty:
               prepare_empty_df(j, mode=mode)
               continue
          save_pkl_pth = '/home/smorandv/ac8_and_aging_NEW/ac8_and_aging/rr_datasets'
          orig_pth = save_pkl_pth
          mode2fig = {'equal': '_equal/', 'non_equal': '/', 'bas_on_int': '_equal/bas_on_int/', 'up2_21': '_equal_up2_21/',
                      'up2_21_bas_on_int': '_equal_up2_21/bas_on_int/'}
          # save_pkl_pth += mode2fig[mode] + 'res_df_' + str(j) + '.pkl'
          # with open(save_pkl_pth, 'rb') as f:
          #      dd = pickle.load(f)
          # chosen_df = select_rows(dd, **dict(Rearrange=[rearrange], State=['int', 'basal', 'comb'], Measure=['ERR', 'F1']))
          mat_EER, mat_age_EER = avg_std_bootstrap_df(orig_pth, mode2fig, mode, p, chosen_state, 50)
          # report_statistics(chosen_df)
          # chosen_state = 'comb'
          stats.kruskal(mat_age_EER[:, 0], mat_age_EER[:, 1], mat_age_EER[:, 2], mat_age_EER[:, 3], mat_age_EER[:, 4], mat_age_EER[:, 5])
          stats.f_oneway(mat_age_EER[:, 0], mat_age_EER[:, 1], mat_age_EER[:, 2], mat_age_EER[:, 3], mat_age_EER[:, 4], mat_age_EER[:, 5])
          # res_dict[mode][chosen_state] = [mat_EER, mat_age_EER]
          a=1
          # plot_res(chosen_df, mode, state=chosen_state, nbeats=50)
          # plot_df = select_rows(chosen_df, **dict(Rearrange=[rearrange], State=[chosen_state], Measure=['ERR']))
          # plot_one_win(plot_df, mode, chosen_state)

a=1