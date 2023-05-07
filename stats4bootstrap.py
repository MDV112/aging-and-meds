import argparse
import pickle

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy import stats

from project_settings import ProSet
from project_settings import init_parser


def select_rows(df, **kwargs):  # values have to be list Rearrange=0,State=0,Measure=0,nbeats=0,age=0
    chosen_df = df
    for key in kwargs.keys():
        if key != 0:
            chosen_df = chosen_df.loc[chosen_df.loc[:, key].isin(kwargs[key])]
    return chosen_df


def calc(avg_std_EER, avg_std_age_EER, res_dict, mode, state):
    EER = avg_std_EER[0, :]
    rel_EER = EER[EER < 0.4]
    #  todo: Notice the following line comment. It may not be appropriate to all situations
    # rel_EER = EER[avg_std_EER[1, :] > 0]  # only the measurements that has bootstrap std larger than 0 count
    res_dict[mode + '_' + state]['std_EER'] = np.std(rel_EER)
    res_dict[mode + '_' + state]['std_age'] = np.std(avg_std_age_EER[0, :])
    res_dict[mode + '_' + state]['std_min_EER'] = np.sqrt(
        (1 / len(rel_EER)) * (np.linalg.norm(rel_EER - rel_EER.min()) ** 2))
    res_dict[mode + '_' + state]['std_min_age'] = np.sqrt(
        (1 / (avg_std_age_EER.shape[1])) * (np.linalg.norm(avg_std_age_EER[0, :] - avg_std_age_EER[0, :].min()) ** 2))
    res_dict[mode + '_' + state]['CV_EER'] = avg_std_EER[2, :]
    res_dict[mode + '_' + state]['CV_age'] = avg_std_age_EER[2, :]
    return res_dict


def avg_std_bootstrap_df(save_pkl_pth, res_dict, mode, p, state, nbeats, data4volcano='nbeats'):
    ages = [6, 9, 12, 15, 18, 21, 24]
    mat_EER = np.zeros((p.bootstrap_total_folds, 12))  # 12 is the number of beats which is > 25, != 75 and <=600
    mat_age_EER = np.zeros((p.bootstrap_total_folds, len(ages)))
    avg_std_EER = np.zeros((3, mat_EER.shape[1]))
    avg_std_age_EER = np.zeros((3, mat_age_EER.shape[1]))
    for j in range(1, p.bootstrap_total_folds + 1):
        pth = save_pkl_pth + 'res_df_' + str(j) + '.pkl'
        with open(pth, 'rb') as f:
            dd = pickle.load(f)
        chosen_df = select_rows(dd,
                                **dict(Rearrange=[rearrange], State=['int', 'basal', 'comb'], Measure=['ERR', 'F1']))
        y_EER = chosen_df.loc[(chosen_df.loc[:, 'Measure'] == 'ERR') & (chosen_df.loc[:, 'State'] == state) &
                              (chosen_df.loc[:, 'nbeats'] <= 600) & (chosen_df.loc[:, 'nbeats'] > 25) & (
                                          chosen_df.loc[:, 'nbeats'] != 75), '6']
        y_age_EER = chosen_df.loc[
            (chosen_df.loc[:, 'Measure'] == 'ERR') & (chosen_df.loc[:, 'State'] == state) & (
                        chosen_df.loc[:, 'nbeats'] == nbeats),
            ['6', '9', '12', '15', '18', '21', '24']].values.flatten()
        mat_EER[j - 1, :] = y_EER.values
        mat_age_EER[j - 1, :] = y_age_EER
    avg_std_EER[0, :] = mat_EER.mean(axis=0)
    avg_std_EER[1, :] = mat_EER.std(axis=0)
    avg_std_EER[2, :] = avg_std_EER[1, :] / avg_std_EER[0, :]
    avg_std_age_EER[0, :] = mat_age_EER.mean(axis=0)
    avg_std_age_EER[1, :] = mat_age_EER.std(axis=0)
    avg_std_age_EER[2, :] = avg_std_age_EER[1, :] / avg_std_age_EER[0, :]
    x_EER = np.unique(chosen_df['nbeats'].loc[(chosen_df['nbeats'] <= 600) & (chosen_df['nbeats'] > 25) & (
                chosen_df.loc[:, 'nbeats'] != 75)])
    barplot(x_EER, avg_std_EER[0, :], avg_std_EER[1, :], mode, state, nbeats, age_flag='nbeats')
    barplot(ages, avg_std_age_EER[0, :], avg_std_age_EER[1, :], mode, state, nbeats, age_flag='age')
    barplot(x_EER, avg_std_EER[2, :], np.zeros_like(avg_std_EER[2, :]), mode, state, nbeats, age_flag='cv')
    # if data4volcano == 'nbeats':
    #      plot_volcano(avg_std_EER, mat_EER, x_EER)
    # else:
    #      plot_volcano(avg_std_age_EER, mat_age_EER, ages)
    # a=1
    res_dict[mode + '_' + state]['mat_EER'] = mat_EER  # for statistical tests
    res_dict[mode + '_' + state]['mat_age_EER'] = mat_age_EER  # for statistical tests
    res_dict = calc(avg_std_EER, avg_std_age_EER, res_dict, mode, state)
    return res_dict


def set_res_dict(p, rearrange, orig_pth):
    Chosen_states = ['basal', 'int', 'comb']
    # Modes = ['equal', 'bas_on_int', 'up2_21', 'up2_21_bas_on_int']
    Modes = ['up2_21', 'up2_21_bas_on_int']  # 'up2_21'
    res_dict = {}
    for mode in Modes:
        for chosen_state in Chosen_states:
            if mode == 'up2_21':
                pth = orig_pth
            else:
                pth = orig_pth + 'bas_on_int/'
            res_dict[mode + '_' + chosen_state] = {}
            res_dict = avg_std_bootstrap_df(pth, res_dict, mode, p, chosen_state, 50)
            # stats.kruskal(mat_age_EER[:, 0], mat_age_EER[:, 1], mat_age_EER[:, 2], mat_age_EER[:, 3], mat_age_EER[:, 4], mat_age_EER[:, 5])
            # stats.f_oneway(mat_age_EER[:, 0], mat_age_EER[:, 1], mat_age_EER[:, 2], mat_age_EER[:, 3], mat_age_EER[:, 4], mat_age_EER[:, 5])

    with open(orig_pth + rearrange + '_res_dict.pkl', 'wb') as f:
        pickle.dump(res_dict, f)


def stat_res_dict(CD_res_dict_pth, PD_res_dict_pth):
    with open(CD_res_dict_pth, 'rb') as f:
        CD = pickle.load(f)
    with open(PD_res_dict_pth, 'rb') as f:
        PD = pickle.load(f)
    plt.scatter(np.arange(len(CD['up2_21_basal']['CV_EER'])),
                CD['up2_21_basal']['CV_EER'] / PD['up2_21_basal']['CV_EER'])
    min_idx_CD = CD['up2_21_basal']['mat_EER'].mean(axis=0).argmin()
    min_idx_PD = PD['up2_21_basal']['mat_EER'].mean(axis=0).argmin()
    # plt.show()
    print(CD['up2_21_basal']['std_EER'] < PD['up2_21_basal']['std_EER'])
    print(CD['up2_21_basal']['std_age'] < PD['up2_21_basal']['std_age'])
    print(CD['up2_21_basal']['std_min_EER'] < PD['up2_21_basal']['std_min_EER'])
    print(CD['up2_21_basal']['std_min_age'] < PD['up2_21_basal']['std_min_age'])

    state = 'up2_21_bas_on_int_int'
    CD_6 = CD[state]['mat_age_EER'][:, 0]
    rep_6_CD = np.repeat(np.expand_dims(CD_6, axis=1), CD[state]['mat_age_EER'].shape[1], axis=1)
    equal_ages_CD = test_equivalence(rep_6_CD, CD, eq_margin=3 * rep_6_CD.std(axis=0)[0], state=state, alph=0.05)

    PD_6 = PD[state]['mat_age_EER'][:, 0]
    rep_6_PD = np.repeat(np.expand_dims(PD_6, axis=1), PD[state]['mat_age_EER'].shape[1], axis=1)
    equal_ages_PD = test_equivalence(rep_6_PD, PD, eq_margin=3 * rep_6_PD.std(axis=0)[0], state=state)
    a = 1


def calc_eq_margin(rep_6, CD_PD_dict):
    # two_std = 0.5*(rep_6.std(axis=0) + CD_PD_dict['up2_21_basal']['mat_age_EER'].std(axis=0))
    cohens_d = rep_6.mean(axis=0) - CD_PD_dict['up2_21_basal']['mat_age_EER'].mean(axis=0)
    n1 = rep_6.shape[0]
    n2 = rep_6.shape[0]
    sigma = np.sqrt(((n1 - 1) * rep_6.var(axis=0) + (n2 - 1) * (CD_PD_dict['up2_21_basal']['mat_age_EER'].var(axis=0)))
                    / (n1 + n2 - 2))
    eq_margin = sigma * cohens_d
    return eq_margin


def test_equivalence(rep_6, CD_PD_dict, alph=0.05, eq_margin='auto', state='up2_21_basal'):
    ages = np.arange(6, 27, 3)
    if eq_margin == 'auto':
        equal_ages = []
        eq_margin = calc_eq_margin(rep_6, CD_PD_dict)
        for i, age in enumerate(list(ages)):
            t1, p1 = stats.mannwhitneyu(rep_6[:, i], CD_PD_dict[state]['mat_age_EER'][:, i] + eq_margin[i],
                                        alternative='less')
            t2, p2 = stats.mannwhitneyu(rep_6[:, i], CD_PD_dict[state]['mat_age_EER'][:, i] - eq_margin[i],
                                        alternative='greater')
            if (p1 < alph / 2) & (p2 < alph / 2):
                equal_ages.append(age)
    else:
        t1, p1 = stats.ttest_ind(rep_6, CD_PD_dict[state]['mat_age_EER'] + eq_margin, axis=0, alternative='less',
                                 equal_var=False)
        t2, p2 = stats.ttest_ind(rep_6, CD_PD_dict[state]['mat_age_EER'] - eq_margin, axis=0, alternative='greater',
                                 equal_var=False)
        eq = (p1 < alph / 2) & (p2 < alph / 2)
        if np.argwhere(eq).shape[0] == np.argwhere(eq).shape[1]:
            equal_ages = ages[np.argwhere(eq).item()]
        else:
            equal_ages = ages[np.argwhere(eq).squeeze()]
    return equal_ages
    a = 1


def barplot(x, y, yerr, mode, state, nbeats, age_flag='nbeats', **extra_kwargs):
    fig, ax = plt.subplots(figsize=(20, 20))
    ax.bar(np.arange(len(y)), y, yerr=yerr, color='red')
    ax.set_xticks(np.arange(len(y)))  # make xticks to be with equal intervals (note that x in the bar is the same)
    ax.set_xticklabels(x, fontsize=38, weight='bold')  # rewrites the labels of ticks
    plt.yticks(fontsize=38, weight='bold')
    # x_dashed = np.arange(len(y) + 1)
    # sns.lineplot(x=x_dashed, y=np.ones_like(x_dashed) * (y.min()), linestyle="dashed", linewidth=6, color='g')
    # ax.text(len(x_dashed) - 1, y.min(), '{:.2f}'.format(y.min()), fontsize=30, weight='bold')
    if age_flag == 'age':
        ax.set_xlabel(xlabel="Age [m]", fontsize=50, weight='bold')
    elif (age_flag == 'nbeats') or (age_flag == 'cv'):
        ax.set_xlabel(xlabel="nbeats", fontsize=50, weight='bold')
    ax.set_ylabel(ylabel="EER", fontsize=50, weight='bold')
    if not (age_flag == 'cv'):
        ax.set_ylim(0, 0.5)

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
    elif age_flag == 'cv':
        plt.savefig(save_fig_pth + 'nbeats_cv_' + state + '.png')
    a = 1


rearrange = 'F'  # todo: Note that changing this overwrites the previous results
orig_pth = '/home/smorandv/ac8_and_aging_NEW/ac8_and_aging/rr_datasets_equal_up2_21/bootstrap_without_stratification/'
p = ProSet()
empty_parser = argparse.ArgumentParser()
parser = init_parser(parent=empty_parser)
p = parser.parse_args()
set_res_dict(p, rearrange, orig_pth)
CD_res_dict_pth = orig_pth + 'T_res_dict.pkl'
PD_res_dict_pth = orig_pth + 'F_res_dict.pkl'
stat_res_dict(CD_res_dict_pth, PD_res_dict_pth)
a = 1
