# coding: utf-8
import os
import gc
import pandas as pd
from joblib import Parallel, delayed
import argparse

home = os.environ['HOME']

def gen_session_list_dsin(uid, t):
    t.sort_values('time_stamp', inplace=True, ascending=True)
    last_time = 1483574401  # pd.to_datetime("2017-01-05 00:00:01")
    session_list = []
    session = []
    for row in t.iterrows():
        time_stamp = row[1]['time_stamp']
        # pd_time = pd.to_datetime(timestamp_datetime(time_stamp))
        delta = time_stamp - last_time
        cate_id = row[1]['cate']
        brand = row[1]['brand']
        # delta.total_seconds()
        if delta > 30 * 60:  # Session begin when current behavior and the last behavior are separated by more than 30 minutes.
            if len(session) > 2:  # Only use sessions that have >2 behaviors
                session_list.append(session[:])
            session = []

        session.append((cate_id, brand, time_stamp))
        last_time = time_stamp
    if len(session) > 2:
        session_list.append(session[:])
    return uid, session_list


def gen_session_list_din(uid, t):
    t.sort_values('time_stamp', inplace=True, ascending=True)
    session_list = []
    session = []
    for row in t.iterrows():
        time_stamp = row[1]['time_stamp']
        # pd_time = pd.to_datetime(timestamp_datetime())
        # delta = pd_time  - last_time
        cate_id = row[1]['cate']
        brand = row[1]['brand']
        session.append((cate_id, brand, time_stamp))

    if len(session) > 2:
        session_list.append(session[:])
    return uid, session_list


def applyParallel(df_grouped, func, n_jobs, backend='multiprocessing'):
    """Use Parallel and delayed """  # backend='threading'
    results = Parallel(n_jobs=n_jobs, verbose=4, backend=backend)(
        delayed(func)(name, group) for name, group in df_grouped)

    return {k: v for k, v in results}


def gen_user_hist_sessions(model, FRAC=0.25):
    if model not in ['din', 'dsin']:
        raise ValueError('model must be din or dsin')

    print("begin gen " + model + " hist sess", FRAC)
    if model=='din':
        name = SAMP_PATH + '/behavior_log_pv_user_filter_enc_' + str(FRAC) + '.pkl'
    elif model=='dsin':
        name = SAMP_PATH + '/behavior_log_pv_user_filter_enc_btag_' + str(FRAC) + '.pkl'
    data = pd.read_pickle(name)
    data = data.loc[data.time_stamp >= 1493769600]  # 0503-0513
    # 0504~1493856000
    # 0503 1493769600

    user = pd.read_pickle(SAMP_PATH + '/user_profile_' + str(FRAC) + '.pkl')

    n_samples = user.shape[0]
    print(n_samples, 'users')
    batch_size = 150000
    iters = (n_samples - 1) // batch_size + 1

    # print("total", iters, "iters", "batch_size", batch_size)
    for i in range(0, iters):
        target_user = user['userid'].values[i * batch_size:(i + 1) * batch_size]
        sub_data = data.loc[data.user.isin(target_user)]
        df_grouped = sub_data.groupby('user')
        if model == 'din':
            user_hist_session = applyParallel(
                df_grouped, gen_session_list_din, n_jobs=20, backend='loky')
        else:
            user_hist_session = applyParallel(
                df_grouped, gen_session_list_dsin, n_jobs=20, backend='multiprocessing')
        pd.to_pickle(user_hist_session, SAMP_PATH + '/user_hist_session_' +
                     str(FRAC) + '_' + model + '_' + str(i) + '.pkl')
        print(i, 'pickled')
        del user_hist_session
        gc.collect()
    print("1_gen " + model + " hist sess done")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate session.')
    parser.add_argument('--frac', type=float, default=0.001, help='fraction')
    parser.add_argument('--dataset', type=str, default=home+'/datasets/dsin2/', help='dataset path.')
    args = parser.parse_args()
    print(args)

    FRAC = args.frac
    DATASET = args.dataset

    SAMP_PATH = DATASET + '/sampled_data/'
    gen_user_hist_sessions('din', FRAC)
    gen_user_hist_sessions('dsin', FRAC)
