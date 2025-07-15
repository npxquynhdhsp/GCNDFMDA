# %%
from lib.gcforest.gcforest import GCForest
from params import args
from gen_feature import gen_feature
from utils.dataprocessing import *
from utils.dataprocessing_CB import split_kfold_CBTr
from utils.dataprocessing2 import *
from utils.utils import *
import numpy as np
import time

# %%
def get_config():
    config = {}
    ca_config = {}
    ca_config["random_state"] = 0
    ca_config["early_stopping_rounds"] = 1
    ca_config["n_classes"] = 2
    ca_config["estimators"] = []
    ca_config["max_layers"] = 5 #Q
    nfold = args.df_nfold

    ca_config["estimators"].append(
        {"n_folds": nfold,
         "type": "LinearRegression"})

    ca_config["estimators"].append(
        {"n_folds": nfold,
         "type": "LinearRegression"})

    ca_config["estimators"].append(
        {"type": "XGBClassifier",
         "n_folds": nfold,
         "n_estimators": args.xg_ne,
         "max_depth": 5,
         "silent": True,
         "nthread": -1,
         "learning_rate": args.xg_lrr})

    ca_config["estimators"].append(
        {"type": "XGBClassifier",
         "n_folds": nfold,
         "n_estimators": args.xg_ne,
         "max_depth": 5,
         "silent": True,
         "nthread": -1,
         "learning_rate": args.xg_lrr})

    config["cascade"] = ca_config
    return config


def models_eval(method_set_name, X_train_enc, X_test_enc, y_train, y_test, ix, loop_i):
    if method_set_name == 'RF':
        print('Random Forest')
        from sklearn.ensemble import RandomForestClassifier
        clf = RandomForestClassifier(n_estimators=args.rf_ne, max_depth=None, n_jobs=-1)
    elif method_set_name == 'ETR':
        print('Extra trees regression')
        from sklearn.ensemble import ExtraTreesRegressor
        clf = ExtraTreesRegressor(n_estimators=args.etr_ne, n_jobs=-1)
    elif method_set_name == 'LR':
        print('Linear regression')
        from sklearn.linear_model import LinearRegression
        clf = LinearRegression()
    else:
        print('XGBoost')
        from xgboost import XGBClassifier
        clf = XGBClassifier(booster='gbtree', n_jobs=2, learning_rate=args.xg_lrr, n_estimators=args.xg_ne)

    clf.fit(X_train_enc, y_train)

    if (method_set_name == 'ETR') or (method_set_name == 'LR'):
        y_prob = clf.predict(X_test_enc)
    else:
        y_prob = clf.predict_proba(X_test_enc)[:,1]

    np.savetxt(args.fi_out + 'L' + str(loop_i) + '_yprob_' + method_set_name.lower() + str(ix) + '.csv', y_prob)
    calculate_score([y_test], [y_prob])
    return y_prob


def save_eval(method_set_name, true_set, prob_set):
    print(method_set_name,':')
    calculate_score(true_set, prob_set) # cal mean
    prob_set_join = np.concatenate(prob_set, axis = 0) # join
    np.savetxt(args.fi_out + 'prob_set_' + method_set_name.lower() + '.csv', prob_set_join)

    if method_set_name == 'DF':
        true_set_join = np.concatenate(true_set, axis = 0)
        np.savetxt(args.fi_out + 'true_set.csv', true_set_join, fmt='%d')

        #QX
    return

def save_eval2_all(method_set_name, true_set, prob_set):
    print(method_set_name,':')
    prob_set_join = np.concatenate(prob_set, axis = 0) # join
    np.savetxt(args.fi_out + 'prob_set' + method_set_name.lower() + '.csv', prob_set_join)

    true_set_join = np.concatenate(true_set, axis = 0)
    np.savetxt(args.fi_out + 'true_set.csv', true_set_join, fmt='%d')
    calculate_score([true_set_join], [prob_set_join])
    
def main():
    #QX

    bg_time = time.time()
    config = get_config()

    if args.db != 'INDE_TEST':
        print(args.db)
        if args.type_eval == 'KFOLD':  # KFOLD/DIS_K/DENO_MI
            set_ix = np.arange(args.bgf, args.nfold + 1) #Q khac cac bai khac
            temp = 'FOLD '
            print('read_tr_te_adj', args.read_tr_te_adj)
        elif args.type_eval == 'DIS_K':
            # set_ix = args.dis_set
            set_ix = np.genfromtxt(args.fi_A + 'dis_set.csv').astype(int).T
            temp = 'DIS '
        else:
            mi_set = np.genfromtxt(args.fi_A + 'mi_setT.csv').astype(int).T[40:]#Q
            set_ix = mi_set
            temp = 'MIRNA '
    else:
        print('INDEPENDENT TEST')
        set_ix = [1]

    prob_set_df, prob_set_rf, prob_set_etr, prob_set_lr, prob_set_xg = [], [], [], [], []
    true_set = []

    for loop_i in range(args.bgl, args.nloop + 1):
        print('.......................................... LOOP ', loop_i,'.........................................')
        if (args.db != 'INDE_TEST') and (args.type_eval == 'KFOLD'):
            idx_pair_train_set, idx_pair_test_set, y_train_set, y_test_set, train_adj_set = \
            split_kfold_CBTr(args.fi_A, args.fi_proc, 'adj_MD.csv', \
                    '_CB', args.type_test,1,  loop_i)
        print('.')

        for ix in set_ix:
            ###-----------------------
            if (args.db == 'INDE_TEST'):
                idx_pair_train, idx_pair_test, y_train, y_test, train_adj = \
                    split_tr_te_adj(args.type_eval, args.fi_A, args.fi_proc, 'adj_4loai.csv', \
                                    '_CB', args.type_test, 1, ix, loop_i)
            else:
                if (args.type_eval == 'KFOLD'):
                    if args.read_tr_te_adj == 1:
                        idx_pair_train, idx_pair_test, y_train, y_test, train_adj = \
                            read_train_test_adj(args.fi_proc, '/md_p', \
                                                '_CB', args.type_test, ix, loop_i)
                    else:
                        idx_pair_train, idx_pair_test, y_train, y_test, train_adj = \
                            idx_pair_train_set[ix-1], idx_pair_test_set[ix-1], y_train_set[ix-1], \
                                y_test_set[ix-1], train_adj_set[ix-1]
                else:
                    idx_pair_train, idx_pair_test, y_train, y_test, train_adj = \
                        split_tr_te_adj(args.type_eval, args.fi_A, args.fi_proc, 'adj_MD.csv', \
                                        '_CB', args.type_test, 1, ix, loop_i)

            np.savetxt(args.fi_out + 'L' + str(loop_i) + '_ytrue' + \
                       str(ix) + '.csv', y_test, fmt='%d')
            ###-----------------------
            if args.db != 'INDE_TEST':
                print(temp, ix, '*' * 50)
            X_train, X_test = gen_feature(idx_pair_train, idx_pair_test, train_adj, ix, loop_i)

            true_set.append(y_test)

            X_train = X_train[:, np.newaxis, :]  # @QS
            X_test = X_test[:, np.newaxis, :]  # @QS

            gc = GCForest(config) #QT
            X_train_enc = gc.fit_transform(X_train, y_train)  # Qcmt k duoc xoa
            print(X_test.shape)  # ex (37917,1,256*2)
            # print('X_train_enc_df.shape', X_train_enc.shape)  # ex (151668, 2*|classifier|)

            # (0)
            print('DF')
            y_prob_df = gc.predict_proba(X_test)[:, 1]
            np.savetxt(args.fi_out + 'L' + str(loop_i) + '_yprob_df' + str(ix) + '.csv', y_prob_df)
            calculate_score([y_test], [y_prob_df])
            prob_set_df.append(y_prob_df)
            print("Finish! ", time.time() - bg_time, "s")
            # QX

        print('--------------------------------FINAL:-------------------------------')
        if (args.db != 'INDE_TEST') or (args.type_eval == 'KFOLD'):
            save_eval2_all('df', true_set, prob_set_df)
        else:
            save_eval('df', true_set, prob_set_df)
        
# %%
if __name__ == "__main__":
    print('fi_ori_feature', args.fi_ori_feature)
    main()



