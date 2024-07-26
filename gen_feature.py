# %%
from torch import nn
from torch_geometric.nn import GCNConv
from params import args
from utils.dataprocessing import gen_dataset, make_adj
import torch
import numpy as np

#%%
torch.backends.cudnn.enabled = False

#%%
class model_feature(nn.Module):
    def __init__(self, args):
        super(model_feature, self).__init__()
        self.gcn1_mi_func = GCNConv(args.em_mi, args.em_mi)
        self.gcn1_mi_gip = GCNConv(args.em_mi, args.em_mi)
        self.gcn2_mi_func = GCNConv(args.em_mi, args.em_mi)
        self.gcn2_mi_gip = GCNConv(args.em_mi, args.em_mi)

        self.gcn1_dis_sem = GCNConv(args.em_dis, args.em_dis)
        self.gcn1_dis_gip = GCNConv(args.em_dis, args.em_dis)
        self.gcn2_dis_sem = GCNConv(args.em_dis, args.em_dis)
        self.gcn2_dis_gip = GCNConv(args.em_dis, args.em_dis)

        self.linear_mi_func = nn.Linear(args.em_mi, args.out_mi_dim) #Q fai khop 256 in va out
        self.linear_mi_gip = nn.Linear(args.em_mi, args.out_mi_dim)

        self.linear_dis_sem = nn.Linear(args.em_dis, args.out_dis_dim)
        self.linear_dis_gip = nn.Linear(args.em_dis, args.out_dis_dim)

    def forward(self, data):
        torch.manual_seed(123)
        mm0 = torch.randn((args.mi_num, args.em_mi))
        dd0 = torch.randn((args.dis_num, args.em_dis))

        mm_f1 = torch.relu(self.gcn1_mi_func(mm0, data['mm_func']['edges'], data['mm_func']['data_matrix'][
            data['mm_func']['edges'][0], data['mm_func']['edges'][1]]))
        mm_f2 = torch.relu(self.gcn2_mi_func(mm_f1, data['mm_func']['edges'], data['mm_func']['data_matrix'][
            data['mm_func']['edges'][0], data['mm_func']['edges'][1]]))

        mm_g1 = torch.relu(self.gcn1_mi_gip(mm0, data['mm_gip']['edges'], data['mm_gip']['data_matrix'][
            data['mm_gip']['edges'][0], data['mm_gip']['edges'][1]]))
        mm_g2 = torch.relu(self.gcn2_mi_gip(mm_g1, data['mm_gip']['edges'], data['mm_gip']['data_matrix'][
            data['mm_gip']['edges'][0], data['mm_gip']['edges'][1]]))

        dd_s1 = torch.relu(self.gcn1_dis_sem(dd0, data['dd_sema']['edges'], data['dd_sema']['data_matrix'][
            data['dd_sema']['edges'][0], data['dd_sema']['edges'][1]]))
        dd_s2 = torch.relu(self.gcn2_dis_sem(dd_s1, data['dd_sema']['edges'], data['dd_sema']['data_matrix'][
            data['dd_sema']['edges'][0], data['dd_sema']['edges'][1]]))

        dd_g1 = torch.relu(self.gcn1_dis_gip(dd0, data['dd_gip']['edges'], data['dd_gip']['data_matrix'][
            data['dd_gip']['edges'][0], data['dd_gip']['edges'][1]]))
        dd_g2 = torch.relu(self.gcn2_dis_gip(dd_g1, data['dd_gip']['edges'], data['dd_gip']['data_matrix'][
            data['dd_gip']['edges'][0], data['dd_gip']['edges'][1]]))

        mmf = self.linear_mi_func(mm_f2)
        mmg = self.linear_mi_gip(mm_g2)
        dds = self.linear_mi_func(dd_s2)
        ddg = self.linear_mi_gip(dd_g2)
        MW = torch.tensor(mmf > 0, dtype=torch.int)
        DW = torch.tensor(dds > 0, dtype=torch.int)
        mi_fea = MW * mmf + (1 - MW) * mmg
        dis_fea = DW * dds + (1 - DW) * ddg
        return mi_fea.mm(dis_fea.t()), mi_fea, dis_fea

# %%
def train(model, train_data, optimizer, opt):
    model.train()
    for epoch in range(0, opt.ne_feature):
        model.zero_grad()
        score, mi_fea, dis_fea = model(train_data)
        print('epoch ', epoch)
        loss = torch.nn.MSELoss(reduction='mean')
        loss = loss(score, train_data['md_p'])
        loss.backward()
        optimizer.step()
        # print(loss.item())
    score = score.detach().cpu().numpy()
    scoremin, scoremax = score.min(), score.max()
    score = (score - scoremin) / (scoremax - scoremin)
    return score, mi_fea, dis_fea

# %%
def  gen_feature(train_pair_list, test_pair_list, train_adj, ix, loop_i):
    dataset = gen_dataset(args, train_adj, ix, loop_i)
    train_data = dataset

    ###---or (1,2)------------
    model = model_feature(args)
    optimizer = torch.optim.Adam(model.parameters(), lr = 0.001)
    score, mi_em, dis_em = train(model, train_data, optimizer, args)

    yprob_ori = score[test_pair_list[:, 0], test_pair_list[:, 1]]
    np.savetxt(args.fi_out + 'L' + str(loop_i) + '_yprob_ori' + str(ix) + '.csv', yprob_ori)

    np.savetxt(args.fi_out + 'L' + str(loop_i) + '_mi_em' + str(ix) + '.csv', mi_em.detach().numpy(), delimiter=',')
    np.savetxt(args.fi_out + 'L' + str(loop_i) + '_dis_em' + str(ix) + '.csv', dis_em.detach().numpy(), delimiter=',')
    # #---
    # mi_em = np.genfromtxt(args.fi_out + 'L' + str(loop_i) + '_mi_em' + str(ix) + '.csv', delimiter=',')
    # dis_em = np.genfromtxt(args.fi_out + 'L' + str(loop_i) + '_dis_em' + str(ix) + '.csv', delimiter=',')
    ###----------------------
    feature_tr = np.hstack((mi_em[train_pair_list[:, 0]].tolist(), dis_em[train_pair_list[:, 1]].tolist()))
    feature_te = np.hstack((mi_em[test_pair_list[:, 0]].tolist(), dis_em[test_pair_list[:, 1]].tolist()))

    # np.savetxt(args.fi_out + 'L' + str(loop_i) + '_fea_train' + str(ix) + '.csv', feature_tr, delimiter = ',')
    # np.savetxt(args.fi_out + 'L' + str(loop_i) + '_fea_test' + str(ix) + '.csv', feature_te, delimiter = ',')

    return feature_tr, feature_te
