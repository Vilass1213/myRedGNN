import os
import argparse
import torch
import numpy as np
from load_data import DataLoader
from base_model import BaseModel


parser = argparse.ArgumentParser(description="Parser for RED-GNN")
parser.add_argument('--data_path', type=str, default='data/onlydrds/')
parser.add_argument('--seed', type=str, default=1234)


args = parser.parse_args()

class Options(object):
    pass

if __name__ == '__main__':
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    dataset = args.data_path
    dataset = dataset.split('/')
    if len(dataset[-1]) > 0:
        dataset = dataset[-1]
    else:
        dataset = dataset[-2]
   
    results_dir = 'results'
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    opts = Options
    opts.perf_file = os.path.join(results_dir,  dataset + '_perf.txt')

    # gpu = select_gpu()
    torch.cuda.set_device("cuda:0")
    # print('gpu:', gpu)

    # loader = DataLoader(args.data_path, embedding_file=None)
    loader = DataLoader(args.data_path, embedding_file='transe_drds_embeddings.txt')  #use pretrained embeddings
    opts.n_ent = loader.n_ent
    opts.n_rel = loader.n_rel


    opts.lr = 0.0036
    opts.decay_rate = 0.999
    opts.lamb = 0.000017
    opts.hidden_dim = 48
    opts.attn_dim = 5
    opts.n_layer = 6
    opts.dropout = 0.29
    opts.act = 'relu'
    opts.n_batch = 60
    opts.n_tbatch = 50


    config_str = '%.4f, %.4f, %.6f,  %d, %d, %d, %d, %.4f,%s\n' % (opts.lr, opts.decay_rate, opts.lamb, opts.hidden_dim, opts.attn_dim, opts.n_layer, opts.n_batch, opts.dropout, opts.act)
    print(config_str)
    with open(opts.perf_file, 'a+') as f:
        f.write(config_str)

    model = BaseModel(opts, loader)

    # best_mrr = 0
    best_mrr_per_relation = {'indication': 0, 'contraindication': 0}
    best_str_per_relation = {'indication': '', 'contraindication': ''}

    for epoch in range(50):
        print('Epoch:', epoch)
        mrr_per_relation, out_str = model.train_batch()
        with open(opts.perf_file, 'a+') as f:
            f.write(out_str)

        # if mrr > best_mrr:
        #     best_mrr = mrr
        #     best_str = out_str
        #     print(str(epoch) + '\t' + best_str)
        for rel, mrr in mrr_per_relation.items():
            if mrr > best_mrr_per_relation[rel]:
                best_mrr_per_relation[rel] = mrr
                best_str_per_relation[rel] = out_str
                print(f'{epoch}\t[Best {rel}] ' + best_str_per_relation[rel])  # best metrics for each relation
    # print(best_str)
    print("Final Best Results:")
    for rel, best_mrr in best_mrr_per_relation.items():
        print(f'Best MRR for {rel}: {best_mrr:.4f}')
        print(f'Best metrics for {rel}: {best_str_per_relation[rel]}')

