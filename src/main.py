import argparse
import itertools
import os
import sys
import time
import pickle
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
sys.path.append("..")
import dgl
import numpy as np
import torch
from tqdm import tqdm
import random
import gc
from rgcn import utils
from rgcn.utils import build_sub_graph
from src.rrgcn import RecurrentRGCN
from src.hyperparameter_range import *
import torch.nn.modules.rnn
from collections import defaultdict
from rgcn.knowledge_graph import _read_triplets_as_list
import scipy.sparse as sp
import con_models.data.data as con_data
import con_models.data.config as con_cfg
import con_models.interactive.functions as con_interactive 
from transformers import LongformerModel, AutoTokenizer

import subprocess

def test(model, history_list, test_list, num_rels, num_nodes, use_cuda, all_ans_list, all_ans_r_list, model_name, static_graph, time_list, history_time_nogt, mode):
    ranks_raw, ranks_filter, mrr_raw_list, mrr_filter_list = [], [], [], []
    ranks_raw_r, ranks_filter_r, mrr_raw_list_r, mrr_filter_list_r = [], [], [], []

    idx = 0
    if mode == "test":
        # test mode: load parameter form file
        if use_cuda:
            checkpoint = torch.load(model_name, map_location=torch.device(args.gpu))
        else:
            checkpoint = torch.load(model_name, map_location=torch.device('cpu'))
        print("Load Model name: {}. Using best epoch : {}".format(model_name, checkpoint['epoch']))  # use best stat checkpoint
        print("\n"+"-"*10+"start testing"+"-"*10+"\n")
        model.load_state_dict(checkpoint['state_dict'])

    model.eval()
    # do not have inverse relation in test input
    input_list = [snap for snap in history_list[-args.test_history_len:]]

    if args.multi_step:
        all_tail_seq = sp.load_npz(
            '../data/{}/history/tail_history_{}.npz'.format(args.dataset, history_time_nogt))
        # rel
        all_rel_seq = sp.load_npz(
            '../data/{}/history/rel_history_{}.npz'.format(args.dataset, history_time_nogt))

    for time_idx, test_snap in enumerate(tqdm(test_list)):
        history_glist = [build_sub_graph(num_nodes, num_rels, g, use_cuda, args.gpu) for g in input_list]
        test_triples_input = torch.LongTensor(test_snap).cuda() if use_cuda else torch.LongTensor(test_snap)
        test_triples_input = test_triples_input.to(args.gpu)
        knowledge_list = get_knowledge(args)
        knowledgeseq = rec_knowledge(args,knowledge_list)
        # get history
        histroy_data = test_triples_input
        inverse_histroy_data = histroy_data[:, [2, 1, 0, 3]]
        inverse_histroy_data[:, 1] = inverse_histroy_data[:, 1] + num_rels
        histroy_data = torch.cat([histroy_data, inverse_histroy_data])
        histroy_data = histroy_data.cpu().numpy()
        if args.multi_step:
            seq_idx = histroy_data[:, 0] * num_rels * 2 + histroy_data[:, 1]
            tail_seq = torch.Tensor(all_tail_seq[seq_idx].todense())
            one_hot_tail_seq = tail_seq.masked_fill(tail_seq != 0, 1)
            # rel
            rel_seq_idx = histroy_data[:, 0] * num_nodes + histroy_data[:, 2]
            rel_seq = torch.Tensor(all_rel_seq[rel_seq_idx].todense())
            one_hot_rel_seq = rel_seq.masked_fill(rel_seq != 0, 1)
        else:
            all_tail_seq = sp.load_npz(
                '../data/{}/history/tail_history_{}.npz'.format(args.dataset, time_list[time_idx]))
            seq_idx = histroy_data[:, 0] * num_rels * 2 + histroy_data[:, 1]
            tail_seq = torch.Tensor(all_tail_seq[seq_idx].todense())
            one_hot_tail_seq = tail_seq.masked_fill(tail_seq != 0, 1)
            # rel
            all_rel_seq = sp.load_npz(
                '../data/{}/history/rel_history_{}.npz'.format(args.dataset, time_list[time_idx]))
            rel_seq_idx = histroy_data[:, 0] * num_nodes + histroy_data[:, 2]
            rel_seq = torch.Tensor(all_rel_seq[rel_seq_idx].todense())
            one_hot_rel_seq = rel_seq.masked_fill(rel_seq != 0, 1)
        if use_cuda:
            one_hot_tail_seq = one_hot_tail_seq.cuda()
            one_hot_rel_seq = one_hot_rel_seq.cuda()

        test_triples, final_score, final_r_score = model.predict(history_glist, num_rels, static_graph, test_triples_input, one_hot_tail_seq, one_hot_rel_seq, use_cuda,knowledgeseq)

        mrr_filter_snap_r, mrr_snap_r, rank_raw_r, rank_filter_r = utils.get_total_rank(test_triples, final_r_score, all_ans_r_list[time_idx], eval_bz=1000, rel_predict=1)
        mrr_filter_snap, mrr_snap, rank_raw, rank_filter = utils.get_total_rank(test_triples, final_score, all_ans_list[time_idx], eval_bz=1000, rel_predict=0)

        # used to global statistic
        ranks_raw.append(rank_raw)
        ranks_filter.append(rank_filter)
        # used to show slide results
        mrr_raw_list.append(mrr_snap)
        mrr_filter_list.append(mrr_filter_snap)

        # relation rank
        ranks_raw_r.append(rank_raw_r)
        ranks_filter_r.append(rank_filter_r)
        mrr_raw_list_r.append(mrr_snap_r)
        mrr_filter_list_r.append(mrr_filter_snap_r)

        # reconstruct history graph list
        if args.multi_step:
            if not args.relation_evaluation:    
                predicted_snap = utils.construct_snap(test_triples, num_nodes, num_rels, final_score, args.topk)
            else:
                predicted_snap = utils.construct_snap_r(test_triples, num_nodes, num_rels, final_r_score, args.topk)
            if len(predicted_snap):
                input_list.pop(0)
                input_list.append(predicted_snap)
        else:
            input_list.pop(0)
            input_list.append(test_snap)
        idx += 1
    
    mrr_raw, hit_result_raw = utils.stat_ranks(ranks_raw, "raw_ent")
    mrr_filter, hit_result_filter = utils.stat_ranks(ranks_filter, "filter_ent")
    mrr_raw_r, hit_result_raw_r = utils.stat_ranks(ranks_raw_r, "raw_rel")
    mrr_filter_r, hit_result_filter_r = utils.stat_ranks(ranks_filter_r, "filter_rel")
    return mrr_raw, mrr_filter, mrr_raw_r, mrr_filter_r, hit_result_raw, hit_result_filter, hit_result_raw_r, hit_result_filter_r


def run_experiment(args, history_len=None, n_layers=None, dropout=None, n_bases=None, angle=None, history_rate=None):
    # load configuration for grid search the best configuration
    if history_len:
        args.train_history_len = history_len
        args.test_history_len = history_len
    if n_layers:
        args.n_layers = n_layers
    if dropout:
        args.dropout = dropout
    if n_bases:
        args.n_bases = n_bases
    if angle:
        args.angle = angle
    if history_rate:
        args.history_rate = history_rate
    mrr_raw = None
    mrr_filter = None
    mrr_raw_r = None
    mrr_filter_r = None
    hit_result_raw = None
    hit_result_filter = None
    hit_result_raw_r = None
    hit_result_filter_r = None

    # load graph data
    print("loading graph data")
    data = utils.load_data(args.dataset)   # 得到data类
    train_list, train_times = utils.split_by_time(data.train)   # 划分为snapshots，逐时间步的数据集
    valid_list, valid_times = utils.split_by_time(data.valid)
    test_list, test_times = utils.split_by_time(data.test)

    num_nodes = data.num_nodes
    num_rels = data.num_rels
    if args.dataset == "ICEWS14s":
        num_times = len(train_list) + len(valid_list) + len(test_list) + 1
    else:
        num_times = len(train_list) + len(valid_list) + len(test_list)
    time_interval = train_times[1]-train_times[0]
    print("num_times", num_times, "--------------", time_interval)
    history_val_time_nogt = valid_times[0]
    history_test_time_nogt = test_times[0]
    if args.multi_step:
        print("val only use global history before:", history_val_time_nogt)
        print("test only use global history before:", history_test_time_nogt)

    all_ans_list_test = utils.load_all_answers_for_time_filter(data.test, num_rels, num_nodes, False)
    all_ans_list_r_test = utils.load_all_answers_for_time_filter(data.test, num_rels, num_nodes, True)
    all_ans_list_valid = utils.load_all_answers_for_time_filter(data.valid, num_rels, num_nodes, False)
    all_ans_list_r_valid = utils.load_all_answers_for_time_filter(data.valid, num_rels, num_nodes, True)

    model_name = "gl_rate_{}-{}-{}-{}-ly{}-dilate{}-his{}-weight_{}-discount_{}-angle_{}-dp{}_{}_{}_{}-gpu{}-{}"\
        .format(args.history_rate, args.dataset, args.encoder, args.decoder, args.n_layers, args.dilate_len, args.train_history_len, args.weight, args.discount, args.angle,
                args.dropout, args.input_dropout, args.hidden_dropout, args.feat_dropout, args.gpu, args.save)
    model_state_file = os.path.join('../models/', model_name)
    print("Sanity Check: stat name : {}".format(model_state_file))
    print("Sanity Check: Is cuda available ? {}".format(torch.cuda.is_available()))

    use_cuda = args.gpu >= 0 and torch.cuda.is_available()

    if args.add_static_graph:
        static_triples = np.array(_read_triplets_as_list("../data/" + args.dataset + "/e-w-graph.txt", {}, {}, load_time=False))
        num_static_rels = len(np.unique(static_triples[:, 1]))
        num_words = len(np.unique(static_triples[:, 2]))
        static_triples[:, 2] = static_triples[:, 2] + num_nodes 
        static_node_id = torch.from_numpy(np.arange(num_words + data.num_nodes)).view(-1, 1).long().cuda(args.gpu) \
            if use_cuda else torch.from_numpy(np.arange(num_words + data.num_nodes)).view(-1, 1).long()
    else:
        num_static_rels, num_words, static_triples, static_graph = 0, 0, [], None

    # create stat
    model = RecurrentRGCN(args.decoder,
                          args.encoder,
                        num_nodes,
                        num_rels,
                        num_static_rels,
                        num_words,
                        num_times,
                        time_interval,
                        args.n_hidden,
                        args.opn,
                        args.history_rate,
                        sequence_len=args.train_history_len,
                        num_bases=args.n_bases,
                        num_basis=args.n_basis,
                        num_hidden_layers=args.n_layers,
                        dropout=args.dropout,
                        self_loop=args.self_loop,
                        skip_connect=args.skip_connect,
                        layer_norm=args.layer_norm,
                        input_dropout=args.input_dropout,
                        hidden_dropout=args.hidden_dropout,
                        feat_dropout=args.feat_dropout,
                        aggregation=args.aggregation,
                        weight=args.weight,
                        discount=args.discount,
                        angle=args.angle,
                        use_static=args.add_static_graph,
                        entity_prediction=args.entity_prediction,
                        relation_prediction=args.relation_prediction,
                        use_cuda=use_cuda,
                        gpu = args.gpu,
                        analysis=args.run_analysis)

    if use_cuda:
        torch.cuda.set_device(args.gpu)
        model.cuda()

    if args.add_static_graph:
        static_graph = build_sub_graph(len(static_node_id), num_static_rels, static_triples, use_cuda, args.gpu)

    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)

    
    if args.test and os.path.exists(model_state_file):
        mrr_raw, mrr_filter, mrr_raw_r, mrr_filter_r, hit_result_raw, hit_result_filter, hit_result_raw_r, hit_result_filter_r = test(model,
                                                            train_list+valid_list, 
                                                            test_list, 
                                                            num_rels, 
                                                            num_nodes, 
                                                            use_cuda, 
                                                            all_ans_list_test, 
                                                            all_ans_list_r_test, 
                                                            model_state_file, 
                                                            static_graph,
                                                            test_times,
                                                            history_test_time_nogt,
                                                            "test")

    elif args.test and not os.path.exists(model_state_file):
        print("--------------{} not exist, Change mode to train and generate stat for testing----------------\n".format(model_state_file))
    else:
        print("----------------------------------------start training----------------------------------------\n")
        best_mrr = 0
        for epoch in range(args.n_epochs):
            model.train()
            losses = []
            losses_e = []
            losses_r = []
            losses_static = []

            idx = [_ for _ in range(len(train_list))]
            random.shuffle(idx)

            for train_sample_num in tqdm(idx):
                if train_sample_num == 0: continue
                output = train_list[train_sample_num:train_sample_num+1]
                if train_sample_num - args.train_history_len<0:
                    input_list = train_list[0: train_sample_num]
                else:
                    input_list = train_list[train_sample_num - args.train_history_len: train_sample_num]

                # generate history graph

                
                history_glist = [build_sub_graph(num_nodes, num_rels, snap, use_cuda, args.gpu) for snap in input_list]
                output = [torch.from_numpy(_).long().cuda() for _ in output] if use_cuda else [torch.from_numpy(_).long() for _ in output]
                knowledge_list = get_knowledge(args)
                knowledgeseq = rec_knowledge(args,knowledge_list)
                # history load
                histroy_data = output[0]
                inverse_histroy_data = histroy_data[:, [2, 1, 0, 3]]
                inverse_histroy_data[:, 1] = inverse_histroy_data[:, 1] + num_rels
                histroy_data = torch.cat([histroy_data, inverse_histroy_data])
                histroy_data = histroy_data.cpu().numpy()
                # tail
                all_tail_seq = sp.load_npz(
                    '../data/{}/history/tail_history_{}.npz'.format(args.dataset, train_times[train_sample_num]))
                seq_idx = histroy_data[:, 0] * num_rels * 2 + histroy_data[:, 1]
                tail_seq = torch.Tensor(all_tail_seq[seq_idx].todense())
                one_hot_tail_seq = tail_seq.masked_fill(tail_seq != 0, 1)
                # rel
                all_rel_seq = sp.load_npz(
                    '../data/{}/history/rel_history_{}.npz'.format(args.dataset, train_times[train_sample_num]))
                rel_seq_idx = histroy_data[:, 0] * num_nodes + histroy_data[:, 2]
                rel_seq = torch.Tensor(all_rel_seq[rel_seq_idx].todense())
                one_hot_rel_seq = rel_seq.masked_fill(rel_seq != 0, 1)
                if use_cuda:
                    one_hot_tail_seq = one_hot_tail_seq.cuda()
                    one_hot_rel_seq = one_hot_rel_seq.cuda()
                # knowledgeseq = ''
                loss_e, loss_r, loss_static = model.get_loss(history_glist, output[0], static_graph, one_hot_tail_seq, one_hot_rel_seq, use_cuda,knowledgeseq)
                loss = args.task_weight*loss_e + (1-args.task_weight)*loss_r + loss_static

                losses.append(loss.item())
                losses_e.append(loss_e.item())
                losses_r.append(loss_r.item())
                losses_static.append(loss_static.item())

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_norm)  # clip gradients
                optimizer.step()
                optimizer.zero_grad()

            print("Epoch {:04d} | Ave Loss: {:.4f} | entity-relation-static:{:.4f}-{:.4f}-{:.4f} Best MRR {:.4f} | Model {} "
                  .format(epoch, np.mean(losses), np.mean(losses_e), np.mean(losses_r), np.mean(losses_static), best_mrr, model_name))

            # validation
            if epoch and epoch % args.evaluate_every == 0:
                mrr_raw, mrr_filter, mrr_raw_r, mrr_filter_r, hit_result_raw, hit_result_filter, hit_result_raw_r, hit_result_filter_r = test(model,
                                                                    train_list, 
                                                                    valid_list, 
                                                                    num_rels, 
                                                                    num_nodes, 
                                                                    use_cuda, 
                                                                    all_ans_list_valid, 
                                                                    all_ans_list_r_valid, 
                                                                    model_state_file, 
                                                                    static_graph,
                                                                    valid_times,
                                                                    history_val_time_nogt,
                                                                    mode="train")
                
                if not args.relation_evaluation:  # entity prediction evalution
                    if mrr_raw < best_mrr:
                        if epoch >= args.n_epochs:
                            break
                    else:
                        best_mrr = mrr_raw
                        torch.save({'state_dict': model.state_dict(), 'epoch': epoch}, model_state_file)
                else:
                    if mrr_raw_r < best_mrr:
                        if epoch >= args.n_epochs:
                            break
                    else:
                        best_mrr = mrr_raw_r
                        torch.save({'state_dict': model.state_dict(), 'epoch': epoch}, model_state_file)
        mrr_raw, mrr_filter, mrr_raw_r, mrr_filter_r, hit_result_raw, hit_result_filter, hit_result_raw_r, hit_result_filter_r = test(model,
                                                            train_list+valid_list,
                                                            test_list, 
                                                            num_rels, 
                                                            num_nodes, 
                                                            use_cuda, 
                                                            all_ans_list_test, 
                                                            all_ans_list_r_test, 
                                                            model_state_file, 
                                                            static_graph,
                                                            test_times,
                                                            history_test_time_nogt,
                                                            mode="test")
    return mrr_raw, mrr_filter, mrr_raw_r, mrr_filter_r, hit_result_raw, hit_result_filter, hit_result_raw_r, hit_result_filter_r

def get_knowledge(args):
    # output_list->[s,r,o,t]*297==>s,r==>[]*297
    datapath = '../con_models/conceptnet/data/all_e_' + args.dataset + '.pickle'
    if os.path.exists(datapath):
        knowledge_l = torch.load(datapath)
        return knowledge_l
    # subprocess.run(["python", "./longformer.py", "-i", i])
    device = "0"
    model_file = '../con_models/1e-05_adam_64_15500.pickle'
    opt, state_dict = con_interactive.load_model_file(model_file)

    data_loader, text_encoder = con_interactive.load_data("conceptnet", opt)

    n_ctx = data_loader.max_e1 + data_loader.max_e2 + data_loader.max_r
    n_vocab = len(text_encoder.encoder) + n_ctx

    model = con_interactive.make_model(opt, n_vocab, n_ctx, state_dict)
    con_cfg.device = int(device)
    con_cfg.do_gpu = True
    torch.cuda.set_device(con_cfg.device)
    model.cuda(con_cfg.device)    
    
    datasetname_l = ['GDELT','ICEWS18','WIKI','YAGO','ICEWS05']
    entity_path_l = ['../data/GDELT/entity2id.txt','../data/ICEWS18/entity2id.txt','../data/WIKI/entity2id.txt','../data/YAGO/entity2id.txt','../data/ICEWS05-15/entity2id.txt']
    entity_path = '../data/ICEWS14/entity2id.txt'
    relation_path = '../data/ICEWS14/relation2id.txt'
    for eni in range(len(entity_path_l)):
        entity_path = entity_path_l[eni]
        datasetname = datasetname_l[eni]
        entity_dict = {}
        relation_dict = {}
        with open(entity_path, 'r+', encoding='utf-8') as f:
            for line in f:
                line = line.strip().split('\t')
                entity_dict[int(line[1])] = line[0]
        with open(relation_path, 'r+', encoding='utf-8') as f:
            for line in f:
                line = line.strip().split('\t')
                relation_dict[int(line[1])] = line[0]

        input_event_l = []
        relation_l = []
        input_event_l = list(entity_dict.values())
        for teple in input_event_l:
        # input_event_id = teple[0]
        # input_event = entity_dict[input_event_id]
        # input_event_l.append(input_event)
            relation_l.append("all")
        sampling_algorithm = 'topk-1'# greedy / beam-1 / topk-1

        sampler = con_interactive.set_sampler(opt, sampling_algorithm, data_loader)
    # if relation not in con_data.conceptnet_data.conceptnet_relations:
    #     relation = "all"
        knowledge_l = []
        for i in range(len(relation_l)):
            outputs = con_interactive.get_conceptnet_sequence(input_event_l[i], model, sampler, data_loader, text_encoder, relation_l[i])
            outputs_l = dconoutput(outputs)
            knowledge_l.append(outputs_l)
        torch.save(knowledge_l,"/home/icdm/NewWorld/lsc/TiRGN-main/con_models/conceptnet/data/all_e_"+ datasetname+".pickle")
    # todo name -> id
    # knowledge_l = output_list
    return knowledge_l

def  rec_knowledge(args,knowledge_l):
    
    datapath = '../recformer/data/all_e_'+ args.dataset + '_rec.pickle'
    if os.path.exists(datapath):
        outputtensor = torch.load(datapath)
        return outputtensor

    t_len = len(knowledge_l)
    step = 50
    f_size = t_len / step
    i_size = t_len // step
    if f_size > i_size:
        i_size = i_size + 1
    for j in range(i_size):
        jj = str(j)
        subprocess.run(["python", "longformer.py","-i",jj])
    # knowledge_lists = []
    # for i in range(0, len(knowledge_l), step):
    #     sub_list = knowledge_l[i:i+step] 
    #     knowledge_lists.append(sub_list) 
    
    output_list =[]
    
    
    # RELATIONS = {         
    # 'AtLocation': 'at location',
    # 'CapableOf': 'capable of',
    # 'Causes': 'causes',
    # 'CausesDesire': 'causes desire',
    # 'DesireOf': 'desire of',
    # 'Desires': 'desires',
    # 'HasA': 'has a',
    # 'HasProperty': 'has property',
    # 'InheritsFrom': 'inherits from',
    # 'IsA': 'is a',
    # 'LocatedNear': 'located near',
    # 'LocationOfAction': 'location of action',
    # 'MotivatedByGoal': 'motivated by goal',
    # 'NotHasA': 'not has a',
    # 'NotHasProperty': 'not has property',
    # 'NotIsA': 'not is a',
    # 'PartOf': 'part of',
    # 'ReceivesAction': 'receives action',
    # 'RelatedTo': 'related to',
    # 'SymbolOf': 'symbol of'
    # }
    # model = LongformerModel.from_pretrained("../recformer/longformer-base-4096")
    # tokenizer = AutoTokenizer.from_pretrained("../recformer/longformer-base-4096")

    # all_knowledge = knowledge_l
    # i = 0
    # output_list = []
    
    
    # for knowledgetext in all_knowledge:
    #     knowledge_list = []
    #     for text in knowledgetext:
        
    #         if len(knowledge_list)==0:
    #             if len(text)==3:
    #                 knowledge_list.append('[CLS]'+ ' ' + text['e1']+ ' ' + RELATIONS[text['relation']] + ' ' + text['beams'])
    #             else:
    #                 knowledge_list.append('[CLS]'+ ' ' + text['e1'])
    #                 break
    #         else:

    #             if len(text)==3:
    #                 knowledge_list.append(' ' + RELATIONS[text['relation']] + ' ' + text['beams'])
    #             # else:
    #             #     knowledge_list.append(text['e1']+ ' ' + RELATIONS[text['relation']])
    #     SAMPLE_TEXT = "".join(knowledge_list)  # long input document
    #     input_ids = torch.tensor(tokenizer.encode(SAMPLE_TEXT)).unsqueeze(0)  # batch of size 1
    #     attention_mask = torch.ones(input_ids.shape, dtype=torch.long, device=input_ids.device)  # initialize to local attention
    #     global_attention_mask = torch.zeros(input_ids.shape, dtype=torch.long, device=input_ids.device)  # initialize to global attention to be deactivated for all tokens
    #     outputs = model(input_ids, attention_mask=attention_mask, global_attention_mask=global_attention_mask)[0].squeeze(0)
    #     output = outputs[0]
    #     output_list.append(output)
    #     print(i)
    #     i += 1
    outputtensor = torch.stack(output_list, dim=0)
    return outputtensor

def text2tensor(lit0,textlen):
    input_ids_l = lit0
    input_ids_head = input_ids_l[:1]
    input_ids_list = input_ids_l[1:]
    n = textlen
    size = len(input_ids_list) // n
    lists = []
    for i in range(n): 
        lists.append(input_ids_head + input_ids_list[i * size:(i + 1) * size])
    input_ids_lsts = np.array(lists)
    # input_ids = torch.from_numpy(input_ids_lsts)
    input_ids = torch.from_numpy(input_ids_lsts).cuda()
    return input_ids
def dconoutput(output):
    output_list =[]
    select_l = ['AtLocation', 'CapableOf', 'Causes', 'CausesDesire', 'DesireOf', 'Desires', 'HasA', 'HasProperty', 'InheritsFrom', 'IsA', 'LocatedNear', 'LocationOfAction',  'MotivatedByGoal', 'NotHasA', 'NotHasProperty', 'NotIsA', 'PartOf', 'ReceivesAction', 'RelatedTo', 'SymbolOf']
    for select in select_l:
        single_dict = output[select]
        if 'beams' not in list(single_dict.keys()):
            output_list.append(single_dict)
            continue
        beams_l = single_dict['beams']
        single_dict['beams'] = beams_l[0]
        output_list.append(single_dict)
    return output_list
def releasecuda(input_idss,item_position_idss,token_type_idss,attention_masks,global_attention_masks,output):
    input_idss = input_idss.cpu()
    item_position_idss = item_position_idss.cpu()
    token_type_idss = token_type_idss.cpu()
    attention_masks = attention_masks.cpu()
    global_attention_masks = global_attention_masks.cpu()
    output = output.cpu()
    
    
if __name__ == '__main__':
    import warnings
    warnings.filterwarnings("ignore")
    parser = argparse.ArgumentParser(description='TIRGN')

    parser.add_argument("--gpu", type=int, default=1,
                        help="gpu,default=-1")
    parser.add_argument("--batch-size", type=int, default=1,
                        help="batch-size")
    parser.add_argument("-d", "--dataset", type=str, default='ICEWS14',
                        help="dataset to use")
    parser.add_argument("--test", action='store_true', default=False,
                        help="load stat from dir and directly test,default=False")
    parser.add_argument("--run-analysis", action='store_true', default=False,
                        help="print log info")
    parser.add_argument("--run-statistic", action='store_true', default=False,
                        help="statistic the result")
    parser.add_argument("--multi-step", action='store_true', default=False,
                        help="do multi-steps inference without ground truth")
    parser.add_argument("--topk", type=int, default=50,
                        help="choose top k entities as results when do multi-steps without ground truth")
    parser.add_argument("--add-static-graph",  action='store_true', default=True,
                        help="use the info of static graph, default=False")
    parser.add_argument("--add-rel-word", action='store_true', default=False,
                        help="use words in relaitons")
    parser.add_argument("--relation-evaluation", action='store_true', default=False,
                        help="save model accordding to the relation evalution")

    # configuration for encoder RGCN stat
    parser.add_argument("--weight", type=float, default=0.5,
                        help="weight of static constraint, default=1")
    parser.add_argument("--task-weight", type=float, default=1,
                        help="weight of entity prediction task")
    parser.add_argument("--discount", type=float, default=1,
                        help="discount of weight of static constraint")
    parser.add_argument("--angle", type=int, default=14,
                        help="evolution speed, default=10")

    parser.add_argument("--encoder", type=str, default="convgcn",
                        help="method of encoder, default='uvrgcn'")
    parser.add_argument("--aggregation", type=str, default="none",
                        help="method of aggregation")
    parser.add_argument("--dropout", type=float, default=0.2,
                        help="dropout probability")
    parser.add_argument("--skip-connect", action='store_true', default=False,
                        help="whether to use skip connect in a RGCN Unit")
    parser.add_argument("--n-hidden", type=int, default=200,
                        help="number of hidden units")
    parser.add_argument("--opn", type=str, default="sub",
                        help="opn of compgcn")

    parser.add_argument("--n-bases", type=int, default=100,
                        help="number of weight blocks for each relation")
    parser.add_argument("--n-basis", type=int, default=100,
                        help="number of basis vector for compgcn")
    parser.add_argument("--n-layers", type=int, default=2,
                        help="number of propagation rounds")
    parser.add_argument("--self-loop", action='store_true', default=True,
                        help="perform layer normalization in every layer of gcn ")
    parser.add_argument("--layer-norm", action='store_true', default=False,
                        help="perform layer normalization in every layer of gcn ")
    parser.add_argument("--relation-prediction", action='store_true', default=True,
                        help="add relation prediction loss, default=False")
    parser.add_argument("--entity-prediction", action='store_true', default=True,
                        help="add entity prediction loss, default=False")
    parser.add_argument("--split_by_relation", action='store_true', default=False,
                        help="do relation prediction")

    # configuration for stat training
    parser.add_argument("--n-epochs", type=int, default=70,
                        help="number of minimum training epochs on each time step")
    parser.add_argument("--lr", type=float, default=0.001,
                        help="learning rate")
    parser.add_argument("--grad-norm", type=float, default=1.0,
                        help="norm to clip gradient to")

    # configuration for evaluating
    parser.add_argument("--evaluate-every", type=int, default=1,
                        help="perform evaluation every n epochs, default=20")

    # configuration for decoder
    parser.add_argument("--decoder", type=str, default="timeconvtranse",
                        help="method of decoder, default='convtranse'")
    parser.add_argument("--input-dropout", type=float, default=0.2,
                        help="input dropout for decoder ")
    parser.add_argument("--hidden-dropout", type=float, default=0.2,
                        help="hidden dropout for decoder")
    parser.add_argument("--feat-dropout", type=float, default=0.2,
                        help="feat dropout for decoder")

    # configuration for sequences stat
    parser.add_argument("--train-history-len", type=int, default=9,
                        help="history length, default=10")
    parser.add_argument("--test-history-len", type=int, default=9,
                        help="history length for test, default=20")
    parser.add_argument("--dilate-len", type=int, default=1,
                        help="dilate history graph")

    # configuration for optimal parameters
    parser.add_argument("--grid-search", action='store_true', default=False,
                        help="perform grid search for best configuration")
    parser.add_argument("-tune", "--tune", type=str, default="history_len,n_layers,dropout,n_bases,angle,history_rate",
                        help="stat to use")
    parser.add_argument("--num-k", type=int, default=500,
                        help="number of triples generated")

    # configuration for global history
    parser.add_argument("--history-rate", type=float, default=0,
                        help="history rate")

    parser.add_argument("--save", type=str, default='checkpoint',
                        help="number of save, default='one'")



    args = parser.parse_args()
    print(args)
    if args.grid_search:
        out_log = '../results/{}.{}.gs'.format(args.dataset, args.encoder+"-"+args.decoder+"-"+args.save)
        o_f = open(out_log, 'w')
        print("** Grid Search **")
        o_f.write("** Grid Search **\n")
        hyperparameters = args.tune.split(',')

        if args.tune == '' or len(hyperparameters) < 1:
            print("No hyperparameter specified.")
            sys.exit(0)
        if args.dataset == "ICEWS14s":
            hp_range_ = hp_range
        if args.dataset == "WIKI":
            hp_range_ = hp_range_WIKI
        if args.dataset == "YAGO":
            hp_range_ = hp_range_YAGO
        if args.dataset == "ICEWS18":
            hp_range_ = hp_range_ICEWS18
        if args.dataset == "ICEWS05-15":
            hp_range_ = hp_range_ICEWS05_15
        if args.dataset == "GDELT":
            hp_range_ = hp_range_GDELT
        grid = hp_range_[hyperparameters[0]]
        for hp in hyperparameters[1:]:
            grid = itertools.product(grid, hp_range_[hp])
        hits_at_1s = {}
        hits_at_10s = {}
        mrrs = {}
        grid = list(grid)
        print('* {} hyperparameter combinations to try'.format(len(grid)))
        o_f.write('* {} hyperparameter combinations to try\n'.format(len(grid)))
        o_f.close()

        for i, grid_entry in enumerate(list(grid)):

            o_f = open(out_log, 'a')

            if not (type(grid_entry) is list or type(grid_entry) is list):
                grid_entry = [grid_entry]
            grid_entry = utils.flatten(grid_entry)
            print('\n\n* Hyperparameter Set {}:'.format(i))
            o_f.write('* Hyperparameter Set {}:\n'.format(i))
            signature = ''
            print(grid_entry)
            o_f.write("\t".join([str(_) for _ in grid_entry]) + "\n")
            # def run_experiment(args, n_hidden=None, n_layers=None, dropout=None, n_bases=None):
            args.test = False
            args.multi_step = False
            mrr_raw, mrr_filter, mrr_raw_r, mrr_filter_r, hit_result_raw, hit_result_filter, hit_result_raw_r, hit_result_filter_r = run_experiment(args, grid_entry[0], grid_entry[1], grid_entry[2], grid_entry[3], grid_entry[4], grid_entry[5])
            hits = [1, 3, 10]
            o_f.write("MRR (raw): {:.6f}\n".format(mrr_raw))
            for hit_i, hit in enumerate(hits):
                o_f.write("Hits (raw) @ {}: {:.6f}\n".format(hit, hit_result_raw[hit_i].item()))
            o_f.write("MRR (raw): {:.6f}\n".format(mrr_filter))
            for hit_i, hit in enumerate(hits):
                o_f.write("Hits (raw) @ {}: {:.6f}\n".format(hit, hit_result_filter[hit_i].item()))
            o_f.write("MRR (raw): {:.6f}\n".format(mrr_raw_r))
            for hit_i, hit in enumerate(hits):
                o_f.write("Hits (raw) @ {}: {:.6f}\n".format(hit, hit_result_raw_r[hit_i].item()))
            o_f.write("MRR (raw): {:.6f}\n".format(mrr_filter_r))
            for hit_i, hit in enumerate(hits):
                o_f.write("Hits (raw) @ {}: {:.6f}\n".format(hit, hit_result_filter_r[hit_i].item()))
            # no ground truth
            args.test = True
            args.topk = 0
            args.multi_step = True
            mrr_raw, mrr_filter, mrr_raw_r, mrr_filter_r, hit_result_raw, hit_result_filter, hit_result_raw_r, hit_result_filter_r = run_experiment(
                args, grid_entry[0], grid_entry[1], grid_entry[2], grid_entry[3], grid_entry[4], grid_entry[5])
            o_f.write("No ground truth result:\n")
            o_f.write("MRR (raw): {:.6f}\n".format(mrr_raw))
            for hit_i, hit in enumerate(hits):
                o_f.write("Hits (raw) @ {}: {:.6f}\n".format(hit, hit_result_raw[hit_i].item()))
            o_f.write("MRR (raw): {:.6f}\n".format(mrr_filter))
            for hit_i, hit in enumerate(hits):
                o_f.write("Hits (raw) @ {}: {:.6f}\n".format(hit, hit_result_filter[hit_i].item()))
            o_f.write("MRR (raw): {:.6f}\n".format(mrr_raw_r))
            for hit_i, hit in enumerate(hits):
                o_f.write("Hits (raw) @ {}: {:.6f}\n".format(hit, hit_result_raw_r[hit_i].item()))
            o_f.write("MRR (raw): {:.6f}\n".format(mrr_filter_r))
            for hit_i, hit in enumerate(hits):
                o_f.write("Hits (raw) @ {}: {:.6f}\n".format(hit, hit_result_filter_r[hit_i].item()))

    # single run
    else:
        run_experiment(args)
    sys.exit()



