import warnings

import torch
import numpy as np
import os
import json
from tqdm import tqdm
from multiprocessing.dummy import Pool as ThreadPool
from multiprocessing import Pool
import torch.nn.functional as F

import networkx as nx
import random

def load_problem(name):
    from problems import TSP, CVRP, SDVRP, OP, PCTSPDet, PCTSPStoch, TopoSort
    problem = {
        'toposort': TopoSort,
        'tsp': TSP,
        'cvrp': CVRP,
        'sdvrp': SDVRP,
        'op': OP,
        'pctsp_det': PCTSPDet,
        'pctsp_stoch': PCTSPStoch,
    }.get(name, None)
    assert problem is not None, "Currently unsupported problem: {}!".format(name)
    return problem


def torch_load_cpu(load_path):
    return torch.load(load_path, map_location=lambda storage, loc: storage)  # Load on CPU


def move_to(var, device):
    if isinstance(var, dict):
        return {k: move_to(v, device) for k, v in var.items()}
    return var.to(device)


def _load_model_file(load_path, model):
    """Loads the model with parameters from the file and returns optimizer state dict if it is in the file"""

    # Load the model parameters from a saved state
    load_optimizer_state_dict = None
    print('  [*] Loading model from {}'.format(load_path))

    load_data = torch.load(
        os.path.join(
            os.getcwd(),
            load_path
        ), map_location=lambda storage, loc: storage)

    if isinstance(load_data, dict):
        load_optimizer_state_dict = load_data.get('optimizer', None)
        load_model_state_dict = load_data.get('model', load_data)
    else:
        load_model_state_dict = load_data.state_dict()

    state_dict = model.state_dict()

    state_dict.update(load_model_state_dict)

    model.load_state_dict(state_dict)

    return model, load_optimizer_state_dict


def load_args(filename):
    with open(filename, 'r') as f:
        args = json.load(f)

    # Backwards compatibility
    if 'data_distribution' not in args:
        args['data_distribution'] = None
        probl, *dist = args['problem'].split("_")
        if probl == "op":
            args['problem'] = probl
            args['data_distribution'] = dist[0]
    return args


def load_model(path, epoch=None):
    from nets.attention_model import AttentionModel
    from nets.pointer_network import PointerNetwork

    if os.path.isfile(path):
        model_filename = path
        path = os.path.dirname(model_filename)
    elif os.path.isdir(path):
        if epoch is None:
            epoch = max(
                int(os.path.splitext(filename)[0].split("-")[1])
                for filename in os.listdir(path)
                if os.path.splitext(filename)[1] == '.pt'
            )
        model_filename = os.path.join(path, 'epoch-{}.pt'.format(epoch))
    else:
        assert False, "{} is not a valid directory or file".format(path)

    args = load_args(os.path.join(path, 'args.json'))

    problem = load_problem(args['problem'])

    model_class = {
        'attention': AttentionModel,
        'pointer': PointerNetwork
    }.get(args.get('model', 'attention'), None)
    assert model_class is not None, "Unknown model: {}".format(model_class)

    model = model_class(
        args['embedding_dim'],
        args['hidden_dim'],
        problem,
        n_encode_layers=args['n_encode_layers'],
        mask_inner=True,
        mask_logits=True,
        normalization=args['normalization'],
        tanh_clipping=args['tanh_clipping'],
        checkpoint_encoder=args.get('checkpoint_encoder', False),
        shrink_size=args.get('shrink_size', None)
    )
    # Overwrite model parameters by parameters to load
    load_data = torch_load_cpu(model_filename)
    model.load_state_dict({**model.state_dict(), **load_data.get('model', {})})

    model, *_ = _load_model_file(model_filename, model)

    model.eval()  # Put in eval mode

    return model, args


def parse_softmax_temperature(raw_temp):
    # Load from file
    if os.path.isfile(raw_temp):
        return np.loadtxt(raw_temp)[-1, 0]
    return float(raw_temp)


def run_all_in_pool(func, directory, dataset, opts, use_multiprocessing=True):
    # # Test
    # res = func((directory, 'test', *dataset[0]))
    # return [res]

    num_cpus = os.cpu_count() if opts.cpus is None else opts.cpus

    w = len(str(len(dataset) - 1))
    offset = getattr(opts, 'offset', None)
    if offset is None:
        offset = 0
    ds = dataset[offset:(offset + opts.n if opts.n is not None else len(dataset))]
    pool_cls = (Pool if use_multiprocessing and num_cpus > 1 else ThreadPool)
    with pool_cls(num_cpus) as pool:
        results = list(tqdm(pool.imap(
            func,
            [
                (
                    directory,
                    str(i + offset).zfill(w),
                    *problem
                )
                for i, problem in enumerate(ds)
            ]
        ), total=len(ds), mininterval=opts.progress_bar_mininterval))

    failed = [str(i + offset) for i, res in enumerate(results) if res is None]
    assert len(failed) == 0, "Some instances failed: {}".format(" ".join(failed))
    return results, num_cpus


def do_batch_rep(v, n):
    if isinstance(v, dict):
        return {k: do_batch_rep(v_, n) for k, v_ in v.items()}
    elif isinstance(v, list):
        return [do_batch_rep(v_, n) for v_ in v]
    elif isinstance(v, tuple):
        return tuple(do_batch_rep(v_, n) for v_ in v)

    return v[None, ...].expand(n, *v.size()).contiguous().view(-1, *v.size()[1:])


def sample_many(inner_func, get_cost_func, input, batch_rep=1, iter_rep=1):
    """
    :param input: (batch_size, graph_size, node_dim) input node features
    :return:
    """
    input = do_batch_rep(input, batch_rep)

    costs = []
    pis = []
    for i in range(iter_rep):
        _log_p, pi = inner_func(input)
        # pi.view(-1, batch_rep, pi.size(-1))
        cost, mask = get_cost_func(input, pi)

        costs.append(cost.view(batch_rep, -1).t())
        pis.append(pi.view(batch_rep, -1, pi.size(-1)).transpose(0, 1))

    max_length = max(pi.size(-1) for pi in pis)
    # (batch_size * batch_rep, iter_rep, max_length) => (batch_size, batch_rep * iter_rep, max_length)
    pis = torch.cat(
        [F.pad(pi, (0, max_length - pi.size(-1))) for pi in pis],
        1
    )  # .view(embeddings.size(0), batch_rep * iter_rep, max_length)
    costs = torch.cat(costs, 1)

    # (batch_size)
    mincosts, argmincosts = costs.min(-1)
    # (batch_size, minlength)
    minpis = pis[torch.arange(pis.size(0), out=argmincosts.new()), argmincosts]

    return minpis, mincosts

def order_compare(order_rl, order_sorted):
    L = len(order_rl)
    cal_relationship = dict()
    sorted_relationship = dict()

    recall_num = 0
    radius = []

    for i in range(L):
        for key in cal_relationship:
            cal_relationship[key].append(order_rl[i]) 
        cal_relationship[order_rl[i].item()] = [i]
        
        for key in sorted_relationship:
            sorted_relationship[key].append(order_sorted[i]) 
        sorted_relationship[order_sorted[i].item()] = [i]

    for key in sorted_relationship:
        recall_num += len([element for element in sorted_relationship[key][1:] if element in cal_relationship[key][1:]]) 
        radius.append(abs(sorted_relationship[key][0]-cal_relationship[key][0]))
    
    #recall_accuracy = recall_num / ((L-1)*L/2.) 
    recall_accuracy = recall_num 
  
    radius_mean = sum(radius) / L

    radius.sort()
    
    return recall_accuracy, radius_mean, radius[-1]

def order_check(idx, idx_sorted):
    batch_size = idx.shape[0]

    accuracy_recall = [0. for _ in range(batch_size)] 
    radius_mean = [0. for _ in range(batch_size)]
    radius_max = [0 for _ in range(batch_size)]
    
    for order in range(batch_size):

        order_training = idx[order]
        order_sorted = idx_sorted[order]
         
        recall_accuracy, radius_m, max_radius = order_compare(order_training, order_sorted)

        accuracy_recall[order] = recall_accuracy
        radius_mean[order] = radius_m
        radius_max[order] = max_radius

    return sum(accuracy_recall)/len(accuracy_recall), sum(radius_mean)/len(radius_mean), max(radius_max), max(accuracy_recall), min(accuracy_recall)
    #return accuracy_recall

def orderCompare(order_rl, order_sorted):
    L = len(order_rl)
    cal_relationship = dict()
    sorted_relationship = dict()

    mis_match = 0
    recall_num = 0

    radius = []

    for i in range(L):
        if order_rl[i] != order_sorted[i]: mis_match += 1
        for key in cal_relationship:
            cal_relationship[key].append(order_rl[i]) 
        cal_relationship[order_rl[i].item()] = [i]
        
        for key in sorted_relationship:
            sorted_relationship[key].append(order_sorted[i]) 
        sorted_relationship[order_sorted[i].item()] = [i]

    for key in sorted_relationship:
        recall_num += len([element for element in sorted_relationship[key][1:] if element in cal_relationship[key][1:]]) 
        radius.append(abs(sorted_relationship[key][0]-cal_relationship[key][0]))
    
    recall_accuracy = recall_num / ((L-1)*L/2.) 
  
    radius_mean = sum(radius) / L

    radius.sort()
    
    return mis_match, recall_accuracy, radius_mean, radius[-1]

def orderCheck(idx, idx_sorted):
    batch_size = idx.shape[0]

    misMatch = [0 for _ in range(batch_size)]
    accuracy_recall = [0. for _ in range(batch_size)] 
    radius_mean = [0. for _ in range(batch_size)]
    radius_max = [0 for _ in range(batch_size)]
    
    for order in range(batch_size):

        order_training = idx[order]
        order_sorted = idx_sorted[order]
         
        mis_match, recall_accuracy, radius_m, max_radius = orderCompare(order_training, order_sorted)

        misMatch[order] = mis_match
        accuracy_recall[order] = recall_accuracy
        radius_mean[order] = radius_m
        radius_max[order] = max_radius

    return misMatch, accuracy_recall, radius_mean, radius_max

def smart_sort(x, permutation, dim=3):
    if dim == 2:
        d1, d2 = x.size()
        ret = x[
            torch.arange(d1).unsqueeze(1).repeat(1, d2).flatten(),
            permutation.flatten()
        ].view(d1, d2)
    else:
        d1, d2, d3 = x.size()
        ret = x[
            torch.arange(d1).unsqueeze(1).repeat(1, d2).flatten(),
            permutation.flatten()
        ].view(d1, d2, d3)
    return ret

def x_sorting(indice, data):
    L = len(data)    
    if len(indice) != L: return torch.empty(0).cuda()

    y_ = data[:,1].view(data.shape[0]) 
    x_ = data[:,0].view(data.shape[0]) 
    left, right = 0, 1
    completed_indices = torch.empty(0).cuda()
    while right <= L:
        if right == L or y_[left] != y_[right]:
            if right - left > 1:
                _, x_indice = torch.sort(x_[left:right], dim=0)
                if len(completed_indices) == 0:
                    completed_indices = smart_sort(indice[left:right].unsqueeze(0), x_indice.unsqueeze(0), 2).squeeze(0)
                else:
                    completed_indices = torch.cat((completed_indices, smart_sort(indice[left:right].unsqueeze(0), x_indice.unsqueeze(0), 2).squeeze(0)), dim=0)
            else:
                if len(completed_indices) == 0:
                    completed_indices = indice[left:right]
                else:
                    completed_indices = torch.cat((completed_indices, indice[left:right]), dim=0)
            left = right
        right += 1

    return completed_indices

def deep_sort_x(indices, d):
    d_y_sorting = smart_sort(d, indices)
    batch_size = len(d)
    
    for i in range(batch_size):
        indices[i] = x_sorting(indices[i], d_y_sorting[i])
    return indices

def level_mismatch_cost(y_s, y):
    L = len(y_s)    
    if L != len(y): 
        print("Warning: lenth mismatch of compared y axis data")
        return torch.empty(0).cuda(), torch.empty(0).cuda()
    
    cos = torch.nn.CosineSimilarity(dim=0, eps=1e-6)
    left, right = 0, 1
    coordi_sorted = torch.empty(0).cuda() 
    coordi = torch.empty(0).cuda() 

    while right <= L:
        if right == L or y_s[left] != y_s[right]:
            if left == 0:
                coordi_sorted = y_s[left:(left+1)]
                coordi = y[left:right].mean().unsqueeze(0)
            else:
                coordi_sorted = torch.cat((coordi_sorted, y_s[left:left+1]), dim=0)
                coordi = torch.cat((coordi, y[left:right].mean().unsqueeze(0)), dim=0)
            #cost_mean += torch.sqrt(torch.sum(torch.square(torch.sub(y[left:right], y_s[left:right])))) 
            left = right
        right += 1
    return cos(coordi_sorted, coordi).unsqueeze(0).cuda()
        
def level_sorting(y_sorted, y_):
    batch_size = len(y_)
        
    cost = torch.empty(0).cuda()
    
    for i in range(batch_size):
        if i == 0:
            cost = level_mismatch_cost(y_sorted[i], y_[i])
        else:
            cost = torch.cat((cost, level_mismatch_cost(y_sorted[i], y_[i])), dim=0)
    return cost

def level_sorting_xy_pairs(indices, d, alpha=0.2, beta=0.8):
    y_ = d[:,:,1].view(d.shape[0], d.shape[1])
    x_ = d[:,:,0].view(d.shape[0], d.shape[1])
    d_y_sorting = smart_sort(d, indices)
    batch_size = len(d)
    
    for i in range(batch_size):
        indices[i] = x_sorting(indices[i], d_y_sorting[i])

    d_xy_sorting = smart_sort(d, indices)
    y_s = d_xy_sorting[:,:,1].view(d_xy_sorting.shape[0], d_xy_sorting.shape[1])
    x_s = d_xy_sorting[:,:,0].view(d_xy_sorting.shape[0], d_xy_sorting.shape[1])
    cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
    cost = alpha * cos(x_s.cuda(), x_.cuda()).cuda() + beta * cos(y_s.cuda(), y_.cuda()).cuda()
    return cost, indices

def topo_sorting(graph):
    head = []
    tail = []

    nodes_of_multiPredecessors = []
    visited = dict()

    def combine(path1, path2):
        if len(path1) <= 0 or len(path2) <= 0: return path1 + path2
        result_path = []
    
        pivot = -1
        pivot_id1 = len(path1)
        pivot_id2 = len(path2)
        
        for _id1 in range(len(path1)):
            if pivot_id1 < len(path1): break
            if path1[_id1] in nodes_of_multiPredecessors:
                for _id2 in range(len(path2)):
                    if path2[_id2] == path1[_id1]:
                        pivot = path1[_id1]
                        pivot_id1 = _id1
                        pivot_id2 = _id2

        sub_combine = []
        if pivot_id1 < len(path1) and pivot_id2 < len(path2):
            sub_combine = combine(path1[pivot_id1+1:], path2[pivot_id2+1:])
            path1 = path1[:pivot_id1]
            path2 = path2[:pivot_id2]
        
        point1, point2 = 0, 0
        while point1 < len(path1) and point2 < len(path2):
            if path1[point1] < path2[point2]:
                result_path.append(path1[point1])
                point1 += 1
            else:
                result_path.append(path2[point2])
                point2 += 1
        if point1 < len(path1): result_path += path1[point1:]
        elif point2 < len(path2): result_path += path2[point2:]
        
        if pivot >= 0: result_path.append(pivot)
        return result_path + sub_combine
        
    def path_finder(node):
        #node_id = np.where(node==2)[0][0]
        node_id = torch.nonzero(node==2).item()

        if node_id in visited:
            return visited[node_id]
        if node_id in tail:
            return []

        sub_paths = []
        #for child_id in np.where(node==1)[0]:
        for child_id in torch.nonzero(node==1):
            sub_paths.append(path_finder(graph[[(graph[i][child_id]==2).item() for i in range(graph.shape[0])].index(True)]))
        
        while len(sub_paths) > 1:
            combine_path = combine(sub_paths[0], sub_paths[1])
            sub_paths = sub_paths[2:] + [combine_path]

        visited[node_id] = sub_paths[0] 

        #if np.where(node==-1)[0].shape[0] > 1:
        if torch.nonzero(node==-1).shape[0] > 1:
            nodes_of_multiPredecessors.append(node_id)

        return [node_id] + sub_paths[0]
            
    for idx in range(graph.shape[0]):
        embedding_node = graph[idx]

        if sum([i>=0 for i in embedding_node]) == embedding_node.shape[0]:
            head.append(embedding_node)    
        elif sum([i<=0 for i in embedding_node]) == embedding_node.shape[0] - 1:
            tail.append(torch.nonzero(embedding_node==2).item())
            
    path_collections = []
    for head_node in head:
        path_collections.append(path_finder(head_node)[1:])

    while len(path_collections) > 1:
        path_combination = combine(path_collections[0], path_collections[1])
        path_collections = path_collections[2:] + [path_combination]

    result = sorted([torch.nonzero(head[i]==2).item() for i in range(len(head))]) + path_collections[0] + sorted(tail)
    print(result)
    return result
        
def graph_sorting_DAG(dataset):
    indices = torch.tensor([[0]*dataset.shape[1] for _ in range(dataset.shape[0])]).cuda()
   
    batch_size = dataset.shape[0]
    for i in range(batch_size):
        print("treated graph: ")
        print(dataset[i])
        indices[i] = torch.tensor(topo_sorting(dataset[i])).cuda()

    return indices

def tensor_to_string(data):
    sequence = []
    for d in data:
        sequence.append(str(d.cpu().numpy())+"\n")
    return sequence
"""
if __name__ == "__main__":
    #data = []

    #size = 10
    #num_samples = 2
    #random.seed(0)
    order = [i for i in range(size)]
    for _ in range(num_samples):
        G = nx.gnp_random_graph(size, random.random())   
        D = None 
        while True:
            if nx.is_connected(G):
                D = nx.DiGraph([(u, v) for u, v in G.edges()]) 
                if nx.is_directed_acyclic_graph(D):
                    break
            G = nx.gnp_random_graph(size, random.random())   
        random.shuffle(order)
        mapping = {idx:item for idx, item in enumerate(order)}
        DAG = nx.relabel.relabel_nodes(D, mapping)
                
        graph = np.diag([2]*size)
        for u, v in DAG.edges():
            graph[u][v] = 1 
            graph[v][u] = -1 
                
        data.append(torch.FloatTensor(sorted(graph, key=lambda x : x[0]**2- x[-1]**3)))
    dataset = torch.stack(data).cuda()

    print(dataset)
    print(graph_sorting_DAG(dataset))

    
    idx = torch.tensor([[2, 1, 5, 0, 3, 4], [4, 1, 3, 2, 0, 5]]).cuda()
    #print(torch.max(idx) / 3)
    #p = 0.
    #t = torch.Tensor([i for i in range(int(list(idx.shape)[1]))]).cuda()
    #print(t)
    #idx_sorted = torch.tensor([i for i in range(idx.shape[1])]).cuda()
    #diff = torch.abs(idx - idx_sorted)
    
    #print(diff)
    #p = torch.max(diff)
    #print(p)
    #print(p.item())
    
    #diff = torch.mul(diff, torch.FloatTensor([1.]).cuda())
    #print(torch.mean(diff, 1).mean())
    #print(round(torch.mean(diff, 1).mean().item(), 2))

    #print(idx[:, 2:]>torch.tensor([[5], [3]]).cuda())
    #print(idx[:, 2:]>idx[:, 2].view(idx.shape[0], -1))
    #k = torch.tensor([torch.sum(idx[:, (i+1):]<idx[:, i].view(idx.shape[0], -1)) for i in range(idx.shape[1]-1)])      
    print(p)
    k = torch.cat([torch.sum(idx[:, (i+1):]>idx[:, i].view(idx.shape[0], -1), dim=1).view(idx.shape[0], -1) for i in range(idx.shape[1]-1)], dim=1)
    kk = torch.mul(torch.sum(k, dim=1), torch.FloatTensor([1.]).cuda()).mean()/2.
    m = torch.FloatTensor([0.]).cuda()
    m = torch.max(m, p)
    print("{:.2f}".format(m.item()))
    #a = torch.FloatTensor([10]).cuda()
    #b = torch.FloatTensor([12]).cuda()
    #print(torch.max(a, b))
    
    #print(k)

    #L = idx.size()[1]
    #idx_sorted = torch.tensor([[1, 3, 2, 0, 4], [4, 0, 3, 1, 2]]).cuda()
    #idx_sorted = torch.tensor([0, 1, 2, 3, 4, 5]).cuda()
    #difference = idx_sorted - idx
    #difference = torch.add(torch.sum(torch.div((torch.negative(torch.abs(difference)) + difference), 2), dim=1), torch.tensor(L*(L-1)/2).cuda())
    #print(difference)
    idx_sorted = torch.tensor([[0, 1, 2, 3, 4, 5], [0, 1, 2, 3, 4, 5]]).cuda()
    print(tensor_to_string(idx_sorted))
    #print(order_check(idx, idx_sorted))

    #misMatch, accuracy_recall, radius_mean, radius_max = orderCheck(idx, idx_sorted)
    #accuracy_recall, radius_mean, radius_max, accuracy_recall_max, accuracy_recall_min = order_check(idx, idx_sorted)
    #print("mis_match: {:.2f}".format(sum(misMatch)/len(misMatch)))
    #print("recall_accuracy: ", accuracy_recall)
    #print("maximum recall: ", accuracy_recall_max)
    #print("minimum recall: ", accuracy_recall_min)
    #print("radius_mean: ", radius_mean)
    #radius_max.sort()
    #print("radius_max: {:d}".format(radius_max))

    #data = torch.tensor([[0.4, 0.3], [0.2, 0.3], [0.3, 0.3], [0.1, 0.5], [0.1, 0.8], [0.5, 0.8], [0.9, 0.9]]).cuda() 
    #indice = torch.tensor([0, 1, 2, 3, 4, 5, 6]).cuda()
    #new_indice = x_sorting(indice, data)
    #dataset = torch.tensor([[[0.4, 0.5], [0.3, 0.5], [0.2, 0.3], [0.1, 0.3], [0.9, 0.1]], [[0.1, 0.8], [0.2, 0.4], [0.3, 0.8], [0.5, 0.7], [0.6, 0.4]]]).cuda()
    #idx = torch.tensor([[4, 2, 3, 1, 0], [4, 1, 3, 2, 0]]).cuda()
    #cost, indices = level_sorting_xy_pairs(idx, dataset)
    #print(cost)
    #print(indices)
    #print(deep_sort_x(idx, dataset))

    #y_s = torch.FloatTensor([[0.3, 0.3, 0.4, 0.5, 0.5, 0.6], [0.1, 0.1, 0.1, 0.3, 0.3, 0.7]]).cuda()
    #y = torch.FloatTensor([[0.4, 0.3, 0.3, 0.5, 0.6, 0.5], [0.3, 0.3, 0.1, 0.7, 0.1, 0.1]]).cuda()
    #y_s = torch.FloatTensor([0.3, 0.3, 0.4, 0.5, 0.5, 0.6]).cuda()
    #y = torch.FloatTensor([0.4, 0.3, 0.3, 0.5, 0.6, 0.5]).cuda()
    #res = level_mismatch_cost(y_s, y)    
    #print(res)
    #res = level_sorting(y_s, y)    
    #print(res)
    #print(res2)
"""
