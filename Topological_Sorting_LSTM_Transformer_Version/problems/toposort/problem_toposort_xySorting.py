from torch.utils.data import Dataset
import torch, random
import os
import pickle
from problems.toposort.state_toposort import StateTopoSort
from utils.beam_search import beam_search
from utils import orderCheck, deep_sort_x, level_sorting, level_sorting_xy_pairs, order_check

import networkx as nx
import numpy as np


class TopoSort(object):

    NAME = 'toposort'

    @staticmethod
    def get_costs(dataset, pi, measures=False, plot_data=False):
        # Gather dataset in order of graph nodes
        d = dataset.gather(1, pi.unsqueeze(-1).expand_as(dataset))
        """
        if plot_data: 
            print("learned dataset is: ")
            for element in d[-1].cpu():
                print(element)
            print("end of batch")
        """
        # generated index compared to optimal index; to be used for cost function below
        y_ = d[:,:,1].view(d.shape[0],d.shape[1])
        #print(d.shape, list(y_.shape), ((d[:, 1:] - d[:, :-1]).norm(p=2, dim=2).sum(1) + (d[:, 0] - d[:, -1]).norm(p=2, dim=1)).shape)
        sorted, indices = torch.sort(y_, dim=1)

        #New cost stragety to satisfy level sorting
        #cost = level_sorting(sorted, y_)
        cost, indices = level_sorting_xy_pairs(indices, d)
        #_, indices = level_sorting_xy_pairs(indices, d)
        """
        idx = torch.Tensor([i for i in range(int(list(y_.shape)[1]))]).cuda().repeat(list(y_.shape)[0]).view(d.shape[0],d.shape[1])
        if mode == 1:
            indices = deep_sort_x(indices, d)        
        #print(indices.shape, idx.shape)
        # sorting cost is measured with Cosine Similarity
        cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
        cost = cos(indices.cuda(),idx).cuda()
        """
        idx = torch.Tensor([i for i in range(int(list(y_.shape)[1]))]).cuda().repeat(list(y_.shape)[0]).view(d.shape[0],d.shape[1])
        #misMatch_y = torch.sub(y_, sorted)
        #misMatch_x = torch.sub(indices, idx)
        #cost = 0.2 * torch.count_nonzero(misMatch_x, dim=1) + 0.8 * torch.count_nonzero(misMatch_y, dim=1)
        if measures:
            recall_accuracy, radius_mean, radius_max, recall_accuracy_max, recall_accuracy_min = order_check(idx, indices.cuda()) 
            """
            graph_size = indices.shape[1]
            full_recall = (graph_size-1) * graph_size / 2.
            recall_elementWise = torch.cat([torch.sum(indices[:, (i+1):]>indices[:, i].view(indices.shape[0], -1), dim=1).view(indices.shape[0], -1) for i in range(indices.shape[1]-1)], dim=1)
            recall = torch.sum(recall_elementWise, dim=1) 
            recall_accuracy_max, recall_accuracy_min = torch.max(recall) / full_recall, torch.min(recall) / full_recall
            recall_accuracy = torch.mul(recall, torch.FloatTensor([1.]).cuda()).mean() / full_recall
            
            #idx = torch.Tensor([i for i in range(int(list(y_.shape)[1]))]).cuda().repeat(list(y_.shape)[0]).view(d.shape[0],d.shape[1])
            #idx = torch.tensor([i for i in range(indices.shape[1])]).cuda()
            radius_elementWise = torch.abs(indices - idx)
            radius_mean = torch.mean(torch.mul(radius_elementWise, torch.FloatTensor([1.]).cuda()), 1).mean()
            radius_max = torch.max(radius_elementWise)
            """
            #recall_accuracy, radius_mean, radius_max, recall_accuracy_max, recall_accuracy_min = None, None, None, None, None 

            #misMatch_y = torch.nonzero(misMatch_y).shape[0] / y_.shape[0]
            #misMatch_x = torch.nonzero(misMatch_x).shape[0] / idx.shape[0]
            misMatch_y = torch.FloatTensor([torch.nonzero(torch.sub(sorted, y_)).shape[0] / y_.shape[0]]).cuda()
            misMatch_x = torch.FloatTensor([torch.nonzero(torch.sub(indices, idx)).shape[0] / idx.shape[0]]).cuda()
            #misMatch_y = None
            #misMatch_x = None
            #recall_accuracy, radius_mean, radius_max = None, None, None 
        else:
            misMatch_y, misMatch_x, recall_accuracy, radius_mean, radius_max, recall_accuracy_max, recall_accuracy_min = None, None, None, None, None, None, None
        #print(cost.shape)
        return 1-cost, None, misMatch_y, misMatch_x, recall_accuracy, radius_mean, radius_max, recall_accuracy_max, recall_accuracy_min 
        #return cost.cuda(), None, misMatch_y, misMatch_x, recall_accuracy, radius_mean, radius_max, recall_accuracy_max, recall_accuracy_min 

    @staticmethod
    def make_dataset(*args, **kwargs):
        return TopoSortDataset(*args, **kwargs)

    @staticmethod
    def make_state(*args, **kwargs):
        return StateTopoSort.initialize(*args, **kwargs)

    @staticmethod
    def beam_search(input, beam_size, expand_size=None,
                    compress_mask=False, model=None, max_calc_batch_size=4096):

        assert model is not None, "Provide model"

        fixed = model.precompute_fixed(input)

        def propose_expansions(beam):
            return model.propose_expansions(
                beam, fixed, expand_size, normalize=True, max_calc_batch_size=max_calc_batch_size
            )

        #state = TSP.make_state(
        state = TopoSort.make_state(
            #input, visited_dtype=torch.int64 if compress_mask else torch.uint8
            input, visited_dtype=torch.int64 if compress_mask else torch.bool
        )

        return beam_search(state, beam_size, propose_expansions)


class TopoSortDataset(Dataset):
    # 50, 1000000 
    def __init__(self, filename=None, size=25, num_samples=1000, offset=0, distribution=None, seed=0):
        super(TopoSortDataset, self).__init__()

        self.data_set = []
        if filename is not None:
            assert os.path.splitext(filename)[1] == '.pkl'

            with open(filename, 'rb') as f:
                data = pickle.load(f)
                self.data = [torch.FloatTensor(row) for row in (data[offset:offset+num_samples])]

        else:
            # Sample points randomly in [0, 1] square
            #self.data = [torch.FloatTensor(size, 2).uniform_(0, 1) for i in range(num_samples)]
            self.data = []
            if seed > 0:
                random.seed(seed)
            for _ in range(num_samples):
                graph = []
                levels = random.randint(2, 20)
                #levels = random.randint(1, 25)
                #levels = random.randint(50, 100)
                num_level = size // levels
                for level in range(levels-1):
                    graph.append([level+1, random.randint(0, level), random.randint(0, level), random.randint(0, level)])
                    y_axis = random.random()
                    for j in range(num_level):
                        graph.append([random.random(), y_axis])
                y_axis = random.random()
                for k in range(num_level+size%levels):
                    graph.append([random.random(), y_axis])
                graph.sort(key = lambda i: i[0]**2-i[1]**2)
                self.data.append(torch.FloatTensor(graph))

                #for i in range(size):
                    #graph.append([i,i+random.randint(1,10)])
                #print(torch.FloatTensor(graph).shape)
                #self.data.append(torch.nn.functional.normalize(torch.FloatTensor(graph)))

        self.size = len(self.data)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.data[idx]
