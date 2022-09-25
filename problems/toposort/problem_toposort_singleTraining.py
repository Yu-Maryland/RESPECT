from torch.utils.data import Dataset
import torch, random
import os
import pickle
from problems.toposort.state_toposort import StateTopoSort
from utils.beam_search import beam_search
#from utils import orderCheck, deep_sort_x, level_sorting, level_sorting_xy_pairs, order_check
from utils import smart_sort

import networkx as nx
import numpy as np

class TopoSort(object):

    NAME = 'toposort'

    @staticmethod
    def get_costs(dataset, pi, labels, measures=False, plot_data=False, graph_name=None):
    #def get_costs(dataset, pi, measures=False, plot_data=False):
        # Gather dataset in order of graph nodes
        #d = dataset.gather(1, pi.unsqueeze(-1).expand_as(dataset))
        order_learned = smart_sort(labels, pi, dim=2)
        #order_sorted, indices = torch.sort(labels, dim=1, descending=True)
        order_sorted, indices = torch.sort(labels, dim=1)

        order_learned = order_learned.cuda()
        order_sorted = order_sorted.cuda()
        cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
        cost = cos(order_learned, order_sorted).cuda()

        if plot_data: 
            dataset_learning_sequence = dataset.gather(1, pi.unsqueeze(-1).expand_as(dataset))
            label_learning_sequence = dataset_learning_sequence[:,:,-2].view(dataset_learning_sequence.shape[0], dataset_learning_sequence.shape[1])
            dataset_labeling_sequence = dataset.gather(1, indices.unsqueeze(-1).expand_as(dataset))

            #layers_order = d[:,:,0].view(d.shape[0], d.shape[1])
            file = open(r"graph_data_collection/graph_data_collection_adaptive_training/" + graph_name, "a")
            #file = open(r"graph_data_collection/graph_data_collection_validation/" + graph_name, "a")
            #file = open(r"graph_data_collection/graph30_50_128k_w35_35_weightFree.txt", "a")
            torch.set_printoptions(profile="full")
            for i in range(dataset.shape[0]):
                file.write("real sequence is:\n")
                #file.writelines([str(sequence.cpu().numpy())+"\n" for sequence in dataset_labeling_sequence[i]])
                file.write(str(dataset_labeling_sequence[i]) + "\n")
                file.write("learning sequence is:\n")
                #file.write(str(dataset_learning_sequence[i]) + "\n")
                file.write(str(label_learning_sequence[i]) + "\n")
                file.write("Nodes Level Distribution:\n")
                file.write(str(order_sorted[i]) + "\n")
                file.write("end\n")
            file.close()
            
            #print(layers_order[-1].cpu())
            #for element in order_learned[-1].cpu():
            #    print(element)
            #print("end of batch")

        # generated index compared to optimal index; to be used for cost function below
        #order_learned = d[:,:,0].view(d.shape[0],d.shape[1])
        #print(d.shape, list(y_.shape), ((d[:, 1:] - d[:, :-1]).norm(p=2, dim=2).sum(1) + (d[:, 0] - d[:, -1]).norm(p=2, dim=1)).shape)
        #order_sorted, indices = torch.sort(order_learned, dim=1)

        #New cost stragety to satisfy level sorting
        #cost = level_sorting(sorted, y_)
        #cost, indices = level_sorting_xy_pairs(indices, d)
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
        #misMatch_y = torch.sub(y_, sorted)
        #misMatch_x = torch.sub(indices, idx)
        #cost = 0.2 * torch.count_nonzero(misMatch_x, dim=1) + 0.8 * torch.count_nonzero(misMatch_y, dim=1)
        #order_learned = order_learned.cuda()
        #order_sorted = order_sorted.cuda()
        #cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
        #cost = cos(order_learned, order_sorted).cuda()
        if measures:
            recall_elementWise = torch.cat([torch.sum(order_learned[:, (i+1):] > order_learned[:, i].view(order_learned.shape[0], -1), dim=1).view(order_learned.shape[0], -1) for i in range(order_learned.shape[1]-1)], dim=1)

            full_recall_elementWise = torch.cat([torch.sum(order_sorted[:, (i+1):] > order_sorted[:, i].view(order_sorted.shape[0], -1), dim=1).view(order_sorted.shape[0], -1) for i in range(order_sorted.shape[1]-1)], dim=1)

            recall = torch.sum(recall_elementWise.cuda(), dim=1) 
            full_recall = torch.sum(full_recall_elementWise.cuda(), dim=1) 
            recall_accuracy = torch.div(recall.cuda(), full_recall.cuda())
            recall_accuracy_mean, recall_accuracy_max, recall_accuracy_min = recall_accuracy.mean(), torch.max(recall_accuracy), torch.min(recall_accuracy)

            #diff = torch.abs(torch.sub(order_learned, order_sorted))

            misMatch = torch.FloatTensor([torch.nonzero(torch.sub(order_learned, order_sorted)).shape[0] / indices.shape[0]]).cuda()
            
            radius_mean, radius_max = None, None

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
            #misMatch_y = torch.FloatTensor([torch.nonzero(torch.sub(sorted, y_)).shape[0] / y_.shape[0]]).cuda()
            #misMatch_x = torch.FloatTensor([torch.nonzero(torch.sub(indices, idx)).shape[0] / idx.shape[0]]).cuda()
            #misMatch_y = None
            #misMatch_x = None
            #recall_accuracy, radius_mean, radius_max = None, None, None 
        else:
            #misMatch_y, misMatch_x, recall_accuracy, radius_mean, radius_max, recall_accuracy_max, recall_accuracy_min = None, None, None, None, None, None, None
            misMatch, recall_accuracy_mean, radius_mean, radius_max, recall_accuracy_max, recall_accuracy_min = None, None, None, None, None, None

        return 1-cost, None, misMatch, None, recall_accuracy_mean, radius_mean, radius_max, recall_accuracy_max, recall_accuracy_min 
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
                #levels = random.randint(2, 10)
                #levels = random.randint(1, 25)
                levels = random.randint(800, 1000)

                level = [1 for _ in range(levels)]
                remaining = size - levels
                traverse = 0
                while remaining > 0:
                    addition = random.randint(0, remaining)
                    level[traverse % levels] += addition
                    traverse += 1
                    remaining -= addition
                caution = [i+1 for i, val in enumerate(level) if val < 3]
                 
                #num_level = size // levels
                for i in range(levels):
                    for j in range(level[i]):
                        embedding = [i+1, random.randint(0, i), random.randint(0, i), random.randint(0, i)]
                        if max(embedding[1:]) < i:
                            embedding[random.randint(1, 3)] = i 

                        for constraint in caution:
                            while embedding[1:].count(constraint) > level[constraint-1]:
                                embedding[embedding.index(constraint)] = -1 
                        graph.append(embedding)
                order = [i for i in range(size)]
                random.shuffle(order)
                graph = [graph[i] for i in order]
                #graph.sort(key = lambda i: i[0]**2-i[1]**2+i[2]**2-i[3]**2)
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
