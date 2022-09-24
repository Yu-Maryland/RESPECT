from torch.utils.data import Dataset
import torch, random
import os
import pickle
#from problems.toposort.state_toposort import StateTopoSort
#from utils.beam_search import beam_search
#from utils import orderCheck, deep_sort_x, level_sorting, level_sorting_xy_pairs, order_check, graph_sorting_DAG

from collections import defaultdict
from itertools import combinations
import networkx as nx
import networkx.algorithms.isomorphism as iso

import numpy as np
from torch.utils.data import DataLoader, SubsetRandomSampler
from tqdm import tqdm

class TopoSortDataset(Dataset):
    # 50, 1000000 
    def __init__(self, filename=None, size=25, num_samples=1000, offset=0, in_degree_fixed=10, resource_constraint_level={-1:4}, level_range=[4, 10], weight_multiply=10, weight_constraint=15, shift=0., distribution=None, seed=0):
        super(TopoSortDataset, self).__init__()

        self.data = []
        #self.label = []
        if filename is not None:
            assert os.path.splitext(filename)[1] == '.pkl'

            with open(filename, 'rb') as f:
                data = pickle.load(f)
                self.data = [torch.FloatTensor(row) for row in (data[offset:offset+num_samples])]
        else:
            self.label = []
            graphs_collection = []
            if seed > 0:
               random.seed(seed)
            for _ in range(num_samples):
                graph, D = self._dag_generator(size, in_degree_fixed, len(resource_constraint_level.keys()), level_range, weight_multiply, shift)

                schedule = self._scheduling(D, D.number_of_nodes(), resource_constraint_level, weight_constraint)

                level_index = [0 for _ in range(size)]
                for i in range(len(schedule)):
                    for node in schedule[i]:
                        level_index[node] = i+1
                
                order = [i for i in range(size)]
                random.shuffle(order)
   
                label_new = []
                graph_new = []
                for i in order:
                    embedding = graph[i]
                    graph_new.append(embedding)
                    label_new.append(level_index[embedding[-2]])

                self.data.append(torch.FloatTensor(graph_new))
                self.label.append(torch.FloatTensor(label_new))

        self.size = len(self.data)
   
    def _embedding_generator(self, size, in_degree_fixed, level_range, weight_multiply, shift):
        num_level = random.randint(level_range[0], level_range[1])
        level = [1 for _ in range(num_level)]
        remaining = size - num_level
        traverse = 0
        while remaining > 0:
            addition = random.randint(0, remaining)
            level[traverse % num_level] += addition
            traverse += 1
            remaining -= addition
        caution = [i+1 for i, val in enumerate(level) if val < in_degree_fixed]

        labels = [i for i in range(size)]
        random.shuffle(labels)
        level_to_nodes = defaultdict(int)
        labels_distribution = []
        distributor = 0
        for i in range(num_level):
            level_to_nodes[i+1] = labels[distributor:(distributor+level[i])]
            labels_distribution.append(labels[distributor:(distributor+level[i])])
            distributor += level[i]
                
        graph = []
        graph_edges = []
        for i in range(num_level):
            for j in range(level[i]):
                embedding = [i+1]
                embedding.extend([random.randint(0, i) for _ in range(in_degree_fixed)])
                if max(embedding[1:]) < i:
                    embedding[random.randint(1, in_degree_fixed)] = i
                for constraint in caution:
                    while embedding[1:].count(constraint) > level[constraint-1]:
                        embedding[embedding.index(constraint)] = 0
                nodes_to_be_assigned = embedding[1:]
                while len(nodes_to_be_assigned) > 0:
                    node = nodes_to_be_assigned[0]
                    occurrences = nodes_to_be_assigned.count(node)
                    if node > 0:
                        embedding.extend(random.sample(level_to_nodes[node], occurrences))
                    else:
                        embedding.extend([-1 for _ in range(occurrences)])
                    nodes_to_be_assigned = [element for element in nodes_to_be_assigned if element != node]
               
                label_to_current_node = random.sample(labels_distribution[i], 1)
                labels_distribution[i].remove(label_to_current_node[0])
                embedding.append(label_to_current_node[0])

                embedding.append(random.random() * weight_multiply + shift)

                graph.append(embedding)

                if i > 0:
                    for predecessor in embedding[-2-in_degree_fixed:-2]:
                        if predecessor > -1:
                            graph_edges.append((predecessor, label_to_current_node[0]))        
        G = nx.DiGraph()
        G.add_edges_from(graph_edges)
        return graph, G

    def _dag_generator(self, size, in_degree_fixed, num_operators, level_range, weight_multiply, shift):
        while True:
            graph, D = self._embedding_generator(size, in_degree_fixed, level_range, weight_multiply, shift)
            if nx.is_connected(D.to_undirected()) and nx.is_directed_acyclic_graph(D) and D.number_of_nodes() == size:
                break

        attributes = {graph[i][-2]:{'operator':random.randint(-num_operators, -1), 'priority':0, 'label':graph[i][-2], 'weight':graph[i][-1]} for i in range(size)}
        nx.set_node_attributes(D, attributes)
        DAG = self._priority_sorting(D)
       
        return graph, DAG

    def _scheduling(self, D, size, resource_constraint_level, weight_constraint):
        DAG = D.reverse(copy=True)
        op = DAG.nodes[0]['operator']
        resource_constraint = resource_constraint_level[op]
        def path_exploring(resource_constraint, weight_constraint):
            if DAG.number_of_nodes() <= 0:
                return []
            candidates = sorted([n for n, d in DAG.in_degree() if d==0], key=lambda x: (DAG.nodes[x]['priority'], -x), reverse=True)
            schedule = candidates[:resource_constraint]
            while True:
                weight_in_all = sum([DAG.nodes[node]['weight'] for node in schedule])
                if weight_in_all <= weight_constraint:
                    break
                schedule.pop()                
            DAG.remove_nodes_from(schedule) 
            return [schedule] + path_exploring(resource_constraint, weight_constraint)
        return path_exploring(resource_constraint, weight_constraint)

    """
    def _scheduling(self, D, size, resource_constraint_level, weight_constraint):
        DAG = D.reverse(copy=True)
        candidates = [head for head in range(size) if DAG.in_degree(head) == 0]
        candidates.sort(key = lambda x: (DAG.nodes[x]['priority'], -x), reverse=True)
        op = DAG.nodes[candidates[0]]['operator']
        resource_constraint = resource_constraint_level[op]
        def path_exploring(candidates, resource_constraint, weight_constraint):
            if len(candidates) <= 0:
                return []
            schedule = candidates[:resource_constraint]
            while True:
                weight_in_all = sum([D.nodes[node]['weight'] for node in schedule])
                if weight_in_all <= weight_constraint:
                    break
                schedule.pop()                
            candidates = set(candidates[len(schedule):])
            for node in schedule:
                candidates = candidates.union(set(DAG.successors(node)))
            return [schedule] + path_exploring(sorted(list(candidates), key=lambda x: (DAG.nodes[x]['priority'], -x), reverse=True), resource_constraint, weight_constraint)
        return path_exploring(candidates, resource_constraint, weight_constraint)
    """         
    def _priority_sorting(self, D):
        DAG = D.reverse(copy=True)
        leaf = set([node for node in range(DAG.number_of_nodes()) if DAG.out_degree(node) == 0])
        def priority_rewrite(nodes, level):
            visited = set()
            for node in nodes:
                if DAG.nodes[node]['priority'] >= level:
                    continue
                DAG.nodes[node]['priority'] = level
                upper_level = set(DAG.predecessors(node)).difference(visited)
                priority_rewrite(upper_level, level+1)
                visited = visited.union(upper_level)
            return
        priority_rewrite(leaf, 1)
        return DAG.reverse(copy=True)
            
    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        #return self.data[idx], self.label[idx]
        return self.data[idx], self.label[idx]

if __name__ == '__main__':

    #resource_constraint_lvl = {-1:1, -2:1, -3:1}
    #resource_constraint_lvl = {-1:3}

    training_lvl = {-1:5000}

    eval_lvl = {-1:5000}

    """
    myDataset = TopoSortDataset(size=30, num_samples=1000, in_degree_fixed=4, resource_constraint_level=eval_lvl, level_range=[10, 25], weight_multiply=375., shift=25., weight_constraint=1000.)
    torch.save(myDataset, "training_dataset/operator_type_1/small_volume_adaptive_learning/TopoSort30_Dataset_4_in_degree_3resource_priorityInverse_1K_cw25to400_weight1000.pt") 

    myDataset = TopoSortDataset(size=30, num_samples=1000, in_degree_fixed=4, resource_constraint_level=eval_lvl, level_range=[10, 25], weight_multiply=275., shift=25., weight_constraint=1000.)
    torch.save(myDataset, "training_dataset/operator_type_1/small_volume_adaptive_learning/TopoSort30_Dataset_4_in_degree_3resource_priorityInverse_1K_cw25to300_weight1000.pt") 

    myDataset = TopoSortDataset(size=30, num_samples=1000, in_degree_fixed=4, resource_constraint_level=eval_lvl, level_range=[10, 25], weight_multiply=175., shift=25., weight_constraint=1000.)
    torch.save(myDataset, "training_dataset/operator_type_1/small_volume_adaptive_learning/TopoSort30_Dataset_4_in_degree_3resource_priorityInverse_1K_cw25to200_weight1000.pt") 

    myDataset = TopoSortDataset(size=30, num_samples=1000, in_degree_fixed=4, resource_constraint_level=eval_lvl, level_range=[10, 25], weight_multiply=75., shift=25., weight_constraint=1000.)
    torch.save(myDataset, "training_dataset/operator_type_1/small_volume_adaptive_learning/TopoSort30_Dataset_4_in_degree_3resource_priorityInverse_1K_cw25to100_weight1000.pt") 

    myDataset = TopoSortDataset(size=50, num_samples=10240, in_degree_fixed=4, resource_constraint_level=eval_lvl, level_range=[16, 40], weight_multiply=375., shift=25., weight_constraint=1000.)
    torch.save(myDataset, "eval_dataset/operator_type_1/validation_set/TopoSort50_Dataset_Eval_4_in_degree_3resource_priorityInverse_10K_cw25to400_weight1000.pt") 

    myDataset = TopoSortDataset(size=50, num_samples=10240, in_degree_fixed=4, resource_constraint_level=eval_lvl, level_range=[16, 40], weight_multiply=275., shift=25., weight_constraint=1000.)
    torch.save(myDataset, "eval_dataset/operator_type_1/validation_set/TopoSort50_Dataset_Eval_4_in_degree_3resource_priorityInverse_10K_cw25to300_weight1000.pt") 

    myDataset = TopoSortDataset(size=50, num_samples=10240, in_degree_fixed=4, resource_constraint_level=eval_lvl, level_range=[16, 40], weight_multiply=175., shift=25., weight_constraint=1000.)
    torch.save(myDataset, "eval_dataset/operator_type_1/validation_set/TopoSort50_Dataset_Eval_4_in_degree_3resource_priorityInverse_10K_cw25to200_weight1000.pt") 

    myDataset = TopoSortDataset(size=50, num_samples=10240, in_degree_fixed=4, resource_constraint_level=eval_lvl, level_range=[16, 40], weight_multiply=75., shift=25., weight_constraint=1000.)
    torch.save(myDataset, "eval_dataset/operator_type_1/validation_set/TopoSort50_Dataset_Eval_4_in_degree_3resource_priorityInverse_10K_cw25to100_weight1000.pt") 
    """
    #myDataset = TopoSortDataset(size=30, num_samples=1000, in_degree_fixed=4, resource_constraint_level=eval_lvl, level_range=[10, 25], weight_multiply=9.5, shift=0.5, weight_constraint=35)
    #torch.save(myDataset, "training_dataset/operator_type_1/small_volume_adaptive_learning/TopoSort30_Dataset_4_in_degree_3resource_priorityInverse_1K_cw05to10_weight35.pt") 

    #myDataset = TopoSortDataset(size=30, num_samples=1000, in_degree_fixed=4, resource_constraint_level=eval_lvl, level_range=[10, 25], weight_multiply=475., shift=25., weight_constraint=1000)
    #torch.save(myDataset, "training_dataset/operator_type_1/small_volume_adaptive_learning/TopoSort30_Dataset_4_in_degree_3resource_priorityInverse_1K_cw25to500_weight1000.pt") 

    #myDataset = TopoSortDataset(size=10, num_samples=5, in_degree_fixed=4, resource_constraint_level=resource_constraint_lvl, level_range=[4, 10], weight_multiply=10, shift=1., weight_constraint=15)
    #torch.save(myDataset, "eval_dataset/operator_type_1/TopoSort15_Dataset_Eval_3_in_degree_5samples.pt") 
    #torch.save(myDataset, "training_dataset/operator_type_1/TopoSort10_Dataset_Training_4_in_degree_10samples.pt") 

    myDataset = TopoSortDataset(size=30, num_samples=128000, in_degree_fixed=6, resource_constraint_level=training_lvl, level_range=[10, 25], weight_multiply=5., weight_constraint=35.)
    #myDataset = TopoSortDataset(size=30, num_samples=100, in_degree_fixed=4, resource_constraint_level=training_lvl, level_range=[10, 25], weight_multiply=5, weight_constraint=5)
    torch.save(myDataset, "training_dataset/operator_type_1/TopoSort30_Dataset_Training_6_in_degree_3resource_priorityInverse_128k_10to25_weight35.pt") 

    myDataset = TopoSortDataset(size=50, num_samples=10240, in_degree_fixed=6, resource_constraint_level=eval_lvl, level_range=[16, 40], weight_multiply=5., weight_constraint=35.)
    torch.save(myDataset, "eval_dataset/operator_type_1/TopoSort50_Dataset_Eval_6_in_degree_3resource_priorityInverse_10K_16to40_weight35.pt") 
    #myDataset = TopoSortDataset(size=50, num_samples=128000, in_degree_fixed=4, resource_constraint_level=training_lvl, level_range=[20, 40], weight_multiply=5, weight_constraint=5)
    #myDataset = TopoSortDataset(size=30, num_samples=100, in_degree_fixed=4, resource_constraint_level=training_lvl, level_range=[10, 25], weight_multiply=5, weight_constraint=5)
    #torch.save(myDataset, "training_dataset/operator_type_1/TopoSort50_Dataset_Training_4_in_degree_3resource_priorityInverse_128k_20to40_weight5.pt") 

    #myDataset = TopoSortDataset(size=20, num_samples=128000, in_degree_fixed=3, resource_constraint_level=training_lvl)
    #torch.save(myDataset, "training_dataset/operator_type_1/small_volume/TopoSort20_Dataset_Training_3_in_degree_1resources_priorityInverse_128K_nonRepeated.pt") 
    #eval_lvl = {-1:3}
    #myDataset = TopoSortDataset(size=20, num_samples=10240, in_degree_fixed=3, resource_constraint_level=eval_lvl)
    #torch.save(myDataset, "eval_dataset/operator_type_1/small_volume/TopoSort20_Dataset_Eval_3_in_degree_1resources_priorityInverse_10K_nonRepeated.pt") 
    """
    myDataset = TopoSortDataset(size=50, num_samples=10240, in_degree_fixed=4, resource_constraint_level=eval_lvl, level_range=[16, 40], weight_multiply=9.5, shift=0.5, weight_constraint=35)
    torch.save(myDataset, "eval_dataset/operator_type_1/validation_set/TopoSort50_Dataset_Eval_4_in_degree_3resource_priorityInverse_10K_cw05to10_weight35.pt") 

    myDataset = TopoSortDataset(size=100, num_samples=10240, in_degree_fixed=4, resource_constraint_level=eval_lvl, level_range=[16*2, 80], weight_multiply=9.5, shift=0.5, weight_constraint=35)
    torch.save(myDataset, "eval_dataset/operator_type_1/validation_set/TopoSort100_Dataset_Eval_4_in_degree_3resource_priorityInverse_10K_cw05to10_weight35.pt") 

    myDataset = TopoSortDataset(size=200, num_samples=10240, in_degree_fixed=4, resource_constraint_level=eval_lvl, level_range=[64, 160], weight_multiply=9.5, shift=0.5, weight_constraint=35)
    torch.save(myDataset, "eval_dataset/operator_type_1/validation_set/TopoSort200_Dataset_Eval_4_in_degree_3resource_priorityInverse_10K_cw05to10_weight35.pt") 

    myDataset = TopoSortDataset(size=300, num_samples=10240, in_degree_fixed=4, resource_constraint_level=eval_lvl, level_range=[96, 240], weight_multiply=9.5, shift=0.5, weight_constraint=35)
    torch.save(myDataset, "eval_dataset/operator_type_1/validation_set/TopoSort300_Dataset_Eval_4_in_degree_3resource_priorityInverse_10K_cw05to10_weight35.pt") 

    myDataset = TopoSortDataset(size=400, num_samples=10240, in_degree_fixed=4, resource_constraint_level=eval_lvl, level_range=[128, 320], weight_multiply=9.5, shift=0.5, weight_constraint=35)
    torch.save(myDataset, "eval_dataset/operator_type_1/validation_set/TopoSort400_Dataset_Eval_4_in_degree_3resource_priorityInverse_10K_cw05to10_weight35.pt") 

    myDataset = TopoSortDataset(size=500, num_samples=10240, in_degree_fixed=4, resource_constraint_level=eval_lvl, level_range=[160, 400], weight_multiply=9.5, shift=0.5, weight_constraint=35)
    torch.save(myDataset, "eval_dataset/operator_type_1/validation_set/TopoSort500_Dataset_Eval_4_in_degree_3resource_priorityInverse_10K_cw05to10_weight35.pt") 

    myDataset = TopoSortDataset(size=50, num_samples=1000, in_degree_fixed=4, resource_constraint_level=eval_lvl, level_range=[16, 40], weight_multiply=1.95, shift=0.05, weight_constraint=8)
    torch.save(myDataset, "training_dataset/operator_type_1/small_volume_adaptive_learning/TopoSort50_Dataset_4_in_degree_3resource_priorityInverse_1K_cw005to2_weight8.pt") 

    myDataset = TopoSortDataset(size=100, num_samples=1000, in_degree_fixed=4, resource_constraint_level=eval_lvl, level_range=[16*2, 80], weight_multiply=1.95, shift=0.05, weight_constraint=8)
    torch.save(myDataset, "training_dataset/operator_type_1/small_volume_adaptive_learning/TopoSort100_Dataset_4_in_degree_3resource_priorityInverse_1K_cw005to2_weight8.pt") 

    myDataset = TopoSortDataset(size=200, num_samples=1000, in_degree_fixed=4, resource_constraint_level=eval_lvl, level_range=[64, 160], weight_multiply=1.95, shift=0.05, weight_constraint=8)
    torch.save(myDataset, "training_dataset/operator_type_1/small_volume_adaptive_learning/TopoSort200_Dataset_4_in_degree_3resource_priorityInverse_1K_cw005to2_weight8.pt") 

    myDataset = TopoSortDataset(size=300, num_samples=1000, in_degree_fixed=4, resource_constraint_level=eval_lvl, level_range=[96, 240], weight_multiply=1.95, shift=0.05, weight_constraint=8)
    torch.save(myDataset, "training_dataset/operator_type_1/small_volume_adaptive_learning/TopoSort300_Dataset_4_in_degree_3resource_priorityInverse_1K_cw005to2_weight8.pt") 

    myDataset = TopoSortDataset(size=400, num_samples=1000, in_degree_fixed=4, resource_constraint_level=eval_lvl, level_range=[128, 320], weight_multiply=1.95, shift=0.05, weight_constraint=8)
    torch.save(myDataset, "training_dataset/operator_type_1/small_volume_adaptive_learning/TopoSort400_Dataset_4_in_degree_3resource_priorityInverse_1K_cw005to2_weight8.pt") 

    myDataset = TopoSortDataset(size=500, num_samples=1000, in_degree_fixed=4, resource_constraint_level=eval_lvl, level_range=[160, 400], weight_multiply=1.95, shift=0.05, weight_constraint=8)
    torch.save(myDataset, "training_dataset/operator_type_1/small_volume_adaptive_learning/TopoSort500_Dataset_4_in_degree_3resource_priorityInverse_1K_cw005to2_weight8.pt") 

    myDataset = TopoSortDataset(size=50, num_samples=10240, in_degree_fixed=4, resource_constraint_level=eval_lvl, level_range=[16, 40], weight_multiply=475., shift=25., weight_constraint=1000)
    torch.save(myDataset, "eval_dataset/operator_type_1/validation_set/TopoSort50_Dataset_Eval_4_in_degree_3resource_priorityInverse_10K_cw25to500_weight1000.pt") 

    myDataset = TopoSortDataset(size=100, num_samples=10240, in_degree_fixed=4, resource_constraint_level=eval_lvl, level_range=[32, 80], weight_multiply=475., shift=25., weight_constraint=1000)
    torch.save(myDataset, "eval_dataset/operator_type_1/validation_set/TopoSort100_Dataset_Eval_4_in_degree_3resource_priorityInverse_10K_cw25to500_weight1000.pt") 

    myDataset = TopoSortDataset(size=200, num_samples=10240, in_degree_fixed=4, resource_constraint_level=eval_lvl, level_range=[64, 160], weight_multiply=475., shift=25., weight_constraint=1000)
    torch.save(myDataset, "eval_dataset/operator_type_1/validation_set/TopoSort200_Dataset_Eval_4_in_degree_3resource_priorityInverse_10K_cw25to500_weight1000.pt") 

    myDataset = TopoSortDataset(size=300, num_samples=10240, in_degree_fixed=4, resource_constraint_level=eval_lvl, level_range=[96, 240], weight_multiply=475., shift=25., weight_constraint=1000)
    torch.save(myDataset, "eval_dataset/operator_type_1/validation_set/TopoSort300_Dataset_Eval_4_in_degree_3resource_priorityInverse_10K_cw25to500_weight1000.pt") 

    myDataset = TopoSortDataset(size=400, num_samples=10240, in_degree_fixed=4, resource_constraint_level=eval_lvl, level_range=[128, 320], weight_multiply=475., shift=25., weight_constraint=1000)
    torch.save(myDataset, "eval_dataset/operator_type_1/validation_set/TopoSort400_Dataset_Eval_4_in_degree_3resource_priorityInverse_10K_cw25to500_weight1000.pt") 

    myDataset = TopoSortDataset(size=500, num_samples=10240, in_degree_fixed=4, resource_constraint_level=eval_lvl, level_range=[160, 400], weight_multiply=475., shift=25., weight_constraint=1000)
    torch.save(myDataset, "eval_dataset/operator_type_1/validation_set/TopoSort500_Dataset_Eval_4_in_degree_3resource_priorityInverse_10K_cw25to500_weight1000.pt") 
    """

    #for i in range(1000, 5001, 1000):
    #    myDataset = TopoSortDataset(size=i, num_samples=10)

    #    torch.save(myDataset, "eval_large_size/TopoSort" + str(i) + "_Dataset_Validation_10_in_degree.pt") 

    #torch.save(myDataset, "TopoSort20_Dataset_Training_10_in_degree.pt") 

    #myDataset = torch.load("eval_dataset/operator_type_1/TopoSort50_Dataset_Eval_4_in_degree_3resource_priorityInverse_10K_20to40_weight5.pt") 
    #dataset = torch.load("eval_dataset/operator_type_1/TopoSort15_Dataset_Eval_3_in_degree_5samples.pt", map_location=torch.device('cuda'))
    #dataset = torch.load("training_dataset/operator_type_1/TopoSort10_Dataset_Training_4_in_degree_10samples.pt", map_location=torch.device('cuda')) 

    #indices = torch.randperm(len(myDataset))[:3]
    #training_dataloader = DataLoader(myDataset, batch_size=1, sampler=SubsetRandomSampler(indices))
    #print(indices)
    #training_dataloader = DataLoader(dataset, batch_size=5)

    #for batch_id, batch in enumerate(tqdm(training_dataloader)):
    #for batch in tqdm(training_dataloader):
        #print(batch_id)
        #print("training_data: ", batch[0])
        #print("training_label: ", batch[1])
