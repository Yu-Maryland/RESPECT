from torch.utils.data import Dataset
import torch, random
import os
import pickle
#from problems.toposort.state_toposort import StateTopoSort
#from utils.beam_search import beam_search
#from utils import orderCheck, deep_sort_x, level_sorting, level_sorting_xy_pairs, order_check, graph_sorting_DAG

from collections import defaultdict
import networkx as nx
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm

class TopoSortDataset(Dataset):
    # 50, 1000000 
    def __init__(self, filename=None, size=25, num_samples=1000, offset=0, in_degree_fixed=3, distribution=None, seed=0):
        super(TopoSortDataset, self).__init__()

        self.data = []
        #self.label = []
        if filename is not None:
            assert os.path.splitext(filename)[1] == '.pkl'

            with open(filename, 'rb') as f:
                data = pickle.load(f)
                self.data = [torch.FloatTensor(row) for row in (data[offset:offset+num_samples])]
        else:
            self.data = []
            if seed > 0:
               random.seed(seed)
            for _ in range(num_samples):
                D = None

                while True:
                    G = nx.gnp_random_graph(size, random.random())   
                    if nx.is_connected(G):
                        Graph = self._in_degree_modification(G, in_degree_fixed)
                        if nx.is_connected(Graph):
                            D = nx.DiGraph([(u, v) for u, v in Graph.edges()]) 
                            try:
                                if nx.is_directed_acyclic_graph(D) and max(D.in_degree(i) for i in range(size)) <= in_degree_fixed:
                                    break
                            except:
                                continue
    
                graph = self._embedding(D, size, in_degree_fixed) 
                
                order = [i for i in range(size)]
                random.shuffle(order)
                graph = [graph[i] for i in order]

                #self.data.append(torch.FloatTensor(sorted(graph, key=lambda x : x[0]**2- x[-1]**3)))
                self.data.append(torch.FloatTensor(graph))

                #self.label.append(torch.FloatTensor(list(nx.topological_sort(D))))

        self.size = len(self.data)

    def _embedding(self, DAG, size, in_degree_fixed):
        visited = defaultdict(int, {head : 1 for head in range(size) if DAG.in_degree(head) == 0})
        def embedding_method(nodes, level, in_degree_fixed, visited):
            if len(nodes) <= 0: return []
            nodes_processed = set(visited.keys())
            nodes_processing = set()
            for node in nodes:
                children = set(DAG.successors(node))
                for child in children:
                    parents = set(DAG.predecessors(child))
                    if parents.issubset(nodes_processed):
                        nodes_processing.add(child)

            embedding_nodes_current = [[level] + [0 for _ in range(in_degree_fixed)] for _ in range(len(nodes))]
            if level > 1:
                node_index = 0
                for node in nodes:
                    embedding_index = 1
                    for p in set(DAG.predecessors(node)):
                        if embedding_index > in_degree_fixed:
                            print("Nodes in degree out of permission")
                            return []
                        embedding_nodes_current[node_index][embedding_index] = visited[p]
                        embedding_index += 1
                    node_index += 1
        
            for node_processing in nodes_processing:
                visited[node_processing] = level+1

            return embedding_nodes_current + embedding_method(nodes_processing, level+1, in_degree_fixed, visited)
              
        return embedding_method(set(visited.keys()), 1, in_degree_fixed, visited) 
               
    def _in_degree_modification(self, G, in_degree_fixed):
        nodes_discon = set()
        node_in_degree = defaultdict(int)
        edges = []
        for u, v in G.edges():
            if node_in_degree[v] == in_degree_fixed:
                nodes_discon.add(u)
            else:
                node_in_degree[v] += 1
                edges.append((u, v)) 

        for node in node_in_degree:
            if len(nodes_discon) <= 0:
                break
            if node_in_degree[node] < in_degree_fixed:
                nodes_to_be_cleared = set()
                for node_loss in nodes_discon:
                    if node_loss != node and (node_loss, node) not in edges:
                        edges.append((node_loss, node))
                        nodes_to_be_cleared.add(node_loss)
                        node_in_degree[node] += 1
                        if node_in_degree[node] >= in_degree_fixed:
                            break
                nodes_discon = nodes_discon.difference(nodes_to_be_cleared)
        return nx.Graph(edges)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        #return self.data[idx], self.label[idx]
        return self.data[idx]

"""
if __name__ == '__main__':

    myDataset = TopoSortDataset(size=20, num_samples=128000)

    torch.save(myDataset, "TopoSort_Dataset_Training.pt") 
    #dataset = torch.load("TopoSort_Dataset_Training.pt", map_location=torch.device('cuda'))
    #training_dataloader = DataLoader(dataset, batch_size=10)

    #for batch_id, batch in enumerate(tqdm(training_dataloader)):
    #    print(batch_id)
    #    print("training_data: ", batch[0])
    #    print("training_label: ", batch[1])
"""


