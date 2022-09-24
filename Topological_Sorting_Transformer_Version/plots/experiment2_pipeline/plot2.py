import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import numpy as np
import random
import networkx as nx
from collections import defaultdict

import math

class cluster:
    def __init__(self):
        self._nodes = set()
        self._highest_lvl = math.inf

    def add_node(self, node):
        self._nodes.add(node)
        self._highest_lvl = min(self._highest_lvl, node[-1])

    def add_nodes(self, nodes):
        if len(nodes) > 0:
            for node_id, lvl in nodes:
                self.add_node((node_id, lvl))

    def merge_cluster(self, cls_target):
        self._nodes = self._nodes.union(cls_target.get_nodes())
        self._highest_lvl = min(self._highest_lvl, cls_target.get_level())

    def get_level(self):
        return self._highest_lvl
  
    def get_nodes(self):
        return self._nodes

    def get_size(self):
        return len(self._nodes)

    def clear(self):
        self._nodes.clear()
        self._highest_lvl = math.inf

def pipelining(node_attributes, weight_constraint=35):
    """
    Given node_attributes: [(level, weight, label), ...]
    weight_constraint: upper bound of every pipeline
    return: [[pipe1], [pipe2], [pipe3]]
    """
    pipelines = []
    temp = []
    weight = 0
    for node in node_attributes:
        weight += node[1]
        if weight <= weight_constraint:
            temp.append(node)
        else:
            pipelines.append(temp)
            temp = [node]
            weight = node[1]

    pipelines.append(temp) 
        
    return pipelines
        
def mispair_cal(nodes_current, node_input, nodes_rear_collections):
    """
    nodes_current: [pipeline_n]
    node_input: {node: {input1, input2, ...}, ...}  
    nodes_rear_collections: set([[pipeline_n+1], [pipeline_n+2], ...])
    return: 
          nodes_mis: set of nodes as input to nodes_current from nodes_rear_collections
          num_mispair_relationships: number of wrong directed edges from nodes_rear_collections to nodes_current
    """
    num_mispair_relationships = 0
    nodes_mis = set()
    for node_level_weight in nodes_current:
        node_id = node_level_weight[-1]
        check = node_input[node_id].intersection(nodes_rear_collections)
        num_mispair_relationships += len(check)
        nodes_mis = nodes_mis.union(check)
        
    return nodes_mis, num_mispair_relationships
        
def relationship_correcting(nodes_wrong_position, pipeline_node, idx):
    """
    nodes_wrong_position: set of nodes to be put in pipeline_node[idx]
    pipeline_node: [[pipeline_1], [pipeline_2], ...]
    idx: index showing the pipeline in processing
    return: 
         new pipeline with correcting relationship to pipeline_node[idx]
    """
    for i in range(idx+1, len(pipeline_node)):
        nodes_impact = list()
        for node in pipeline_node[i]:
            if node[-1] in nodes_wrong_position:
                nodes_impact.append(node)
        pipeline_node[i] = [element for element in pipeline_node[i] if element not in nodes_impact]
        pipeline_node[idx].extend(nodes_impact)
            
    return pipeline_node

def memory_adjust(pipeline_node, idx, weight_constraint=35):
    """
    pipeline_node: [[pipeline_1], [pipeline_2], ...] 
    idx: current pipeline in processing
    weight_constraint: memory upper bound
    return:
          list of node to be removed from pipeline_node[idx] to lower pipeline due to memory overflow
    """
    current_pipe = pipeline_node[idx]
    current_pipe.sort(key=lambda p : p[0])
    node_to_remove = []
    weight = 0
    for i in range(len(current_pipe)):  
        weight += current_pipe[i][1]
        if weight > weight_constraint:
            node_to_remove = current_pipe[i:]
            break
    return node_to_remove

def pipeline_correcting(pipeline_node, node_input, weight_constraint=35):
    """
    pipeline_node: [[pipeline_1], [pipeline_2], ...]
    node_input: {node:{input1, input_2}, ...}
    weight_constraint: memory upper bound
    return:
        adjusted pipeline_node to treat memory bound and relationship mispositioned
    """
    num_pipeline = len(pipeline_node)
    pipeline_overflow = []
    for i in range(num_pipeline):
        if i < num_pipeline - 1:
            nodes_behind = [set([n[-1] for n in pipeline_node[j]]) for j in range(i+1, num_pipeline)]
            nodes_wrong_position, _ = mispair_cal(pipeline_node[i], node_input, set().union(*nodes_behind))
            pipeline_node = relationship_correcting(nodes_wrong_position, pipeline_node, i) 

        nodes_to_push_down = memory_adjust(pipeline_node, i, weight_constraint)
        pipeline_node[i] = [element for element in pipeline_node[i] if element not in nodes_to_push_down]
        if i < num_pipeline-1:
            pipeline_node[i+1].extend(nodes_to_push_down)
        else:
            pipeline_overflow.extend(nodes_to_push_down)
    if len(pipeline_overflow) > 0:
            pipeline_node.append(pipeline_overflow)
    return pipeline_node
        
def memory_extracting(graph_file, layer_index):
    G = nx.read_gpickle(graph_file)
    node_memory = nx.get_node_attributes(G, 'param_bytes')
    layer = layer_index[0]
    memory = [0 for _ in range(len(layer))]
    for node in node_memory.keys():
        memory[layer.index(node)] = node_memory[node]
    return memory

def traverse_parent_to_level(node, node_input, node_to_pipeline, target_lvl):
    nodes_impacted = set()
    if node_to_pipeline[node] <= target_lvl:
        return nodes_impacted
    for parent in node_input[node]:
        nodes_impacted = nodes_impacted.union(traverse_parent_to_level(parent, node_input, node_to_pipeline, target_lvl))
    nodes_impacted.add((node, node_to_pipeline[node]))

    return nodes_impacted    

def children_aligned_adjustment(pipeline_node, node_output, node_input):
    node_to_pipeline = defaultdict(int)
    for lvl in range(len(pipeline_node)):
        for n in pipeline_node[lvl]:
            node_to_pipeline[n[-1]] = lvl
    
    cluster_collection = []
    for parent in node_output.keys():
        nodes_impacted = cluster()
        for child in node_output[parent]:
            nodes_impacted.add_node((child, node_to_pipeline[child]))        

        nodes_upward = set()
        for child, lvl in nodes_impacted.get_nodes():
            nodes_upward = nodes_upward.union(traverse_parent_to_level(child, node_input, node_to_pipeline, nodes_impacted.get_level()))

        nodes_impacted.add_nodes(nodes_upward)
        cluster_collection.append(nodes_impacted)

    pointer = 0 
    while pointer < len(cluster_collection) - 1:
        position_to_remove = None
        for i in range(pointer+1, len(cluster_collection)):
            if len(cluster_collection[pointer].get_nodes().intersection(cluster_collection[i].get_nodes())) > 0:
                target_nodes = None
                target_lvl = math.inf
                left_lvl, right_lvl = cluster_collection[pointer].get_level(), cluster_collection[i].get_level()
                if left_lvl > right_lvl:
                    target_nodes = cluster_collection[pointer].get_nodes()
                    target_lvl = right_lvl
                elif right_lvl > left_lvl:
                    target_nodes = cluster_collection[i].get_nodes()
                    target_lvl = left_lvl

                nodes_to_be_checked = set()
                if target_nodes is not None:
                    for node_id, node_lvl in target_nodes:
                        nodes_to_be_checked = nodes_to_be_checked.union(traverse_parent_to_level(node_id, node_input, node_to_pipeline, target_lvl))
                        
                cluster_collection[pointer].merge_cluster(cluster_collection[i])
                cluster_collection[pointer].add_nodes(nodes_to_be_checked)
                position_to_remove = i
                break
        if position_to_remove is not None:
            cluster_collection.pop(position_to_remove)
        else:
            pointer += 1
                    
    for cls in cluster_collection:
        nodes_to_replace = cls.get_nodes()                   
        lvl_target = cls.get_level()
        for node_id, node_lvl in nodes_to_replace:
            if node_lvl <= lvl_target: continue
            node_to_be_treated = None
            for node_candidate in pipeline_node[node_lvl]:
                if node_id == node_candidate[-1]:
                    node_to_be_treated = node_candidate
                    break
            pipeline_node[node_lvl].remove(node_to_be_treated)
            pipeline_node[lvl_target].append(node_to_be_treated)
            
    return pipeline_node                           
 
    """
        lvl_common = len(pipeline_node) - 1
        children_lvl = set()
        nodes_impact = [set([child, node_to_pipeline[child] for child in node_output[parent]])]
        for child in node_output[parent]:
            children_lvl.add((child, node_to_pipeline[child]))
            lvl_common = min(lvl_common, node_to_pipeline[child])
            
        
        for child, lvl in children_lvl:
            if lvl > lvl_common:
                node_replacement = None
                for node in pipeline_node[lvl]:
                    if node[-1] == child:
                        node_replacement = node
                        break
                pipeline_node[lvl].remove(node_replacement) 
                pipeline_node[lvl_common].append(node_replacement)
                
    return pipeline_node
    """
        
def node_pipeline_relationship(graph_file, layer_index, pipeline_node):
    """
    print the node lvl relationship
    """ 
 
    G = nx.read_gpickle(graph_file)

    model_idx_original = nx.get_node_attributes(G, 'idx')
    valid_layer_count = 0

    for layer in model_idx_original.keys():
        if model_idx_original[layer]:
            valid_layer_count += 1
    
    layer_pipeline = [None for _ in range(valid_layer_count)]
    
    for i in range(len(pipeline_node)):
        for node in pipeline_node[i]:
            layer = layer_index[node[-1]]
            if model_idx_original[layer]:
                layer_pipeline[model_idx_original[layer]['layer_idx']] = i
            
    print(layer_pipeline)
    return None
            
def remove_empty(pipeline_node):
    empty_idx = []
    for i in range(len(pipeline_node)):
        if len(pipeline_node[i]) <= 0:
            empty_idx.append(i)

    empty_idx.sort(reverse=True)
    for idx in empty_idx:
        pipeline_node.pop(idx)

    return pipeline_node
    
def data_treatment(path_file, graph_file, weight_constraint=35):
    learning_graph_embedding = []
    learning_graph_sequence = []
    layer_index = []

    graph_reading, learning_sequence_reading, layer_name_reading = False, False, False

    with open(path_file, 'rt') as myfile:
        for myline in myfile:
            if myline.startswith("learning dataset is:"):
                graph_reading = True
                node_embedding = []
                data = ""
            elif myline.startswith("learning sequence is:"):
                learning_sequence_reading = True
                data = ""
            elif myline.startswith("layer index corresponding:"):
                layer_name_reading = True
                data = ""
            else:
                if graph_reading:
                    data += myline.strip().replace('tensor([[', '').replace('[', '')
                    if data[-3:] == ']],':
                        temp_s = data.replace("]],", "").split(', ')
                        temp_n = [int(float(temp_s[i])) for i in range(len(temp_s)-1)]    
                        temp_n.append(float(temp_s[-1]))
                        node_embedding.append(temp_n)
                        learning_graph_embedding.append(node_embedding)
                        graph_reading = False
                    elif data[-2:] == '],':
                        temp_s = data.replace('],', '').split(', ')
                        temp_n = [int(float(temp_s[i])) for i in range(len(temp_s)-1)]    
                        temp_n.append(float(temp_s[-1]))
                        node_embedding.append(temp_n)
                        data = ""
                    else:
                        data += ' '
                elif learning_sequence_reading:
                    data += myline.strip().replace('tensor([', '')
                    if data[-1] == ')':
                        learning_graph_sequence.append([int(float(node)) for node in data.replace("], device='cuda:0')", "").split(', ')])
                        learning_sequence_reading = False
                    else:
                        data += ' '
                elif layer_name_reading:
                    temp = myline.strip().replace('[', '').replace(']', '').split(', ')
                    layer_index.append([s.replace("('", "").replace("',)", "") for s in temp])
                    layer_name_reading = False

    node_memory = memory_extracting(graph_file, layer_index)
    """
    print("graph_embedding_before: ") 
    for embedding in learning_graph_embedding:
        print(embedding)
    print("learning_sequence_before: ")
    for sequence in learning_graph_sequence:
        print(sequence)
    """

    learning_graph_embedding[0].reverse()  
    learning_graph_sequence[0].reverse()

    """
    print("graph_embedding_after: ") 
    for embedding in learning_graph_embedding:
        print(embedding)
    print("learning_sequence_after: ")
    for sequence in learning_graph_sequence:
        print(sequence)
    """

    data_embedding_extracting = []

    node_input = defaultdict(set)
    node_output = defaultdict(set)

    weight = 0
    for i in range(len(learning_graph_embedding)):
        for embedding in learning_graph_embedding[i]:
            #data_embedding_extracting.append((embedding[0], embedding[-1], embedding[-2])) # (level, weight, idx)
            data_embedding_extracting.append((embedding[0], node_memory[embedding[-2]], embedding[-2])) # (level, weight, idx)
            #node_input[embedding[-2]] = set([node_idx for node_idx in embedding[5:9] if node_idx != -1])
            node_input[embedding[-2]] = set([node_idx for node_idx in embedding[7:13] if node_idx != -1])

            for node in node_input[embedding[-2]]:
                node_output[node].add(embedding[-2])

            #node_input[embedding[-2]] = set([node_idx for node_idx in embedding[7:13] if node_idx != -1])
            weight += node_memory[embedding[-2]]
             
    pipeline_node = pipelining(data_embedding_extracting, weight_constraint)
        
    num_mispairs = 0
    nodes_mispair = []
    for idx in range(len(pipeline_node)-1):
        nodes_behind = [set([n[-1] for n in pipeline_node[i]]) for i in range(idx+1, len(pipeline_node))]
        nodes_wrong_position, num_relationships = mispair_cal(pipeline_node[idx], node_input, set().union(*nodes_behind))
        num_mispairs += num_relationships
        nodes_mispair.append(nodes_wrong_position)
    
    num_mispairs_record = num_mispairs

    while num_mispairs > 0:
        pipeline_node = pipeline_correcting(pipeline_node, node_input, weight_constraint)
        num_mispairs = 0
        for idx in range(len(pipeline_node)-1):
            nodes_behind = [set([n[-1] for n in pipeline_node[i]]) for i in range(idx+1, len(pipeline_node))]
            _, num_relationships = mispair_cal(pipeline_node[idx], node_input, set().union(*nodes_behind))
            num_mispairs += num_relationships

    pipeline_node = remove_empty(pipeline_node)

    pipeline_node = children_aligned_adjustment(pipeline_node, node_output, node_input)
    pipeline_node = remove_empty(pipeline_node)

    num_pipeline = len(pipeline_node)
    for i in range(len(pipeline_node)):
        if sum([p[1] for p in pipeline_node[i]]) <= 0:
            num_pipeline -= 1       

    #node_pipeline_relationship(graph_file, layer_index[0], pipeline_node)

    return pipeline_node, num_pipeline, weight, num_mispairs_record, layer_index


if __name__ == "__main__":
    
    graph_files = ('densenet121.txt', 'densenet169.txt', 'densenet201.txt', 'inception_resnet_v2.txt', 'inception_v3.txt', 'mobilenet_1.00_224.txt', 'mobilenetv2_1.00_224.txt', 'resnet101.txt', 'resnet101v2.txt', 'resnet152.txt', 'resnet152v2.txt', 'vgg16.txt', 'vgg19.txt', 'xception.txt', 'resnet50.txt', 'resnet50v2.txt', 'NASNet.txt')
    #graph_files = ('resnet50.txt', 'resnet50v2.txt')
    #graph_files = ('densenet121_batchnorm.txt', 'densenet169_batchnorm.txt', 'densenet201_batchnorm.txt', 'inception_resnet_v2_batchnorm.txt', 'inception_v3_batchnorm.txt', 'mobilenet_batchnorm.txt', 'mobilenetv2_batchnorm.txt', 'resnet101_batchnorm.txt', 'resnet101v2_batchnorm.txt', 'resnet152_batchnorm.txt', 'resnet152v2_batchnorm.txt', 'vgg16_batchnorm.txt', 'vgg19_batchnorm.txt', 'xception_batchnorm.txt')

    data_files = ('model_0_densenet121.gpickle', 'model_0_densenet169.gpickle', 'model_0_densenet201.gpickle', 'model_0_inception_resnet_v2.gpickle', 'model_0_inception_v3.gpickle', 'model_0_mobilenet_1.00_224.gpickle', 'model_0_mobilenetv2_1.00_224.gpickle', 'model_0_resnet101.gpickle', 'model_0_resnet101v2.gpickle', 'model_0_resnet152.gpickle', 'model_0_resnet152v2.gpickle', 'model_0_vgg16.gpickle', 'model_0_vgg19.gpickle', 'model_0_xception.gpickle', 'model_0_resnet50.gpickle', 'model_0_resnet50v2.gpickle', 'model_0_NASNet.gpickle')
    #data_files = ('model_0_resnet50.gpickle', 'model_0_resnet50v2.gpickle')

    #graph_files = ('NASNet.txt', 'NASNet_reverse_batchnorm.txt')

    #data_files = ('model_0_NASNet.gpickle', 'model_0_NASNet.gpickle')
    memory_upper_bound = (7077888., 13339392., 18668544., 54949632, 21915648., 4210688., 2525440., 43798528., 43778048., 59378688., 59355136., 18878464., 23462144., 21705184., 25366016., 25323520., 4276800.)  
    #memory_upper_bound = (7077888, 13339392, 18668544, 54949632, 21915648, 4210688, 2525440, 43798528, 43778048, 59378688, 59355136, 18878464, 23462144, 21705184)  
    #memory_upper_bound = (25366016, 25323520)

    for i in range(len(graph_files)):
        #path_file = '../../graph_data_collection/graph_data_collection_model_run/inDim15_6_6/' + graph_files[i]
        path_file = '../../graph_data_collection/graph_data_collection_model_run/inDim15_6_6_resource_free/' + graph_files[i]
        graph_file = '../../../dataset/model_check/graph/' + data_files[i]
        #graph_file = '../../dataset/test/graph2/' + data_files[i]

        #left, right = 0.6*memory_upper_bound[i], 1.5*memory_upper_bound[i]
        #left, right = 0.1*memory_upper_bound[i], 1.*memory_upper_bound[i]
        left, right = 0.1*memory_upper_bound[i], 0.9*memory_upper_bound[i]
        pipeline_node, layer_index = [], []
        num_pipeline, weight, num_mispairs_record = 0., 0., 0.

        while right - left > 0.001 or num_pipeline > 6:
            mid = (right + left) / 2.
            pipeline_node, num_pipeline, weight, num_mispairs_record, layer_index = data_treatment(path_file, graph_file, mid)
            if num_pipeline <= 6:
                right = mid
            else:
                left = mid

        neural_net_name = path_file.split('/')[-1].replace('.txt', '') 
        print(neural_net_name)
        """
        print("total weight is: ", weight)
        print("memory upper bound is: ", right)
        print("num of wrong edges is:", num_mispairs_record)
        print("After result treatment, the pipeline memory is:")
        for i in range(len(pipeline_node)):
            print("pipeline_"+str(i)+": ", sum([p[1] for p in pipeline_node[i]]))        
        """
        node_pipeline_relationship(graph_file, layer_index[0], pipeline_node)
        print("end")

        print("")

