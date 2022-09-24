import numpy as np
import math
import networkx as nx
import random

"""
Data generator for mathematical symbolic expression;

Given input: 
    File consisting of a series of mathematical equation of string type--(m0=n1+n2)

Expected output:
    A bunch of 4-dim nodes of shape (variable_out, variable_in1, operation, variable_in2); m0=n1+n2 will be encoded as(4, 0, 5, 1, 6)

Constraints:
    Only three operations are supported: add(1), subtract(2), multi(3)    
    Only support symbolic mathematic expression:
                                               --2*m1*m2 should be rewritten as m1*m2+m1*m2; 
                                               --m1-2 is forbidden 
    variable will be given a unique number starting from 4;
    0 will be assigned to "equal" sign
"""

def data_extracting_embedding(file_path):
    
    outputSignature = '' 
    var2num = dict()
    num2var = dict()
    num_assigner = 4 # Variable assigned with unique number starting from 4
    graph = []
    operations = {'+':1, '-':2, '*':3}
    
    def data_trace(signal, num):
        if signal not in var2num:
            var2num[signal] = num
            num2var[num] = signal
            return True
        return False

    def expression_decomposition(expression):
        if '+' in expression:
            inputs = expression.split('+')
            return inputs[0], inputs[1], '+'
        if '-' in expression:
            inputs = expression.split('-')
            return inputs[0], inputs[1], '-'
        if '*' in expression:
            inputs = expression.split('*')
            return inputs[0], inputs[1], '*'

    with open(file_path, 'rt') as myfile:
        for equation in myfile:
            equation = equation.replace(' ', '').replace('\n', '')
            if len(equation) == 0:
                continue
            if '=' in equation:
                expressions = equation.split('=')
                out_signal = expressions[0]
                input1_signal, input2_signal, operation = expression_decomposition(expressions[-1])

                if data_trace(input1_signal, num_assigner): num_assigner += 1     
                if data_trace(input2_signal, num_assigner): num_assigner += 1     
                if data_trace(out_signal, num_assigner): num_assigner += 1     

                graph.append([var2num[out_signal], 0, var2num[input1_signal], operations[operation], var2num[input2_signal]])
            else:
                outputSignature = equation                

    return outputSignature, graph            



if __name__ == '__main__':
    
    #path_file = 'mult2b-abc.v.eqn'

    #output_signature, graph_embedding = data_extracting_embedding(path_file) 
    #print(output_signature)
    #print(graph_embedding)
    D = nx.gn_graph(10)
    G=nx.gnp_random_graph(10,0.5,directed=True)
    DAG = nx.DiGraph([(u,v,{'weight':random.randint(1,10)}) for (u,v) in G.edges() if u<v])
    print(list(DAG))
    print(list(DAG.edges(data=True)))
    leaf = ([x for x in DAG.nodes() if DAG.out_degree(x)==0 and DAG.in_degree(x)!=0])
    print(len(leaf))
    for ln in leaf:
        G.add_edge(ln, 10+1, weight=random.randint(1,10))
    leaf = ([x for x in G.nodes() if G.out_degree(x)==0 and G.in_degree(x)!=0])
    print(len(leaf))
    import matplotlib.pyplot as plt
    #pos = nx.spring_layout(G)
    #nx.draw_networkx(G, node_color ='green', pos=nx.spectral_layout(G))
    #plt.show()

    #for ele in data_embedding:
    #    print(ele)




