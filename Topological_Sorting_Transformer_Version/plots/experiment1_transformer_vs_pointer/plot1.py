import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import numpy as np

test_files_baseline = ('Toposort30_50_4inDegree_1operator_3resource_priorityInverse_weightCostraint35_128k',)
test_files = ('toposort30_50_4_in_degree_3_resource_weight35_35_same_as_toposortsetting', 'toposort30_50_4_in_degree_3_resource_weight35_35_lr_decay_option1', 'toposort30_50_4_in_degree_3_resource_weight35_35_options2', 'toposort30_50_4_in_degree_3_resource_weight35_35_options3')
reader_order = (0, 1, 2, 3)

Mismatch = {"pointer_network":[], "transformer_same":[], "transformer_lr1-e4_decay05":[], "transformer_lr1-e6":[], "transformer_lr1-e8":[]}
recall_accuracy = {"pointer_network":[], "transformer_same":[], "transformer_lr1-e4_decay05":[], "transformer_lr1-e6":[], "transformer_lr1-e8":[]}
recall_accuracy_min = {"pointer_network":[], "transformer_same":[], "transformer_lr1-e4_decay05":[], "transformer_lr1-e6":[], "transformer_lr1-e8":[]}
recall_accuracy_max = {"pointer_network":[], "transformer_same":[], "transformer_lr1-e4_decay05":[], "transformer_lr1-e6":[], "transformer_lr1-e8":[]}
avg_cost = {"pointer_network":[], "transformer_same":[], "transformer_lr1-e4_decay05":[], "transformer_lr1-e6":[], "transformer_lr1-e8":[]}

epochs = 500
count = 0
with open('../../results/pointer_network/' + test_files_baseline[0], 'rt') as myfile:
    for myline in myfile:
        if myline.startswith('Validation count of misMatch:'):
            data_collected = myline.split()
            Mismatch["pointer_network"].append(float(data_collected[-1]))
            count += 1
        if myline.startswith('Validation mean of recall_accuracy:'):
            data_collected = myline.split()
            recall_accuracy["pointer_network"].append(float(data_collected[-1]))
        if myline.startswith('Validation max of recall_accuracy:'):
            data_collected = myline.split()
            recall_accuracy_max["pointer_network"].append(float(data_collected[-1]))
        if myline.startswith('Validation min of recall_accuracy:'):
            data_collected = myline.split()
            recall_accuracy_min["pointer_network"].append(float(data_collected[-1]))
        if myline.startswith('Validation overall avg_cost:'):
            data_collected = myline.split()
            avg_cost["pointer_network"].append(round(float(data_collected[-3]), 3))

    epochs = min(count, epochs)


data_fetch = [0, 1, 2, 3] 
transfer_setting = ("transformer_same", "transformer_lr1-e4_decay05", "transformer_lr1-e6", "transformer_lr1-e8")
for i in range(4):
    order = reader_order[i]
    count = 0
    with open('../../results/transformer/' + test_files[order], 'rt') as myfile:
        for myline in myfile:
            if myline.startswith('Validation count of misMatch:'):
                data_collected = myline.split()
                Mismatch[transfer_setting[order]].append(float(data_collected[-1]))
                count += 1
            if myline.startswith('Validation mean of recall_accuracy:'):
                data_collected = myline.split()
                recall_accuracy[transfer_setting[order]].append(float(data_collected[-1]))
            if myline.startswith('Validation max of recall_accuracy:'):
                data_collected = myline.split()
                recall_accuracy_max[transfer_setting[order]].append(float(data_collected[-1]))
            if myline.startswith('Validation min of recall_accuracy:'):
                data_collected = myline.split()
                recall_accuracy_min[transfer_setting[order]].append(float(data_collected[-1]))
            if myline.startswith('Validation overall avg_cost:'):
                data_collected = myline.split()
                avg_cost[transfer_setting[order]].append(round(float(data_collected[-3]), 3))
        epochs = min(count, epochs)

"""
fig1, axs1 = plt.subplots()
fig1.suptitle("Final Configured Graph", fontsize=12)
color = ['r', 'orange', 'y', 'g', 'cyan', 'b', 'purple']
for i in data_fetch:
    b = i
    configured_graph = calculated_dataset[graph_size_choices[b]]
    target_graph = configured_graph[-1]
    target_graph_y_sorted = sorted(target_graph, key=lambda x: x[1])
    level = 0
    left, right = 0, 1
    x, y = [t[0] for t in target_graph_y_sorted], [t[1] for t in target_graph_y_sorted]
    x_cal, y_cal = [t[0] for t in target_graph], [t[1] for t in target_graph]
    while right <= len(y):
        if right == len(y) or y[left] != y[right]:
            axs1.scatter(x_cal[left:right], y_cal[left:right], c=color[level%7], marker='o', s=40, label="level: "+ str(level))
            while left < right-1:
                axs1.arrow(x_cal[left], y_cal[left], x_cal[left+1]-x_cal[left], y_cal[left+1]-y_cal[left], head_width=0.02, width=0.001, color=color[level%7])
                left += 1
            left = right
            level += 1
        right += 1

axs1.legend(loc='upper right', prop={'size':5})
axs1.set_title('batch_size=512 with 250 batches, 40 pred 20-5levels')
axs1.set_ylabel('y axis')
axs1.set_xlabel('x axis')

fig1.savefig('toposort40to20_bat512_graph_5level_after.pdf')  
"""

colors = ['r', 'orange', 'y', 'g', 'cyan', 'b', 'purple', 'k', 'magenta']
fig2, axs2 = plt.subplots()
fig2.suptitle("avg count of misMatch, weight constraint35", fontsize=12)
for i in data_fetch:
    b = i
    mis_match = Mismatch[transfer_setting[b]]
    best_val = np.inf
    index = 0
    #for j in range(len(mis_match)):
    for j in range(epochs):
        if mis_match[j] < best_val:
            best_val = mis_match[j]
            index = j
    axs2.plot(np.arange(1, epochs+1, 1), mis_match[:epochs], color=colors[i%9], label=transfer_setting[b]+", best val: "+str(best_val))
    axs2.plot(index+1, best_val, marker='o', color=colors[i%9], markersize=5)

mis_match = Mismatch["pointer_network"]
best_val = np.inf
index = 0
for j in range(epochs):
    if mis_match[j] < best_val:
        best_val = mis_match[j]
        index = j
axs2.plot(np.arange(1, epochs+1, 1), mis_match[:epochs], color=colors[-1], label="pointer_network, best val: "+str(best_val))
axs2.plot(index+1, best_val, marker='o', color=colors[-1], markersize=5)

axs2.set_title('pointer_network vs transformer')
axs2.set_ylabel('misMatch')
axs2.set_xlabel('epochs')
axs2.legend(loc='upper right', prop={'size':5})

fig2.savefig('toposort30_50_misMatch_1operator_3resource_4inDegree_pointer_vs_transformer.pdf')  

"""
fig3, axs3 = plt.subplots()
fig3.suptitle("Avg count of mis_match y axis", fontsize=12)
for i in data_fetch:
    b = i
    mis_match_y = Mismatch_y[graph_size_choices[b]]
    best_val = np.inf
    index = 0
    for j in range(len(mis_match_y)):
        if mis_match_y[j] < best_val:
            best_val = mis_match_y[j]
            index = j
    axs3.plot(np.arange(1, epochs+1, 1), mis_match_y[:epochs], color='b', label=graph_size_choices[b]+" best val: "+str(best_val))
    axs3.plot(index+1, best_val, marker='o', color='b', markersize=5)

axs3.set_title('batch_size=512 with 250 batches, 40 pred 20-5levels')
axs3.set_ylabel('Mismatch on y axis')
axs3.set_xlabel('epochs')
axs3.legend(loc='upper right', prop={'size':5})

fig3.savefig('toposort40to20_bat512_misMatch_5level_yaxis.pdf')  
"""
fig4, axs4 = plt.subplots()
fig4.suptitle("avg recall accuracy, weight constraint35", fontsize=12)
for i in data_fetch:
    b = i
    recall = recall_accuracy[transfer_setting[b]]
    best_val_mean = 0.
    index_mean = 0

    for j in range(epochs):
        if recall[j] > best_val_mean:
            best_val_mean = recall[j]
            index_mean = j

    axs4.plot(np.arange(1, epochs+1, 1), recall[:epochs], color=colors[i], label=transfer_setting[b]+", best val-mean: "+ str(best_val_mean))
    axs4.plot(index_mean+1, best_val_mean, marker='o', color=colors[i], markersize=5)

recall = recall_accuracy["pointer_network"]
best_val_mean = 0.
index_mean = 0
for j in range(epochs):
    if recall[j] > best_val_mean:
        best_val_mean = recall[j]
        index_mean = j

axs4.plot(np.arange(1, epochs+1, 1), recall[:epochs], color=colors[-1], label="pointer_network, best val-mean: "+ str(best_val_mean))
axs4.plot(index_mean+1, best_val_mean, marker='o', color=colors[-1], markersize=5)

axs4.set_title('pointer_network vs transformer')
axs4.legend(loc='lower right', prop={'size':5})
axs4.set_ylabel('avg_recall_accuracy')
axs4.set_xlabel('epochs')

fig4.savefig('toposort30_50_avg_recall_1operator_3resource_4inDegree_pointer_vs_transformer.pdf')  

fig5, axs5 = plt.subplots()
fig5.suptitle("min recall accuracy, weight constraint35", fontsize=12)
for i in data_fetch:
    b = i
    recall_min = recall_accuracy_min[transfer_setting[b]]
    best_val_min = 0.
    index_min = 0

    for j in range(epochs):
        if recall_min[j] > best_val_min:
            best_val_min = recall_min[j]
            index_min = j

    axs5.plot(np.arange(1, epochs+1, 1), recall_min[:epochs], color=colors[i], label=transfer_setting[b]+", best val-min: "+ str(best_val_min))
    axs5.plot(index_min+1, best_val_min, marker='o', color=colors[i], markersize=5)

recall_min = recall_accuracy_min["pointer_network"]
best_val_min = 0.
index_min = 0
for j in range(epochs):
    if recall_min[j] > best_val_min:
        best_val_min = recall_min[j]
        index_min = j

axs5.plot(np.arange(1, epochs+1, 1), recall_min[:epochs], color=colors[-1], label="pointer_network, best val-min: "+ str(best_val_min))
axs5.plot(index_min+1, best_val_min, marker='o', color=colors[-1], markersize=5)

axs5.set_title('pointer_network vs transformer')
axs5.legend(loc='lower right', prop={'size':5})
axs5.set_ylabel('min_recall_accuracy')
axs5.set_xlabel('epochs')

fig5.savefig('toposort30_50_min_recall_1operator_3resource_4inDegree_pointer_vs_transformer.pdf')  

fig6, axs6 = plt.subplots()
fig6.suptitle("max recall accuracy, weight constraint35", fontsize=12)
for i in data_fetch:
    b = i
    recall_max = recall_accuracy_max[transfer_setting[b]]
    best_val_max = 0.
    index_max = 0

    for j in range(epochs):
        if recall_max[j] > best_val_max:
            best_val_max = recall_max[j]
            index_max = j

    axs6.plot(np.arange(1, epochs+1, 1), recall_max[:epochs], color=colors[i], label=transfer_setting[b]+", best val-max: "+ str(best_val_max))
    axs6.plot(index_max+1, best_val_max, marker='o', color=colors[i], markersize=5)

recall_max = recall_accuracy_max["pointer_network"]
best_val_max = 0.
index_max = 0
for j in range(epochs):
    if recall_max[j] > best_val_max:
        best_val_max = recall_max[j]
        index_max = j

axs6.plot(np.arange(1, epochs+1, 1), recall_max[:epochs], color=colors[-1], label="pointer_network, best val-max: "+ str(best_val_max))
axs6.plot(index_max+1, best_val_max, marker='o', color=colors[-1], markersize=5)

axs6.set_title('pointer_network vs transformer')
axs6.legend(loc='lower right', prop={'size':5})
axs6.set_ylabel('max_recall_accuracy')
axs6.set_xlabel('epochs')

fig6.savefig('toposort30_50_max_recall_1operator_3resource_4inDegree_pointer_vs_transformer.pdf')  

"""
fig5, axs5 = plt.subplots()
fig5.suptitle("radius range misPosition", fontsize=12)
for i in data_fetch:
    b = i
    radiusMean = radius_mean[graph_size_choices[b]]
    radiusMax = radius_max[graph_size_choices[b]]
    best_val_mean = np.inf
    best_val_max = np.inf
    index_mean = 0
    index_max = 0
    for j in range(len(radiusMean)):
        if radiusMean[j] < best_val_mean:
            best_val_mean = radiusMean[j]
            index_mean = j
        if radiusMax[j] < best_val_max:
            best_val_max = radiusMax[j]
            index_max = j

    axs5.plot(np.arange(1, epochs+1, 1), radiusMean[:epochs], color=colors[i*2+0], label=graph_size_choices[b]+" best val-mean: "+str(best_val_mean))
    axs5.plot(np.arange(1, epochs+1, 1), radiusMax[:epochs], color=colors[i*2+1], label=graph_size_choices[b]+" best val-max: "+str(best_val_max))
    axs5.plot(index_mean+1, best_val_mean, marker='o', color=colors[i*2+0], markersize=5)
    axs5.plot(index_max+1, best_val_max, marker='o', color=colors[i*2+1], markersize=5)

axs5.set_title('batch_size=512 graph20 with transformer dim=64')
axs5.set_xlabel('epochs')
axs5.set_ylabel('radius range misMatch')
axs5.legend(loc='upper right', prop={'size':5})

fig5.savefig('toposort20_bat512_DAG_radius_TransGAN_batchNorm_dim64.pdf')  

fig6, axs6 = plt.subplots()
fig6.suptitle("Max radius misPosition", fontsize=12)
for i in data_fetch:
    b = i
    radiusMax = radius_max[graph_size_choices[b]]
    best_val = np.inf
    index = 0
    for j in range(len(radiusMax)):
        if radiusMax[j] < best_val:
            best_val = radiusMax[j]
            index = j

    axs6.plot(np.arange(1, epochs+1, 1), radiusMax[:epochs], color='r', label=graph_size_choices[b]+" best val: "+str(best_val))
    axs6.plot(index+1, best_val, marker='o', color='r', markersize=5)


axs6.set_title('batch_size=512 with 250 batches, 40 pred 20-5levels')
axs6.set_xlabel('epochs')
axs6.legend(loc='upper right', prop={'size':5})
axs6.set_ylabel('radius_max')

fig6.savefig('toposort40to20_bat512_radius_max_5level.pdf')  
"""

fig7, axs7 = plt.subplots()
fig7.suptitle("avg cost, weight constraint35", fontsize=12)
for i in data_fetch:
    b = i
    cost = avg_cost[transfer_setting[b]]
    best_val = np.inf
    index = 0
    for j in range(epochs):
        if cost[j] < best_val:
            best_val = cost[j]
            index = j

    axs7.plot(np.arange(1, epochs+1, 1), cost[:epochs], color=colors[i], label=transfer_setting[b]+", best val: "+str(best_val))
    axs7.plot(index+1, best_val, marker='o', color=colors[i], markersize=5)

cost = avg_cost["pointer_network"]
best_val = np.inf
index = 0
for j in range(epochs):
    if cost[j] < best_val:
        best_val = cost[j]
        index = j

axs7.plot(np.arange(1, epochs+1, 1), cost[:epochs], color=colors[-1], label="pointer_network, best val: "+str(best_val))
axs7.plot(index+1, best_val, marker='o', color=colors[-1], markersize=5)

axs7.set_title('pointer_network vs transformer')
axs7.set_xlabel('epochs')
axs7.set_ylabel('avg cost')
axs7.legend(loc='upper right', prop={'size':5})

fig7.savefig('toposort30_50_avg_cost_1operator_3resource_4inDegree_pointer_vs_transformer.pdf')  
