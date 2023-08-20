import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import cv2
from pycocotools.coco import COCO
from Utils.eval import COCOEvalCap
import os
import pandas as pd

def plot_with_df(coco_evals, columns, SOTA=None, display_df=True, **kwargs):
    """
    Plot the automated metrics with using dataframe
    """
    # List of dict
    list_data = [coco_evals[i].eval for i in range(len(coco_evals))]

    if SOTA is not None:
        list_data.insert(0, SOTA)

    df = pd.DataFrame(data = list_data, index=columns)
    df = df.T

    # Set the default color cycle
    mpl.rcParams['axes.prop_cycle'] = mpl.cycler(color=['blue', 'green', 'red', 'orange', 'yellow', 'pink', 'gray', 'black'])

    ax = df.plot(kind='bar', rot=0, xlabel='Metrics', ylabel='Value', table=False, edgecolor='black', **kwargs)

    # add a little space at the top of the plot for the annotation
    ax.margins(y=0.1)

    # move the legend out of the plot
    #ax.legend(title='Model', bbox_to_anchor=(1, 1.02), loc='upper left')
    ax.legend(title='Model')
    #plt.show()
    if display_df:
        print(df)


def get_files(res_files_path):
    # Get the json files and the temperature values
    files = os.listdir(res_files_path)
    files_path = []
    temp_values = []
    for i in range(len(files)):
        if len(files[i].split('.json'))!=1:
            files_path.append(os.path.join(res_files_path, files[i]))
            temp_values.append(float(files[i].split('.json')[0].split('_t')[-1]))
    return files_path, temp_values


def read_result_dir(path, coco):
    dir_list = os.listdir(path)
    nb_ca_list = []
    nb_ep_list = []
    res_files = []
    temp_values = []
    for i in range(len(dir_list)):
        if len(dir_list[i].split('ca'))!=1 and len(dir_list[i].split('ep'))!=1:
            dir_name = dir_list[i]
            nb_ca = int(dir_name.split('ca')[0])
            nb_ep = int(dir_name.split('{}ca_ep'.format(nb_ca))[-1])
            res_files_path, temp = get_files(os.path.join(path, dir_name))
            # Add in the list
            nb_ca_list.append(nb_ca)
            nb_ep_list.append(nb_ep)
            res_files.append(res_files_path)
            temp_values.append(temp)
    # Organize a dictionary with all the info [ca1:[1ca_ep1:[temp1:res_file1, temp2:res_file2], ep2:[temp1:res_file1, temp2:res_file2]], ca2:[temp1:res_file1, temp2:res_file2]]
    res = {}
    unique_ca = list(np.unique(np.array(nb_ca_list)))
    for n in unique_ca:
        res[n] = {}
    for i in range(len(nb_ca_list)):
        res[nb_ca_list[i]][nb_ep_list[i]] = {}
    for i in range(len(nb_ca_list)):
        coco_evals = get_coco_evals(res_files[i], coco)
        for j in range(len(temp_values[i])):
            res[nb_ca_list[i]][nb_ep_list[i]][temp_values[i][j]] = coco_evals[j]
    return res

def get_stat(coco_ex):
    metrics = ['Bleu_1', 'Bleu_2', 'Bleu_3', 'Bleu_4', 'METEOR', 'ROUGE_L', 'CIDEr']
    metrics_res = {m:[] for m in metrics}
    dict_list = list(coco_ex.imgToEval.values())
    for i in range(len(dict_list)):
        for m in metrics:
            metrics_res[m].append(dict_list[i][m])

    df = pd.DataFrame(metrics_res)
    print(df.describe())
    return df




def single_boxplot(result_dict, nb_ca, ep, temp):
    coco_ex = result_dict[nb_ca][ep][temp]
    metrics = ['Bleu_1', 'Bleu_2', 'Bleu_3', 'Bleu_4', 'METEOR', 'ROUGE_L', 'CIDEr']
    metrics_res = {m:[] for m in metrics}
    dict_list = list(coco_ex.imgToEval.values())
    for i in range(len(dict_list)):
        for m in metrics:
            metrics_res[m].append(dict_list[i][m])

    df = pd.DataFrame(metrics_res)
    #print(df.info)
    print('---'*10)
    print('Statistic for results with {} CA, epoch  {} and t={}'.format(nb_ca, ep, temp))
    print(df.describe())
    plt.figure()
    plt.boxplot(metrics_res.values(), showfliers=False)
    plt.xticks(range(1, len(list(metrics_res.keys())) + 1), list(metrics_res.keys()))
    plt.xlabel('Metrics')
    plt.ylabel('Value')
    plt.title('Boxplot metrics for {} CA, epoch {} and t={}'.format(nb_ca, ep, temp))


def multiple_boxplot(result_dict, nb_ca, ep, temp, metrics=['Bleu_1', 'METEOR', 'ROUGE_L', 'CIDEr']):
    f, ax = plt.subplots(2, 2)
    data_list = []
    for i in range(len(metrics)):
        metrics_res = {n: [] for n in nb_ca}
        data_list.append(metrics_res)
    for i in range(len(nb_ca)):
        coco_ex = result_dict[nb_ca[i]][ep][temp]
        dict_list = list(coco_ex.imgToEval.values())
        for j in range(len(dict_list)):
            for idx in range(len(metrics)):
                data_list[idx][nb_ca[i]].append(dict_list[j][metrics[idx]])
    n = 0
    for i in range(ax.shape[0]):
        for j in range(ax.shape[1]):
            ax[i][j].boxplot(data_list[n].values(), showfliers=False)
            ax[i][j].set_xticks(range(1, len(list(data_list[n].keys())) + 1), list(data_list[n].keys()))
            ax[i][j].set_xlabel('Number of added layers')
            ax[i][j].set_ylabel('Value')
            ax[i][j].set_title('{}'.format(metrics[n]))
            n+=1
    plt.tight_layout()



def plot_results(res_dict, SOTA, **kwargs):
    #1. Plot metrics depending on the temp
    nb_ca = list(res_dict.keys())
    for i in range(len(nb_ca)):
        nb_ep = list(res_dict[nb_ca[i]].keys())
        for j in range(len(nb_ep)):
            plot_with_df(list(res_dict[nb_ca[i]][nb_ep[j]].values()), ['t={}'.format(t) for t in list(res_dict[nb_ca[i]][nb_ep[j]].keys())],
                         SOTA=None, display_df=False,
                         title='Influence of the temperature on the quality of the captions ({} CA, epoch {})'.format(nb_ca[i], nb_ep[j]),
                         **kwargs)
    #2. Plot metrics depending on the nb of ca (t=0) after 1 epoch
    columns = ['{} CA'.format(i) for i in nb_ca]
    columns.insert(0, 'SOTA(ClipClap)')
    plot_with_df([res_dict[i][5][0.0] for i in nb_ca], columns,
                 SOTA=SOTA, display_df=False,
                 title='Comparison of the quality of the captions after 5 epoch',
                 **kwargs)
    #3. Plot metrics depending on the epochs
    for i in range(len(nb_ca)):
        plot_with_df([res_dict[nb_ca[i]][j][0.0] for j in list(res_dict[nb_ca[i]].keys())], ['epoch {}'.format(j) for j in list(res_dict[nb_ca[i]].keys())],
                     SOTA=None, display_df=False,
                     title='Influence of the number of epochs on the quality of the captions ({} CA)'.format(
                         nb_ca[i]),
                     **kwargs)
    #4. Plot best against SOTA
    pass


def get_coco_evals(files_path, coco):
    coco_res = []
    coco_evals = []
    for i in range(len(files_path)):
        coco_result = coco.loadRes(files_path[i])
        # create coco_eval object by taking coco and coco_result
        coco_eval = COCOEvalCap(coco, coco_result)
        coco_eval.params['image_id'] = coco_result.getImgIds()
        # Evaluate
        coco_eval.evaluate()
        # Store the objects each list
        coco_res.append(coco_result)
        coco_evals.append(coco_eval)
    return coco_evals


if __name__=='__main__':
    # Path
    annotation_file = './example/captions_val2014.json'

    # Create the coco object
    coco = COCO(annotation_file)
    result_dic = read_result_dir('./res_files/new_CA/', coco)

    # Plots

    ClipClap = {'Bleu_1': None, 'Bleu_2': None, 'Bleu_3': None,'Bleu_4':0.335, 'METEOR':0.274, 'ROUGE_L': None, 'CIDEr':1.13}

    plot_results(result_dic, SOTA=ClipClap)

    plt.show(block=True)
