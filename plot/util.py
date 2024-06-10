from scipy import stats, spatial

def dedupe_legend_labels_and_handles(labels, handles):
    handle_list, label_list = [], []
    for handle, label in zip(handles, labels):
        if label not in label_list:
            handle_list.append(handle)
            label_list.append(label)
    return label_list, handle_list

def welch_test(a, b):
    t, p = stats.ttest_ind(a, b, equal_var=False)
    return t, p

def calculate_dist_corr(v1, v2) -> float:
    dist_corr = spatial.distance.correlation(v1, v2)
    return dist_corr

def calculate_Kendall(v1, v2): #Kendall rank correlation coefficient
    tau, p_value = stats.kendalltau(v1, v2)
    return tau, p_value