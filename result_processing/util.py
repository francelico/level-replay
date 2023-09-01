from scipy import stats

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