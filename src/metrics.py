import numpy as np

def precision_special(y_true, y_pred):
    return np.sum(y_true == y_pred)/len(y_true)

def precision_top_k(y_true, y_pred, k):
    return np.sum(y_true[:k] == y_pred[:k])/k

def average_precision(y_true, y_pred, m):
    return np.sum([precision_top_k(y_true, y_pred, k) for k in range(1,m+1)])/m

def argsort_top_n(y_list, n):
    indices_unsorted = np.argpartition(y_list, -n)[-n:]
    combined_time_order_unsorted = np.array([np.array(y_list)[indices_unsorted], -indices_unsorted]).T
    return indices_unsorted[np.lexsort(combined_time_order_unsorted[:,::-1].T)][::-1]

def mean_reciprocal_rank(relevant_items_list, predicted_candidates_lists):
    ranks = [list(predicted_candidates_lists[i_i]).index(
        item)+1 if item in predicted_candidates_lists[i_i] else 0 for i_i, item in enumerate(relevant_items_list)]
    return np.sum([1/rank if rank > 0 else 0 for rank in ranks])/len(relevant_items_list)
