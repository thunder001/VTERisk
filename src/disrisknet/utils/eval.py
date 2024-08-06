import sklearn.metrics
from disrisknet.utils.c_index import concordance_index
import warnings
import numpy as np
from sklearn.metrics._ranking import _binary_clf_curve


N_POINTS = 10000


def compute_eval_metrics(args, loss, golds, patient_golds, preds,
                         probs, exams, pids, dates, censor_times, days_to_final_censors, stats_dict, key_prefix):
    
    stats_dict['{}_loss'.format(key_prefix)].append(loss)
    preds_dict = {
        'golds': golds,
        'probs': probs,
        'patient_golds': patient_golds,
        'exams': exams,
        'pids': pids,
        'dates': dates,
        'censor_times': censor_times,
        'days_to_final_censors': days_to_final_censors
    }

    log_statement = '-- loss: {:.6f}\n'.format(loss)
    sub_golds = {}
    if args.annual_eval_type == 'both':
        annual_eval_types = [False, True]
    elif args.annual_eval_type == 'cumulative':
        annual_eval_types = [False]
    elif args.annual_eval_type == 'independent':
        annual_eval_types = [True]
    else:
        raise Exception('Illegal annual_eval_type. Choose from [independent,cumulative, both]')

    global INCIDENCE
    INCIDENCE = sum(golds)/len(golds)

    for independent_eval in annual_eval_types:

        independent_eval_label = 'i' if independent_eval else 'c'
        time_points = args.day_endpoints if args.pred_day else args.month_endpoints  

<<<<<<< HEAD
        for index, time in enumerate(args.day_endpoints):
=======
        for index, time in enumerate(time_points):
>>>>>>> d4f26c810f0e0115007bb8362475c8096add88b9
            probs_for_eval, golds_for_eval = [], []

            for prob_arr, censor_time, gold in zip(probs, censor_times, golds):
                include, label = include_exam_and_determine_label(index, censor_time, gold,  independent_eval)
                if include:
                    probs_for_eval.append(prob_arr[index])
                    golds_for_eval.append(label)
            
            if args.eval_auroc:
                key_name = '{}_{}day_auroc_{}'.format(key_prefix, time, independent_eval_label)
                auc, curve = compute_auroc(golds_for_eval, probs_for_eval)
                log_statement += " -{}: {} (n={} , c={} )\n".format(key_name, auc, len(golds_for_eval), sum(golds_for_eval))
                stats_dict[key_name].append(auc)
                stats_dict[key_name+'_curve'].append(curve)

            if args.eval_auprc:
                key_name = '{}_{}day_auprc_{}'.format(key_prefix, time, independent_eval_label)
                auc, fig_curve = compute_auprc(golds_for_eval, probs_for_eval)
                log_statement += " -{}: {} (n={} , c={} )\n".format(key_name, auc, len(golds_for_eval), sum(golds_for_eval))    
                stats_dict[key_name].append(auc)
                stats_dict[key_name+'_curve'].append(fig_curve)

                key_name = '{}_{}day_auprc_corrected_{}'.format(key_prefix, time, independent_eval_label)
                auc, fig_curve = compute_auprc_corrected(golds_for_eval, probs_for_eval, golds)
                log_statement += " -{}: {} (n={} , c={} )\n".format(key_name, auc, len(golds_for_eval),
                                                                  sum(golds_for_eval))
                stats_dict[key_name].append(auc)
                stats_dict[key_name + '_curve'].append(fig_curve)

            if args.eval_mcc:
                key_name = '{}_{}day_mcc_{}'.format(key_prefix, time, independent_eval_label)
                mcc, fig_curve = compute_mcc(golds_for_eval, probs_for_eval)
                log_statement += " -{}: {} (n={} , c={} )\n".format(key_name, mcc, len(golds_for_eval), sum(golds_for_eval))
                stats_dict[key_name].append(mcc)
                stats_dict[key_name+'_curve'].append(fig_curve)

                key_name = '{}_{}day_mcc_corrected_{}'.format(key_prefix, time, independent_eval_label)
                mcc, fig_curve = compute_mcc_corrected(golds_for_eval, probs_for_eval, golds)
                log_statement += " -{}: {} (n={} , c={} )\n".format(key_name, mcc, len(golds_for_eval), sum(golds_for_eval))
                stats_dict[key_name].append(mcc)
                stats_dict[key_name+'_curve'].append(fig_curve)

        if args.eval_c_index:
            c_index = compute_c_index(probs, censor_times, golds)
            stats_dict['{}_c_index_{}'.format(key_prefix, independent_eval_label)].append(c_index)
            log_statement += " -c_index_{}: {}\n".format(independent_eval_label, c_index)

    return log_statement, stats_dict, preds_dict


def include_exam_and_determine_label(followup, censor_time, gold, independent_eval=False):
    if independent_eval:
        valid_pos = gold and censor_time == followup
    else:
        valid_pos = gold and censor_time <= followup
    valid_neg = censor_time >= followup
    included, label = (valid_pos or valid_neg), valid_pos
    return included, label


def compute_c_index(probs, censor_times, golds):
    try:
        c_index = concordance_index(censor_times, probs, golds)
    except Exception as e:
        warnings.warn("Failed to calculate C-index because {}".format(e))
        c_index = 'NA'
    return c_index


def compute_auroc(golds_for_eval, probs_for_eval):
    try:
        fpr, tpr, thresholds = sklearn.metrics.roc_curve(golds_for_eval, probs_for_eval, pos_label=1)
        auc = sklearn.metrics.roc_auc_score(golds_for_eval, probs_for_eval, average='samples')
        fig_dict = {'fpr':fpr[::max(1,len(fpr)//N_POINTS)],
                    'tpr':tpr[::max(1,len(tpr)//N_POINTS)]}
    except Exception as e:
        warnings.warn("Failed to calculate AUROC because {}".format(e))
        auc = 'NA'
        fig_dict = {}
    return auc, fig_dict


def compute_auprc(golds_for_eval, probs_for_eval):
    try:
        precisions, recalls, thresholds = sklearn.metrics.precision_recall_curve(golds_for_eval, probs_for_eval, pos_label=1)
        auc = sklearn.metrics.auc(recalls, precisions)
        fig_dict = {'precisions': precisions[::max(1,len(precisions)//N_POINTS)],
                    'recalls': recalls[::max(1,len(precisions)//N_POINTS)]}
    except Exception as e:
        warnings.warn("Failed to calculate AUPRC because {}".format(e))
        auc = 'NA'
        fig_dict = {}
    return auc, fig_dict


def compute_auprc_corrected(golds_for_eval, probs_for_eval, golds):
    try:
        global INCIDENCE
        p = len(golds) * INCIDENCE
        n = len(golds) - p
        fp, tp, thresholds = _binary_clf_curve(golds_for_eval, probs_for_eval)
        tn, fn = n - fp, p - tp
        precisions = tp / (tp + fp)
        recalls = tp / (tp + fn)
        auc = sklearn.metrics.auc(recalls, precisions)
        fig_dict = {'precisions': precisions[::max(1,len(precisions)//N_POINTS)],
                    'recalls': recalls[::max(1,len(precisions)//N_POINTS)]}
    except Exception as e:
        warnings.warn("Failed to calculate AUPRC (corrected) because {}".format(e))
        auc = 'NA'
        fig_dict = {}
    return auc, fig_dict


def compute_mcc(golds_for_eval, probs_for_eval):
    try:
        p = sum(golds_for_eval)
        n = sum([not el for el in golds_for_eval])
        fp, tp, thresholds = _binary_clf_curve(golds_for_eval, probs_for_eval)
        tn, fn = n - fp, p - tp
        mcc = (tp * tn - fp * fn) / (np.sqrt(((tp + fp) * (fp + tn) * (tn + fn) * (fn + tp)))+ 1e-10)
        fig_dict = {'mcc': mcc, 'thresholds': thresholds}
    except Exception as e:
        warnings.warn("Failed to calculate MCC because {}".format(e))
        mcc = 'NA'
        fig_dict = {}
    return max(mcc), fig_dict


def compute_mcc_corrected(golds_for_eval, probs_for_eval, golds):
    try:
        global INCIDENCE
        p = len(golds) * INCIDENCE
        n = len(golds) - p
        fp, tp, thresholds = _binary_clf_curve(golds_for_eval, probs_for_eval)
        tn, fn = n - fp, p - tp
        mcc = (tp * tn - fp * fn) / (np.sqrt(((tp + fp) * (fp + tn) * (tn + fn) * (fn + tp)))+1e-10)
        fig_dict = {'mcc': mcc, 'thresholds': thresholds}
    except Exception as e:
        warnings.warn("Failed to calculate MCC (corrected) because {}".format(e))
        mcc = 'NA'
        fig_dict = {}
    return max(mcc), fig_dict
