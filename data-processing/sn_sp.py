def SensitivityAndSpecificity(pred,real):
    tp,tn,fp,fn = 0,0,0,0

    for i,est in enumerate(pred):
        if round(est) == round(real[i]) and round(est) == 1.:
            tp += 1.
        elif round(est) == round(real[i]) and round(est) == 0:
            tn += 1.
        elif round(est) != round(real[i]) and round(est) ==1:
            fp += 1.
        elif round(est) != round(real[i]) and round(est) == 0:
            fn += 1.
    # tpr -- tnr
    return tp/(tp+fn),tn/(tn+fp)


