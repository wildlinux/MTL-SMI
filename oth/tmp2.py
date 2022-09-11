def calRPF(list_pred: list, list_label: list, label=0):
    length = len(list_pred)
    TP = TN = FP = FN = 0
    for i in range(length):
        y_pred = list_pred[i]
        y_label = list_label[i]
        if y_pred != label and y_label != label:
            continue
        elif y_pred == label and y_label == label:
            TP += 1
        elif y_pred == label and y_label != label:
            FP += 1
        elif y_pred != label and y_label == label:
            FN += 1
    if TP + FN == 0 or TP + FP == 0:
        return 0.0, 0.0, 0.0
    recall = TP / (TP + FN)
    precision = TP / (TP + FP)
    if recall == 0 and precision == 0:
        return 0.0, 0.0, 0.0
    f1 = 2 * precision * recall / (precision + recall)
    return recall, precision, f1


# calRPFmacro(list_pred, list_label, labels=(0,1,2))
def calRPFmacro(list_pred: list, list_label: list, labels=(0, 1, 2)):
    list_recall = []
    list_precision = []
    list_f1 = []
    for label in labels:
        recall, precision, f1 = calRPF(list_pred, list_label, label)
        list_recall.append(recall)
        list_precision.append(precision)
        list_f1.append(f1)
    # list_pred和list_label中一共出现几种类别, 就算那几种类别的指标, 没出现的就不考虑了
    length = len(set(list_pred+list_label))
    macro_recall = sum(list_recall) / length
    macro_precision = sum(list_precision) / length
    macro_f1 = sum(list_f1) / length
    return macro_recall, macro_precision, macro_f1


# calRPFmacro9(list_pred2, list_label2, labels=(0,1,2))
def calRPFmacro9(list_pred2: list, list_label2: list, labels=(0, 1, 2)):
    list_recall = []
    list_precision = []
    list_f1 = []
    length = len(list_pred2)
    for i in range(length):
        lis_pred2 = list_pred2[i]
        lis_label2 = list_label2[i]
        recall, precision, f1 = calRPFmacro(lis_pred2, lis_label2, (0, 1, 2))
        list_recall.append(recall)
        list_precision.append(precision)
        list_f1.append(f1)
    macro_recall = sum(list_recall) / length
    macro_precision = sum(list_precision) / length
    macro_f1 = sum(list_f1) / length
    print(list_f1)
    return macro_recall, macro_precision, macro_f1