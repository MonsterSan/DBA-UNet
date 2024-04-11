def save_log(confmat, losses, log_path, epoch):
    acc_global, acc, iu, Rec, Pre, F1 = confmat.compute()
    acc_global = acc_global.item() * 100
    aver_row_correct = ['{:.1f}'.format(i) for i in (acc * 100).tolist()]
    iou = ['{:.1f}'.format(i) for i in (iu * 100).tolist()]
    miou = iu.mean().item() * 100
    if F1 != 0:
        F1 = F1.item() * 100,
    Rec = Rec.item() * 100,
    if Pre!=0:
        Pre = Pre.item() * 100
    with open(log_path, "a") as lpath:
        lpath.write(
            str(epoch+1) + "\t" + str(losses.avg) + "\t" + str(miou) + "\t" + str(acc_global) + "\t" + str(
                aver_row_correct[0]) + "-" + str(aver_row_correct[1]) + "\t" + str(iou[0]) + "-" + str(
                iou[1]) + "\t" + str(F1) + "\t" + str(Rec) + "\t" + str(Pre) + "\n")
    return miou
