from neko_2020nocr.dan.common.common import flatten_label_idx,flatten_label
import torch
def mklabel_mk7(module_dict,proto, plabel, tdict, label):
    target, lengthl = module_dict["sampler"].model.encode_noeos(proto, plabel, tdict, label);
    label_flatten, length, idx = flatten_label_idx(target, EOSlen=0, length=lengthl);
    return label_flatten,target,length,lengthl,idx;

def mk8_log(logger_dict,length,label,tdict,gtdict,target,gtarget,choutput,ctxout,terms,loss,DBGKEY):
    tarswunk = ["".join([tdict[i.item()] for i in target[j]])[:length[j]] for j in
                range(len(target))];
    gtarswunk = ["".join([gtdict[i.item()] for i in gtarget[j]])[:length[j]] for j in
                 range(len(gtarget))];
    if (DBGKEY is not None):
        for i in range(len(label)):
            print(gtarswunk[i], ":", choutput[i], "->", ctxout[i])
    logger_dict["accr"].add_iter(choutput, length, tarswunk)
    logger_dict["loss"].add_iter(loss, terms)