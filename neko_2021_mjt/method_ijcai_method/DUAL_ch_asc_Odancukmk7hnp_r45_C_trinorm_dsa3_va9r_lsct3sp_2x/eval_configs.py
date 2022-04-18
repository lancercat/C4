import os
from configs import model_mod_cfg as modcfg
from configs import arm_eval_tasks;
def dan_mjst_eval_cfg(
        save_root,dsroot,
        log_path,iterkey,maxT=30):

    if(log_path):
        epath=os.path.join(log_path, "closeset_benchmarks");
    else:
        epath=None;
    task_dict=arm_eval_tasks(dsroot,log_path);

    return \
    {
        "root": save_root,
        "iterkey": iterkey, # something makes no sense to start fresh
        "modules": modcfg(None,None),
        "export_path":epath,
        "tasks":task_dict
    }
