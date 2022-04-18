import os
from configs import model_mod_cfg as modcfg
from configs import arm_base_task_default2
from configs import osdanmk7_eval_routine_cfg

def dan_mjst_dict_eval_cfg(
        save_root,dsroot,
        log_path,iterkey,temeta,maxT=30,name="base_chs_"):

    if(log_path):
        epath=os.path.join(log_path, "closeset_benchmarks");
    else:
        epath=None;
    task_dict = {}
    task_dict = arm_base_task_default2(task_dict, "base_chs_", osdanmk7_eval_routine_cfg, maxT,
                                       temeta, None,
                                       log_path);
    # task_dict = arm_base_mk8_task_default(task_dict, "base_chs_", osdanmk8_eval_routine_cfg, 30,
    #                                      te_meta_path_chsjap, chs_eval_ds,
    #                                      log_path,force_skip_ctx=True);

    return \
    {
        "root": save_root,
        "iterkey": iterkey, # something makes no sense to start fresh
        "modules": modcfg(None,None,25,30),
        "export_path":epath,
        "tasks":task_dict
    }
