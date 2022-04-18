import os
from configs import model_mod_cfg as modcfg
from configs import osdanmk7_eval_routine_cfg
from neko_2021_mjt.configs.loadouts.base_module_set import arm_base_task_default2
from neko_2021_mjt.dss_presets.dual_chhwctw_eval_32 import get_eval_rej_dss;

def arm_eval_tasks_rej(dsroot,log_path,rej_spl="dictrej.pt"):
    te_meta_path_hwdb, te_meta_path_ctw, hwdb_eval_ds, ctw_eval_ds= get_eval_rej_dss(dsroot,rej_spl);
    task_dict = {}
    task_dict = arm_base_task_default2(task_dict, "base_hwdb_", osdanmk7_eval_routine_cfg, 2, te_meta_path_hwdb,
                                       hwdb_eval_ds, log_path,measure_rej=True);
    task_dict = arm_base_task_default2(task_dict, "base_ctw_", osdanmk7_eval_routine_cfg, 2, te_meta_path_ctw,
                                       ctw_eval_ds, log_path,measure_rej=True);


    return task_dict;


def dan_mjst_eval_cfg(
        save_root,dsroot,
        log_path,iterkey,rej_spl):

    if(log_path):
        epath=os.path.join(log_path, "closeset_benchmarks");
    else:
        epath=None;
    task_dict=arm_eval_tasks_rej(dsroot,log_path,rej_spl);

    return \
    {
        "root": save_root,
        "iterkey": iterkey, # something makes no sense to start fresh
        "modules": modcfg(None,None),
        "export_path":epath,
        "tasks":task_dict
    }
