import os
from configs import model_mod_cfg as modcfg
from configs import arm_eval_tasks_osr;
from neko_2021_mjt.configs.data.mjst_data import get_test_all_uncased_dsrgb,get_uncased_dsrgb_d_tr
from configs import osdanmk7_eval_routine_cfg
from neko_2021_mjt.configs.data.chs_jap_data import get_chs_HScqa,get_eval_jap_color,get_chs_tr_meta,get_jap_te_meta
from neko_2021_mjt.dss_presets.dual_no_lsct_32 import get_eval_dss;
def dan_mjst_eval_cfg(
        save_root,dsroot,
        log_path,iterkey,maxT=30):

    if(log_path):
        epath=os.path.join(log_path, "closeset_benchmarks");
    else:
        epath=None;
    task_dict=arm_eval_tasks_osr(dsroot,log_path);

    return \
    {
        "root": save_root,
        "iterkey": iterkey, # something makes no sense to start fresh
        "modules": modcfg(None,None,25,30),
        "export_path":epath,
        "tasks":task_dict
    }
