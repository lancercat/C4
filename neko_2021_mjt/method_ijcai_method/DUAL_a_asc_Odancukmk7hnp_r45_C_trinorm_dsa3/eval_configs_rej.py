import os
from configs import model_mod_cfg as modcfg
from configs import arm_base_task_default2
from neko_2021_mjt.configs.data.mjst_data import get_test_all_uncased_dsrgb,get_uncased_dsrgb_d_tr
from configs import osdanmk7_eval_routine_cfg
from neko_2021_mjt.configs.data.chs_jap_data import get_chs_HScqa,get_eval_jap_color,get_chs_tr_meta,get_jap_te_meta
from neko_2021_mjt.dss_presets.dual_no_lsct_32 import get_eval_dss,get_eval_dssgosr,get_eval_dssosr;
def dan_mjst_eval_cfg(
        save_root,dsroot,
        log_path,iterkey,maxT=30):

    if(log_path):
        epath=os.path.join(log_path, "closeset_benchmarks");
    else:
        epath=None;
    te_meta_path_chsjapz, te_meta_path_mjstz, mjst_eval_dsz, chs_eval_dsz = get_eval_dss(dsroot, 25, 30);
    te_meta_path_chsjap, te_meta_path_mjst, mjst_eval_ds, chs_eval_ds=get_eval_dssosr(dsroot,25,30);
    te_meta_path_chsjapg, te_meta_path_mjstg, mjst_eval_dsg, chs_eval_dsg=get_eval_dssgosr(dsroot,25,30);

    task_dict = {}

    task_dict = arm_base_task_default2(task_dict, "base_chs_", osdanmk7_eval_routine_cfg, 30,
                                         te_meta_path_chsjapz, chs_eval_dsz,
                                         log_path,measure_rej=False,name="GZSL");
    task_dict = arm_base_task_default2(task_dict, "base_chs_", osdanmk7_eval_routine_cfg, 30,
                                         te_meta_path_chsjap, chs_eval_ds,
                                         log_path,measure_rej=True,name="OSR");
    task_dict = arm_base_task_default2(task_dict, "base_chs_", osdanmk7_eval_routine_cfg, 30,
                                         te_meta_path_chsjapg, chs_eval_dsg,
                                         log_path,measure_rej=True,name="GOSR");
    return \
    {
        "root": save_root,
        "iterkey": iterkey, # something makes no sense to start fresh
        "modules": modcfg(None,None,25,30),
        "export_path":epath,
        "tasks":task_dict
    }
