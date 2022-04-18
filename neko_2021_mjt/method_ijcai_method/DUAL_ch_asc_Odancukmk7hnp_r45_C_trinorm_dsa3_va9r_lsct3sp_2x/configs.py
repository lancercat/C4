


from neko_2021_mjt.configs.routines.ocr_routines.mk7.osdanmk7_routine_cfg import osdanmk7_eval_routine_cfg
from neko_2021_mjt.configs.routines.synth_routine.osfsl_routine_cfg import osfsl_ocr_routine
from neko_2021_mjt.configs.routines.ocr_routines.mk7.osdanmk7_routine_cfg_va9 import osdanmk7_va9r_ocr_routine


from neko_2021_mjt.dss_presets.dual_chhwctw_lsct_32 import get_dss,get_eval_dss;

from neko_2021_mjt.configs.loadouts.mk7.lsctt_module_set_mk7 import \
    arm_lsctsp_module_set_r45pt_trinorm_orig_dsa3hGTAnp_va9_mk7,arm_lsct_mk7_va9_routine
from neko_2021_mjt.configs.loadouts.base_module_set import arm_base_task_default2

def arm_eval_tasks(dsroot,log_path,maxT_mjst=25,maxT_chs=30):
    te_meta_path_hwdb, te_meta_path_ctw, hwdb_eval_ds, ctw_eval_ds= get_eval_dss(dsroot);
    task_dict = {}
    task_dict = arm_base_task_default2(task_dict, "base_hwdb_", osdanmk7_eval_routine_cfg, 2, te_meta_path_hwdb,
                                       hwdb_eval_ds, log_path);
    task_dict = arm_base_task_default2(task_dict, "base_ctw_", osdanmk7_eval_routine_cfg, 2, te_meta_path_ctw,
                                       ctw_eval_ds, log_path);


    return task_dict;


def model_mod_cfg(tr_meta_path_ctw,tr_meta_path_hwdb):
    capacity=256;
    feat_ch=512;
    mods={};
    mods=arm_lsctsp_module_set_r45pt_trinorm_orig_dsa3hGTAnp_va9_mk7(mods,"base_hwdb_",2,capacity,feat_ch,tr_meta_path_hwdb,wemb=0);
    mods=arm_lsctsp_module_set_r45pt_trinorm_orig_dsa3hGTAnp_va9_mk7(mods,"base_ctw_",2,capacity,feat_ch,tr_meta_path_ctw,wemb=0);
    return mods;



def dan_single_model_train_cfg(save_root,dsroot,chcnt,
                               log_path,log_each,itrk= "Top Nep",bsize=160):

    tr_meta_path_ctwch,tr_meta_path_hwdb,te_meta_path_hwdb,te_meta_path_ctw,\
    hwdb_eval_ds,ctw_eval_ds,train_joint_ds=get_dss(dsroot,chcnt,bsize,pfac=2);

    task_dict=arm_eval_tasks(dsroot,log_path)
    routines = {};
    # routines = arm_base_routine2(routines, "base_mjst_", osdanmk7dt_ocr_routine, maxT_mjst, log_path,
    #                             log_each, "dan_mjst_");
    routines = arm_lsct_mk7_va9_routine(routines, "base_hwdb_", osdanmk7_va9r_ocr_routine, 2,osfsl_ocr_routine, log_path,
                                  log_each, "hwdb_",lsct_proto_name="lsct_prototyper");
    routines = arm_lsct_mk7_va9_routine(routines, "base_ctw_", osdanmk7_va9r_ocr_routine, 2,osfsl_ocr_routine, log_path,
                                  log_each, "ctw_",lsct_proto_name="lsct_prototyper");


    return \
        {
            "root": save_root,
            "val_each": 2000,
            "vitr": 10000,
            "vepoch": 4,
            "iterkey": itrk,  # something makes no sense to start fresh
            "dataloader_cfg":train_joint_ds,
            # make sure the unseen characters are unseen.
            "modules": model_mod_cfg(tr_meta_path_ctwch,tr_meta_path_hwdb),
            "routine_cfgs": routines,
            "tasks": task_dict,
        }

