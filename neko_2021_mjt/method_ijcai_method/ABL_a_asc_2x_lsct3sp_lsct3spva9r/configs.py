
from neko_2021_mjt.configs.loadouts.mk7.lsctt_module_set_mk7 import \
    arm_lsctsp_module_set_r45trinorm_orig_dsa3hGTAnp_mk7,arm_lsctsp_module_set_r45_trinorm_orig_dsa3hGTAnp_va9_mk7,\
    arm_lsct_mk7_routine,arm_lsct_mk7_va9_routine
from neko_2021_mjt.configs.loadouts.base_module_set import arm_base_task_default2
from neko_2021_mjt.configs.routines.ocr_routines.mk7.osdanmk7_routine_cfg import osdanmk7_eval_routine_cfg
from neko_2021_mjt.dss_presets.dual_no_lsct_32 import get_eval_dss,get_eval_dssgosr,get_eval_dssosr;
from neko_2021_mjt.dss_presets.dual_lsct32 import get_dsssch,get_teds;
from neko_2021_mjt.dss_presets.dual_no_lsct_32 import get_eval_dss_kr,get_eval_dssenjp;

from neko_2021_mjt.configs.routines.ocr_routines.mk7.osdanmk7_routine_cfg import  osdanmk7_ocr_routine
from neko_2021_mjt.configs.routines.ocr_routines.mk7.osdanmk7_routine_cfg_va9 import osdanmk7_va9r_ocr_routine

from neko_2021_mjt.configs.routines.synth_routine.osfsl_routine_cfg import osfsl_ocr_routine
# we only want detailed log for our method (Just in case overwritting)

def arm_eval_tasks_full(dsroot,log_path,maxT_mjst=25,maxT_chs=30):
    te_meta_path_chsjap, te_meta_path_mjst, mjst_eval_ds, chs_eval_ds = \
        get_eval_dssenjp(dsroot, maxT_mjst, maxT_chs);
    te_meta_path_kr, _, _, kr_eval_ds = \
        get_eval_dss_kr(dsroot, maxT_mjst, maxT_chs);


    task_dict = {}
    task_dict = arm_base_task_default2(task_dict, "base_chs_lsctsp_2x_va9r_", osdanmk7_eval_routine_cfg, 30,
                                       te_meta_path_mjst,
                                       mjst_eval_ds,
                                       log_path,name="English");
    task_dict = arm_base_task_default2(task_dict, "base_chs_lsctsp_2x_va9r_", osdanmk7_eval_routine_cfg, maxT_chs,
                                       te_meta_path_chsjap,
                                       chs_eval_ds,
                                       log_path,name="Japanese");
    task_dict = arm_base_task_default2(task_dict, "base_chs_lsctsp_2x_va9r_", osdanmk7_eval_routine_cfg, maxT_chs,
                                       te_meta_path_kr,
                                       kr_eval_ds,
                                       log_path,name="Korean");
    return task_dict;

def arm_eval_tasks_osr(dsroot,log_path,maxT_mjst=25,maxT_chs=30):
    te_meta_path_chsjap, te_meta_path_mjst, mjst_eval_ds, chs_eval_ds = \
        get_eval_dss(dsroot, maxT_mjst, maxT_chs);
    te_meta_path_chsjapo, _,_,_ = \
        get_eval_dssosr(dsroot, maxT_mjst, maxT_chs);
    te_meta_path_chsjapg, _,_,_  = \
        get_eval_dssgosr(dsroot, maxT_mjst, maxT_chs);

    task_dict = {}
    task_dict = arm_base_task_default2(task_dict, "base_chs_lsctsp_2x_", osdanmk7_eval_routine_cfg, maxT_chs,
                                       te_meta_path_chsjap, chs_eval_ds,
                                       log_path,name="GZSL");
    task_dict = arm_base_task_default2(task_dict, "base_chs_lsctsp_2x_va9r_", osdanmk7_eval_routine_cfg, maxT_chs,
                                       te_meta_path_chsjap,
                                       chs_eval_ds,
                                       log_path,name="GZSL");
    task_dict = arm_base_task_default2(task_dict, "base_chs_lsctsp_2x_", osdanmk7_eval_routine_cfg, maxT_chs,
                                       te_meta_path_chsjapo, chs_eval_ds,
                                       log_path,measure_rej=True,name="OSR");
    task_dict = arm_base_task_default2(task_dict, "base_chs_lsctsp_2x_va9r_", osdanmk7_eval_routine_cfg, maxT_chs,
                                       te_meta_path_chsjapo,
                                       chs_eval_ds,
                                       log_path,measure_rej=True,name="OSR");
    task_dict = arm_base_task_default2(task_dict, "base_chs_lsctsp_2x_", osdanmk7_eval_routine_cfg, maxT_chs,
                                       te_meta_path_chsjapg, chs_eval_ds,
                                       log_path,measure_rej=True,name="GOSR");
    task_dict = arm_base_task_default2(task_dict, "base_chs_lsctsp_2x_va9r_", osdanmk7_eval_routine_cfg, maxT_chs,
                                       te_meta_path_chsjapg,
                                       chs_eval_ds,
                                       log_path,measure_rej=True,name="GOSR");
    return task_dict;

def model_mod_cfg(tr_meta_path_chs,tr_meta_path_mjst,maxT_mjst,maxT_chs):
    capacity=256;
    feat_ch=512;
    mods={};
    mods=arm_lsctsp_module_set_r45trinorm_orig_dsa3hGTAnp_mk7(mods,"base_chs_lsctsp_2x_",maxT_chs,capacity,feat_ch,tr_meta_path_chs,wemb=0);
    mods=arm_lsctsp_module_set_r45_trinorm_orig_dsa3hGTAnp_va9_mk7(mods,"base_chs_lsctsp_2x_va9r_",maxT_chs,capacity,feat_ch,tr_meta_path_chs,wemb=0);
    return mods;

def dan_single_model_train_cfg(save_root,dsroot,
                               log_path,log_each,itrk= "Top Nep",bsize=48,tvitr=200000):
    maxT_chs=30;
    maxT_mjst=25;

    tr_meta_path_chsjap,te_meta_path_chsjap,tr_meta_path_mjst,te_meta_path_mjst,train_joint_ds=get_dsssch(dsroot,maxT_mjst,maxT_chs,bsize,pfac=2);
    mjst_eval_ds, chs_eval_ds=get_teds(dsroot,maxT_mjst,maxT_chs)
    task_dict = {}
    # task_dict = arm_base_task_default2(task_dict, "base_mjst_", osdanmk7_eval_routine_cfg, maxT_mjst, te_meta_path_mjst,mjst_eval_ds , log_path);
    task_dict = arm_base_task_default2(task_dict, "base_chs_lsctsp_2x_", osdanmk7_eval_routine_cfg, maxT_chs, te_meta_path_chsjap, chs_eval_ds,
                                      log_path);
    task_dict = arm_base_task_default2(task_dict, "base_chs_lsctsp_2x_va9r_", osdanmk7_eval_routine_cfg, maxT_chs, te_meta_path_chsjap,
                                       chs_eval_ds,
                                       log_path);

    #task_dict = arm_base_task_default2(task_dict, "base_chs_lsct_4x_", osdanmk7_eval_routine_cfg, maxT_chs, te_meta_path_chsjap, chs_eval_ds,
#                                      log_path);

    routines = {};
    routines = arm_lsct_mk7_va9_routine(routines, "base_chs_lsctsp_2x_va9r_", osdanmk7_va9r_ocr_routine, maxT_chs, osfsl_ocr_routine,
                                        log_path,
                                        log_each, "dan_chs_", lsct_proto_name="lsct_prototyper");
    routines = arm_lsct_mk7_routine(routines, "base_chs_lsctsp_2x_", osdanmk7_ocr_routine, maxT_chs,osfsl_ocr_routine, log_path,
                                log_each, "dan_chs_",lsct_proto_name="lsct_prototyper");
    #routines = arm_lsct_mk7_routine(routines, "base_chs_lsct_4x_", osdanmk7_ocr_routine, maxT_chs,osfsl_ocr_routine, log_path,
     #                             log_each, "dan_chs_",lsct_proto_name="prototyper");

    return \
        {
            "root": save_root,
            "val_each": 10000,
            "vitr": 200000,
            "vepoch":  4,
            "iterkey": itrk,  # something makes no sense to start fresh
            "dataloader_cfg":train_joint_ds,
            # make sure the unseen characters are unseen.
            "modules": model_mod_cfg(tr_meta_path_chsjap,tr_meta_path_mjst, maxT_mjst,maxT_chs),
            "routine_cfgs": routines,
            "tasks": task_dict,
        }
