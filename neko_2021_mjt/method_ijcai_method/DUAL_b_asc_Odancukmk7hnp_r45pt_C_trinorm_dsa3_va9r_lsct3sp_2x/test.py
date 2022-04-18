from eval_configs import dan_mjst_eval_cfg
from neko_sdk.root import find_model_root,find_export_root
if __name__ == '__main__':
    import sys
    if(len(sys.argv)<2):
        argv = ["",
                find_export_root(),
                "_E4",
                find_model_root()+"DUAL_b_asc_Odancukmk7hnp_r45pt_C_trinorm_dsa3_va9r_lsct3sp_2x",
                ]
    else:
        argv=sys.argv;
    from neko_2021_mjt.lanuch_std_test import launchtest
    launchtest(argv,dan_mjst_eval_cfg)
