from eval_configs import dan_mjst_eval_cfg
from neko_sdk.root import find_model_root,find_export_root
if __name__ == '__main__':
    import sys
    if(len(sys.argv)<2):
        argv = ["",
                find_export_root(),
                "_E0",
                find_model_root()+"ABL_a_asc_2x_lsct3sp_lsct3spva9r",
                ]
    else:
        argv=sys.argv;
    from neko_2021_mjt.lanuch_std_test import launchtest
    launchtest(argv,dan_mjst_eval_cfg)
