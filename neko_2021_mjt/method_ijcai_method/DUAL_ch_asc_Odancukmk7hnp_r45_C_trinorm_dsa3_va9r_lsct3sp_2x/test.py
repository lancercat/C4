from eval_configs import dan_mjst_eval_cfg
from neko_sdk.root import find_export_root,find_model_root

def test(tsz):
    argv = ["",
            find_export_root(),
            "_E3",
            find_model_root() + "DUAL_ch_asc_Odancukmk7hnp_r45_C_trinorm_dsa3_va9r_lsct3sp_2x/jtrmodels"+tsz,
            ]
    from neko_2021_mjt.lanuch_std_test import launchtest
    launchtest(argv,dan_mjst_eval_cfg,miter=100000000);

if __name__ == '__main__':
    for tsz in ["500","1000","1500","2000"]:
        print("SZ",tsz);
        test(tsz);