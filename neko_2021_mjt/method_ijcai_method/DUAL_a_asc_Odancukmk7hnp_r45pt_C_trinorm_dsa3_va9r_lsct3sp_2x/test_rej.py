from eval_configs_rej import dan_mjst_eval_cfg

if __name__ == '__main__':
    import sys
    if(len(sys.argv)<2):
        argv = ["Meeeeooooowwww",
                None,#"/home/lasercat/cat/c4-models/g2/DUAL_a_asc_Odancukmk7hnp_r45pt_C_trinorm_dsa3_va9r_lsct3sp_2x/",
                "_E0",
                "/home/lasercat/cat/c4-models/g2/DUAL_a_asc_Odancukmk7hnp_r45pt_C_trinorm_dsa3_va9r_lsct3sp_2x/",
                ]
    else:
        argv=sys.argv;
    from neko_2021_mjt.lanuch_std_test import launchtest
    launchtest(argv,dan_mjst_eval_cfg)
