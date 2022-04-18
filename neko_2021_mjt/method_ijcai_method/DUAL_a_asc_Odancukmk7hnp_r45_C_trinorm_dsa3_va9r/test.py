from eval_configs import dan_mjst_eval_cfg
def launch(argv):
    from neko_2021_mjt.lanuch_std_test import launchtest
    launchtest(argv,dan_mjst_eval_cfg)
if __name__ == '__main__':
    import sys
    if(len(sys.argv)<2):
        argv = ["Meeeeooooowwww",
                "/home/lasercat/cat/c4-models/DUAL_asc_Odancukmk7hnp_r45_C_trinorm_dsa3_va9r/jtrmodels/",
                "_E0",
                "/home/lasercat/cat/c4-models/DUAL_asc_Odancukmk7hnp_r45_C_trinorm_dsa3_va9r/jtrmodels/",
                ]
    else:
        argv=sys.argv;
    launch(argv);
