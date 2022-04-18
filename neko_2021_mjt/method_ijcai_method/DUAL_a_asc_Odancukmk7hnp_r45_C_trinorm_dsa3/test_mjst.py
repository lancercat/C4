from eval_configs_mjst import dan_mjst_eval_cfg
# One for all, all for one

if __name__ == '__main__':
    import sys
    if(len(sys.argv)<2):
        argv = ["Meeeeooooowwww",
                "/thumbnails/advantage/DUAL_a_asc_Odancukmk7hnp_r45_C_trinorm_dsa3/",
                "_E0",
                "/home/lasercat/cat/c4-models/DUAL_a_asc_Odancukmk7hnp_r45_C_trinorm_dsa3/jtrmodels/",
                ]
    else:
        argv=sys.argv;
    from neko_2021_mjt.lanuch_std_test import launchtest
    launchtest(argv,dan_mjst_eval_cfg)
