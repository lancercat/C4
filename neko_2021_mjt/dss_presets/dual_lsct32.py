from neko_2021_mjt.dataloaders.neko_joint_loader import neko_joint_loader
from neko_2021_mjt.configs.data.mjst_data import get_mjstcqa_cfg,get_test_all_uncased_dsrgb
from neko_2021_mjt.eval_tasks.dan_eval_tasks import neko_odan_eval_tasks
from neko_2021_mjt.configs.data.chs_jap_data import get_chs_HScqa,get_eval_jap_color,get_chs_tr_meta,get_chs_sc_meta,get_jap_te_meta,get_eval_monkey_color
import os
from neko_2021_mjt.configs.data.large_scale_synch_data import get_LSsynch

def get_dataloadercfgs(root,te_meta_path,tr_meta_path,maxT_mjst,maxT_chsHS,bsize,vendor_name="charset_lmdb_oss"):
    return \
    {
        "loadertype": neko_joint_loader,
        "subsets":
        {
            # reset iterator gives deadlock, so we give a large enough repeat number
            "dan_mjst":get_mjstcqa_cfg(root, maxT_mjst, bs=bsize,hw=[32,128]),
            "dan_chs": get_chs_HScqa(root, maxT_chsHS, bsize, 128,hw=[32,128]),
            "lscs": get_LSsynch(root, te_meta_path, max(1, bsize // 96), tr_meta_path, prosize=[32, 32],msize=[64,64], pcnt=128,
                                ucnt=16, batch_size=128, vendor_name=vendor_name)
        }
    };
def get_dataloadercfgs_ch(root,te_meta_path,tr_meta_path,maxT_mjst,maxT_chsHS,bsize,vendor_name="charset_lmdb_oss",pfac=1):
    return \
    {
        "loadertype": neko_joint_loader,
        "subsets":
        {
            # reset iterator gives deadlock, so we give a large enough repeat number
            # "dan_mjst":get_mjstcqa_cfg(root, maxT_mjst, bs=bsize,hw=[32,128]),
            "dan_chs": get_chs_HScqa(root, maxT_chsHS, bsize, 128,hw=[32,128]),
            "lscs": get_LSsynch(root, te_meta_path, max(1, bsize // 96), tr_meta_path, prosize=[32, 32],msize=[64,64], pcnt=int(128*pfac),
                                ucnt=int(16*pfac), batch_size=int(128*pfac), vendor_name=vendor_name)
        }
    };
def get_dataloadercfgs_ch_nofor(root,te_meta_path,tr_meta_path,maxT_mjst,maxT_chsHS,bsize,vendor_name="charset_lmdb_oss",pfac=1):
    return \
    {
        "loadertype": neko_joint_loader,
        "subsets":
        {
            # reset iterator gives deadlock, so we give a large enough repeat number
            # "dan_mjst":get_mjstcqa_cfg(root, maxT_mjst, bs=bsize,hw=[32,128]),
            "dan_chs": get_chs_HScqa(root, maxT_chsHS, bsize, 128,hw=[32,128]),
            "lscs": get_LSsynch(root, None, max(1, bsize // 96), tr_meta_path, prosize=[32, 32],msize=[64,64], pcnt=int(128*pfac),
                                ucnt=int(16*pfac), batch_size=int(128*pfac), vendor_name=vendor_name,inclmode=True)
        }
    };
def get_dataloadercfgs_m(root,te_meta_path,tr_meta_path,maxT_mjst,maxT_chsHS,bsize,vendor_name="charset_lmdb_oss",pfac=1):
    return \
    {
        "loadertype": neko_joint_loader,
        "subsets":
        {
            # reset iterator gives deadlock, so we give a large enough repeat number
            "dan_mjst":get_mjstcqa_cfg(root, maxT_mjst, bs=bsize,hw=[32,128]),
            # "dan_chs": get_chs_HScqa(root, maxT_chsHS, bsize, 128,hw=[32,128]),
            "lscs": get_LSsynch(root, te_meta_path, max(1, bsize // 96), tr_meta_path, prosize=[32, 32],msize=[64,64], pcnt=int(128*pfac),
                                ucnt=int(16*pfac), batch_size=int(128*pfac), vendor_name=vendor_name)
        }
    };
def get_dss(dsroot,maxT_mjst,maxT_chs,bsize,vendor_name="charset_lmdb_oss"):

    tr_meta_path_chsjap = get_chs_tr_meta(dsroot);
    te_meta_path_chsjap = get_jap_te_meta(dsroot);
    tr_meta_path_mjst = os.path.join(dsroot, "dicts", "dab62cased.pt");
    te_meta_path_mjst = os.path.join(dsroot, "dicts", "dab62cased.pt");

    train_joint_ds=get_dataloadercfgs(dsroot,te_meta_path_mjst,tr_meta_path_chsjap,maxT_mjst,maxT_chs,bsize,vendor_name);
    return tr_meta_path_chsjap,te_meta_path_chsjap,tr_meta_path_mjst,te_meta_path_mjst,train_joint_ds

def get_dsssc(dsroot,maxT_mjst,maxT_chs,bsize,vendor_name="charset_lmdb_oss"):

    tr_meta_path_chsjap = get_chs_sc_meta(dsroot);
    te_meta_path_chsjap = get_jap_te_meta(dsroot);
    tr_meta_path_mjst = os.path.join(dsroot, "dicts", "dab62cased.pt");
    te_meta_path_mjst = os.path.join(dsroot, "dicts", "dab62cased.pt");

    train_joint_ds=get_dataloadercfgs(dsroot,te_meta_path_mjst,tr_meta_path_chsjap,maxT_mjst,maxT_chs,bsize,vendor_name);
    return tr_meta_path_chsjap,te_meta_path_chsjap,tr_meta_path_mjst,te_meta_path_mjst,train_joint_ds

def get_dssm(dsroot,maxT_mjst,maxT_chs,bsize,vendor_name="charset_lmdb_oss",pfac=1):

    tr_meta_path_chsjap = None;
    te_meta_path_chsjap = None;
    tr_meta_path_mjst = os.path.join(dsroot, "dicts", "dab62cased.pt");
    te_meta_path_mjst = os.path.join(dsroot, "dicts", "dab62cased.pt");

    train_joint_ds=get_dataloadercfgs_m(dsroot,te_meta_path_mjst,tr_meta_path_chsjap,maxT_mjst,maxT_chs,bsize,vendor_name,pfac=pfac);
    return tr_meta_path_chsjap,te_meta_path_chsjap,tr_meta_path_mjst,te_meta_path_mjst,train_joint_ds

def get_dsssch(dsroot,maxT_mjst,maxT_chs,bsize,vendor_name="charset_lmdb_oss",pfac=1):

    tr_meta_path_chsjap = get_chs_sc_meta(dsroot);
    te_meta_path_chsjap = get_jap_te_meta(dsroot);
    # tr_meta_path_mjst = os.path.join(dsroot, "dicts", "dab62cased.pt");
    # te_meta_path_mjst = os.path.join(dsroot, "dicts", "dab62cased.pt");
    te_meta_path_mjst=None;
    tr_meta_path_mjst=None;
    train_joint_ds=get_dataloadercfgs_ch(dsroot,te_meta_path_mjst,tr_meta_path_chsjap,maxT_mjst,maxT_chs,bsize,vendor_name,pfac=pfac);
    return tr_meta_path_chsjap,te_meta_path_chsjap,tr_meta_path_mjst,te_meta_path_mjst,train_joint_ds
def get_dsssch_nofor(dsroot,maxT_mjst,maxT_chs,bsize,vendor_name="charset_lmdb_oss",pfac=1):

    tr_meta_path_chsjap = get_chs_sc_meta(dsroot);
    te_meta_path_chsjap = get_jap_te_meta(dsroot);
    # tr_meta_path_mjst = os.path.join(dsroot, "dicts", "dab62cased.pt");
    # te_meta_path_mjst = os.path.join(dsroot, "dicts", "dab62cased.pt");
    te_meta_path_mjst=None;
    tr_meta_path_mjst=None;
    train_joint_ds=get_dataloadercfgs_ch_nofor(dsroot,te_meta_path_mjst,tr_meta_path_chsjap,maxT_mjst,maxT_chs,bsize,vendor_name,pfac=pfac);
    return tr_meta_path_chsjap,te_meta_path_chsjap,tr_meta_path_mjst,te_meta_path_mjst,train_joint_ds
def get_teds(dsroot,maxT_mjst,maxT_chs):
    mjst_eval_ds = get_test_all_uncased_dsrgb(maxT_mjst, dsroot, None, 32, hw=[32, 128])
    chs_eval_ds = get_eval_jap_color(dsroot, maxT_chs, hw=[32, 128]);
    return mjst_eval_ds,chs_eval_ds