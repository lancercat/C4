from neko_2021_mjt.dataloaders.neko_joint_loader import neko_joint_loader
from neko_2021_mjt.configs.data.mjst_data import get_mjstcqa_cfg,get_test_all_uncased_dsrgb,get_test_all_uncased_dsrgb_cute5k
from neko_2021_mjt.eval_tasks.dan_eval_tasks import neko_odan_eval_tasks
from neko_2021_mjt.configs.data.chs_jap_data import get_chs_HScqa,get_eval_jap_color,\
    get_chs_tr_meta,get_chs_sc_meta,\
    get_jap_te_meta,get_eval_monkey_color,\
    get_jap_te_metagosr,get_jap_te_metaosr,\
    get_eval_kr_color,get_kr_te_meta
import os

def get_dataloadercfgs(root,te_meta_path,tr_meta_path,maxT_mjst,maxT_chsHS,bsize):
    return \
    {
        "loadertype": neko_joint_loader,
        "subsets":
        {
            # reset iterator gives deadlock, so we give a large enough repeat number
            "dan_mjst":get_mjstcqa_cfg(root, maxT_mjst, bs=bsize,hw=[32,128]),
            "dan_chs": get_chs_HScqa(root, maxT_chsHS, bsize, -1,hw=[32,128]),
        }
    }
def get_dataloadercfgsch(root,te_meta_path,tr_meta_path,maxT_mjst,maxT_chsHS,bsize):
    return \
    {
        "loadertype": neko_joint_loader,
        "subsets":
        {
            # reset iterator gives deadlock, so we give a large enough repeat number
            # "dan_mjst":get_mjstcqa_cfg(root, maxT_mjst, bs=bsize,hw=[32,128]),
            "dan_chs": get_chs_HScqa(root, maxT_chsHS, bsize, -1,hw=[32,128]),
        }
    }
def get_dataloadercfgsms(root,te_meta_path,tr_meta_path,maxT_mjst,maxT_chsHS,bsize):
    return \
    {
        "loadertype": neko_joint_loader,
        "subsets":
        {
            # reset iterator gives deadlock, so we give a large enough repeat number
            "dan_mjst":get_mjstcqa_cfg(root, maxT_mjst, bs=bsize,hw=[32,128]),
            #"dan_chs": get_chs_HScqa(root, maxT_chsHS, bsize, -1,hw=[32,128]),
        }
    }
def get_eval_dss(dsroot,maxT_mjst,maxT_chs,batch_size=1):
    te_meta_path_chsjap = get_jap_te_meta(dsroot);
    te_meta_path_mjst = os.path.join(dsroot, "dicts", "dab62cased.pt");
    mjst_eval_ds = get_test_all_uncased_dsrgb(maxT_mjst, dsroot, None,hw=[32,128],batchsize=batch_size)
    chs_eval_ds = get_eval_jap_color(dsroot, maxT_chs,hw=[32,128]);
    return te_meta_path_chsjap,te_meta_path_mjst,mjst_eval_ds,chs_eval_ds
def get_eval_dssenjp(dsroot,maxT_mjst,maxT_chs,batch_size=1):
    te_meta_path_chsjap = get_jap_te_meta(dsroot);
    te_meta_path_mjst = os.path.join(dsroot, "dicts", "dab62cased.pt");
    mjst_eval_ds = get_test_all_uncased_dsrgb_cute5k(maxT_mjst, dsroot, None,hw=[32,128],batchsize=batch_size)
    chs_eval_ds = get_eval_jap_color(dsroot, maxT_chs,hw=[32,128]);
    return te_meta_path_chsjap,te_meta_path_mjst,mjst_eval_ds,chs_eval_ds
def get_eval_dssgosr(dsroot,maxT_mjst,maxT_chs):
    te_meta_path_chsjap = get_jap_te_metagosr(dsroot);
    te_meta_path_mjst = os.path.join(dsroot, "dicts", "dab62cased.pt");
    mjst_eval_ds = get_test_all_uncased_dsrgb(maxT_mjst, dsroot, None, 16,hw=[32,128])
    chs_eval_ds = get_eval_jap_color(dsroot, maxT_chs,hw=[32,128]);
    return te_meta_path_chsjap,te_meta_path_mjst,mjst_eval_ds,chs_eval_ds
def get_eval_dssosr(dsroot,maxT_mjst,maxT_chs):
    te_meta_path_chsjap = get_jap_te_metaosr(dsroot);
    te_meta_path_mjst = os.path.join(dsroot, "dicts", "dab62cased.pt");
    mjst_eval_ds = get_test_all_uncased_dsrgb(maxT_mjst, dsroot, None, 16,hw=[32,128])
    chs_eval_ds = get_eval_jap_color(dsroot, maxT_chs,hw=[32,128]);
    return te_meta_path_chsjap,te_meta_path_mjst,mjst_eval_ds,chs_eval_ds
def get_bench_dss_all(dsroot,maxT_mjst,maxT_chs,batch_size=1):
    te_meta_path_chsjap = get_jap_te_meta(dsroot);
    te_meta_path_mjst = os.path.join(dsroot, "dicts", "dab62cased.pt");
    mjst_eval_ds = get_test_all_uncased_dsrgb_all(maxT_mjst, dsroot, None, batch_size,hw=[32,128])
    chs_eval_ds = get_eval_jap_color(dsroot, maxT_chs,hw=[32,128]);
    return te_meta_path_chsjap,te_meta_path_mjst,mjst_eval_ds,chs_eval_ds

def get_bench_dss(dsroot,maxT_mjst,maxT_chs,batch_size=1):
    te_meta_path_chsjap = get_jap_te_meta(dsroot);
    te_meta_path_mjst = os.path.join(dsroot, "dicts", "dab62cased.pt");
    mjst_eval_ds = get_test_all_uncased_dsrgb(maxT_mjst, dsroot, None, batch_size,hw=[32,128])
    chs_eval_ds = get_eval_jap_color(dsroot, maxT_chs,hw=[32,128]);
    return te_meta_path_chsjap,te_meta_path_mjst,mjst_eval_ds,chs_eval_ds
def get_eval_dss_kr(dsroot,maxT_mjst,maxT_chs):
    te_meta_path_chsjap = get_kr_te_meta(dsroot);
    te_meta_path_mjst = os.path.join(dsroot, "dicts", "dab62cased.pt");
    mjst_eval_ds = get_test_all_uncased_dsrgb(maxT_mjst, dsroot, None,hw=[32,128])
    chs_eval_ds = get_eval_kr_color(dsroot, maxT_chs,hw=[32,128]);

    return te_meta_path_chsjap,te_meta_path_mjst,mjst_eval_ds,chs_eval_ds
# Howdy. So you know that we have our eyes on the Hindi and Bangla!
# def get_eval_dss_ba(dsroot,maxT_mjst,maxT_chs):
#     te_meta_path_chsjap = get_be_te_meta(dsroot);
#     te_meta_path_mjst = os.path.join(dsroot, "dicts", "dab62cased.pt");
#     mjst_eval_ds = get_test_all_uncased_dsrgb(maxT_mjst, dsroot, None, 16,hw=[32,128])
#     chs_eval_ds = get_eval_be_color(dsroot, maxT_cshs,hw=[32,128]);
#     return te_meta_path_chsjap,te_meta_path_mjst,mjst_eval_ds,chs_eval_ds
#
# def get_eval_dss_hn(dsroot,maxT_mjst,maxT_chs):
#     te_meta_path_chsjap = get_hn_te_meta(dsroot);
#     te_meta_path_mjst = os.path.join(dsroot, "dicts", "dab62cased.pt");
#     mjst_eval_ds = get_test_all_uncased_dsrgb(maxT_mjst, dsroot, None, 16,hw=[32,128])
#     chs_eval_ds = get_eval_hn_color(dsroot, maxT_chs,hw=[32,128]);
#     return te_meta_path_chsjap,te_meta_path_mjst,mjst_eval_ds,chs_eval_ds
#

def get_dss(dsroot,maxT_mjst,maxT_chs,bsize):
    te_meta_path_chsjap, te_meta_path_mjst, mjst_eval_ds, chs_eval_ds=\
        get_eval_dss(dsroot,maxT_mjst,maxT_chs);
    tr_meta_path_chsjap = get_chs_tr_meta(dsroot);
    tr_meta_path_mjst = os.path.join(dsroot, "dicts", "dab62cased.pt");
    train_joint_ds=get_dataloadercfgs(dsroot,te_meta_path_mjst,tr_meta_path_chsjap,maxT_mjst,maxT_chs,bsize);
    return tr_meta_path_chsjap,te_meta_path_chsjap,tr_meta_path_mjst,te_meta_path_mjst,mjst_eval_ds,chs_eval_ds,train_joint_ds


def get_eval_dss_m(dsroot,maxT_mjst,maxT_chs,lang="chs"):
    te_meta_path_chsjap = get_jap_te_meta(dsroot);
    te_meta_path_mjst = os.path.join(dsroot, "dicts", "dab62cased.pt");
    mjst_eval_ds = get_test_all_uncased_dsrgb(maxT_mjst, dsroot, None, 16,hw=[32,128])
    japm_eval_ds = get_eval_monkey_color(dsroot, maxT_chs,lang,hw=[32,128]);

    return te_meta_path_chsjap,te_meta_path_mjst,mjst_eval_ds,japm_eval_ds



def get_dsssc(dsroot,maxT_mjst,maxT_chs,bsize):
    te_meta_path_chsjap, te_meta_path_mjst, mjst_eval_ds, chs_eval_ds=\
        get_eval_dss(dsroot,maxT_mjst,maxT_chs);
    tr_meta_path_chsjap = get_chs_sc_meta(dsroot);
    tr_meta_path_mjst = os.path.join(dsroot, "dicts", "dab62cased.pt");
    train_joint_ds=get_dataloadercfgs(dsroot,te_meta_path_mjst,tr_meta_path_chsjap,maxT_mjst,maxT_chs,bsize);
    return tr_meta_path_chsjap,te_meta_path_chsjap,tr_meta_path_mjst,te_meta_path_mjst,mjst_eval_ds,chs_eval_ds,train_joint_ds

def get_dsssch(dsroot,maxT_mjst,maxT_chs,bsize):
    te_meta_path_chsjap, te_meta_path_mjst, mjst_eval_ds, chs_eval_ds=\
        get_eval_dss(dsroot,maxT_mjst,maxT_chs);
    tr_meta_path_chsjap = get_chs_sc_meta(dsroot);
    tr_meta_path_mjst = None;
    train_joint_ds=get_dataloadercfgsch(dsroot,te_meta_path_mjst,tr_meta_path_chsjap,maxT_mjst,maxT_chs,bsize);
    return tr_meta_path_chsjap,te_meta_path_chsjap,tr_meta_path_mjst,te_meta_path_mjst,mjst_eval_ds,chs_eval_ds,train_joint_ds

def get_dssscm(dsroot,maxT_mjst,maxT_chs,bsize):
    te_meta_path_chsjap, te_meta_path_mjst, mjst_eval_ds, chs_eval_ds=\
        get_eval_dss(dsroot,maxT_mjst,maxT_chs);
    tr_meta_path_chsjap =None;
    tr_meta_path_mjst = os.path.join(dsroot, "dicts", "dab62cased.pt");
    train_joint_ds=get_dataloadercfgsms(dsroot,te_meta_path_mjst,tr_meta_path_chsjap,maxT_mjst,maxT_chs,bsize);
    return tr_meta_path_chsjap,te_meta_path_chsjap,tr_meta_path_mjst,te_meta_path_mjst,mjst_eval_ds,chs_eval_ds,train_joint_ds
