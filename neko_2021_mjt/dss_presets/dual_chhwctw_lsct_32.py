from neko_2021_mjt.dataloaders.neko_joint_loader import neko_joint_loader
from neko_2021_mjt.configs.data.ctwch_data import get_stdctwchtr,get_stdctwchte,get_ctw_tr_meta,get_ctw_te_meta,get_chs_ctwS,get_eval_chs_ctwS;
from neko_2021_mjt.configs.data.hwdbch_data import get_stdhwdbtr,get_stdhwdbte,get_hwdb_tr_meta,get_hwdb_te_meta,get_chs_hwdbS,get_eval_chs_hwdbS;
from neko_2021_mjt.configs.data.large_scale_synch_data import get_LSsynch

from neko_2021_mjt.eval_tasks.dan_eval_tasks import neko_odan_eval_tasks
import os


def get_dataloadercfgs(root,chcnt,bsize=160,pfac=1):
    te_meta_path_ctw = get_ctw_te_meta(root);
    te_meta_path_hwdb = get_hwdb_te_meta(root);

    return \
    {

        "loadertype": neko_joint_loader,
        "subsets":
        {
            # reset iterator gives deadlock, so we give a large enough repeat number
            "ctw":get_chs_ctwS(root,chcnt,True,bsize),
            "hwdb": get_chs_hwdbS(root, chcnt,True,bsize),
            "lscs": get_LSsynch(root, [te_meta_path_ctw,te_meta_path_hwdb], max(1, bsize // 96), None, prosize=[32, 32], msize=[64, 64],
                                pcnt=int(128 * pfac),
                                ucnt=int(16 * pfac), batch_size=int(128 * pfac), vendor_name="charset_lmdb_oss")
        }
    }
def get_eval_dss(dsroot):
    te_meta_path_hwdb = get_hwdb_te_meta(dsroot);
    te_meta_path_ctw = get_ctw_te_meta(dsroot);
    hwdb_eval_ds = get_eval_chs_hwdbS(dsroot,True);
    ctw_eval_ds = get_eval_chs_ctwS(dsroot,True);
    return te_meta_path_hwdb,te_meta_path_ctw,hwdb_eval_ds,ctw_eval_ds;

def get_dss(dsroot,chcnt,bsize=160,pfac=1):
    te_meta_path_hwdb,te_meta_path_ctw,hwdb_eval_ds,ctw_eval_ds=\
        get_eval_dss(dsroot);

    tr_meta_path_ctwch = get_ctw_tr_meta(dsroot,chcnt);
    tr_meta_path_hwdb=get_hwdb_tr_meta(dsroot,chcnt);
    train_joint_ds=get_dataloadercfgs(dsroot,chcnt,bsize,pfac=pfac);

    return tr_meta_path_ctwch,tr_meta_path_hwdb,te_meta_path_hwdb,te_meta_path_ctw,hwdb_eval_ds,ctw_eval_ds,train_joint_ds
