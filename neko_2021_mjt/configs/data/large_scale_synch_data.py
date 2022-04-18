import torch

from neko_2020nocr.dan.methods_pami.pami_osds_paths import get_synchtr
from neko_2021_mjt.dataloaders.neko_side_task_chr_loader import character_proto_task_dataset,character_proto_task_datasetg2,collate_fn

from torchvision import transforms
import os;
# pure sideload task. No test branch available


def get_LSsynch(root,avoid_meta,shot=1,include_meta=None,vendor_name="charset_lmdb",dsize=[32,32],prosize=[32,32],msize=[32,32],pcnt=320,ucnt=32,ccnt=0,batch_size=256,inclmode=False):
    if(not inclmode):
        if(avoid_meta is not None):
            if(type(avoid_meta) is list):
                cta=[];
                for m in avoid_meta:
                    m = torch.load(m);
                    cta += m["achars"][1:-1];
                cta=list(set(cta));
            else:
                m=torch.load(avoid_meta);
                cta=m["achars"][1:-1];
        else:
            cta={}
        if(include_meta is not None):
            m = torch.load(include_meta);
            cta = set(cta)-set(m["achars"][1:-1]);

        croot=os.path.join(root,vendor_name);
        broot=os.path.join(root,"synth_data","bgim");
        return \
        {
            "type": character_proto_task_dataset,
            'ds_args': {
                "charset_root":croot,
                "chars_to_avoid": cta,
                "bglist":broot,
                "batch_size": batch_size,# dsize, shots
                "pcnt":pcnt,
                "ucnt": ucnt,
                "ccnt":ccnt,
                "shots":shot,
                "dsize": dsize,
                "prosize" : prosize,
                "msize": msize
            },
            "dl_args":
                {
                    "collate_fn":collate_fn,
                    'batch_size': 1,
                    'num_workers': 16,
                }
        }
    else:
        m = torch.load(include_meta);
        cti = set(m["achars"][1:-1]);
        croot=os.path.join(root,vendor_name);
        broot=os.path.join(root,"synth_data","bgim");
        return \
        {
            "type": character_proto_task_dataset,
            'ds_args': {
                "charset_root":croot,
                "chars_to_avoid": None,
                "bglist":broot,
                "batch_size": batch_size,# dsize, shots
                "pcnt":pcnt,
                "ucnt": ucnt,
                "ccnt":ccnt,
                "shots":shot,
                "dsize": dsize,
                "prosize" : prosize,
                "msize": msize,
                "chars_to_include": cti,
            },
            "dl_args":
                {
                    "collate_fn":collate_fn,
                    'batch_size': 1,
                    'num_workers': 16,
                }
        }

def get_LSsynchG2(root,avoid_meta,shot=1,include_meta=None,vendor_name="charset_lmdb",
                  dsize=[32,32],prosize=[32,32],pcnt=320,ucnt=32,batch_size=256):
    if(avoid_meta is not None):
        m=torch.load(avoid_meta);
        cta=m["achars"][1:-1];
    else:
        cta={}
    if(include_meta is not None):
        m = torch.load(include_meta);
        cta = set(cta)-set(m["achars"][1:-1]);

    croot=os.path.join(root,vendor_name);
    broot=os.path.join(root,"synth_data","bgim");
    return \
    {
        "type": character_proto_task_datasetg2,
        'ds_args': {
            "charset_root":croot,
            "chars_to_avoid": cta,
            "bglist":broot,
            "batch_size": batch_size,# dsize, shots
            "pcnt":pcnt,
            "ucnt": ucnt,
            "shots":shot,
            "dsize": dsize,
            "prosize" : prosize
        },
        "dl_args":
            {
                "collate_fn":collate_fn,
                'batch_size': 1,
                'num_workers': 16,
            }
    }
def get_LSsynch_eval(root,avoid_meta,shot=1,include_meta=None,vendor_name="charset_lmdb_oss",
                     dsize=[32, 32], prosize=[32, 32], pcnt=32, batch_size=32):
    if(avoid_meta is not None):
        m=torch.load(avoid_meta);
        cta=m["achars"][1:-1];
    else:
        cta=[];
    if(include_meta is not None):
        m = torch.load(include_meta);
        cta = set(cta)-set(m["achars"][1:-1]);

    croot=os.path.join(root,vendor_name);
    broot=os.path.join(root,"synth_data","bgim");
    return \
    {
        "type": character_proto_task_dataset,
        'ds_args': {
            "charset_root":croot,
            "chars_to_avoid": cta,
            "bglist":broot,
            "batch_size": 256,# dsize, shots
            "pcnt":pcnt,
            "ucnt": 0,
            "shots":shot,
            "dsize": dsize,
            "prosize" : prosize
        },
        "dl_args":
            {
                "collate_fn":collate_fn,
                'batch_size': batch_size,
                'num_workers': 4,
            }
    }

from neko_2021_mjt.dataloaders.neko_side_task_chr_loader import character_proto_task_dataset_incl
def get_LSsynch_incl(root,shot=1,include_meta=None,vendor_name="charset_lmdb",dsize=[32,32],prosize=[32,32],pcnt=320,ucnt=32,batch_size=256):
    cti=set()
    if(include_meta is not None):
        m = torch.load(include_meta);
        cti=set(m["achars"][1:-1]);

    croot=os.path.join(root,vendor_name);
    broot=os.path.join(root,"synth_data","bgim");
    return \
    {
        "type": character_proto_task_dataset_incl,
        'ds_args': {
            "charset_root":croot,
            "chars_to_incl": cti,
            "bglist":broot,
            "batch_size": batch_size,# dsize, shots
            "pcnt":pcnt,
            "ucnt": ucnt,
            "shots":shot,
            "dsize": dsize,
            "prosize" : prosize
        },
        "dl_args":
            {
                "collate_fn":collate_fn,
                'batch_size': 1,
                'num_workers': 20,
            }
    }