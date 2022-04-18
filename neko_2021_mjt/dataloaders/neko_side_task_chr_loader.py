import glob
import os.path

import torch

import cv2
import numpy as np
import random
#returns raw image,rectified_image and so on.
#let's make this a small 2-phasemodel
from neko_sdk.lmdb_wrappers.ocr_lmdb_reader import neko_ocr_lmdb_mgmt;
from PIL import Image
from torch.utils.data import DataLoader, Dataset
# we have a few anchors. The model selects on aspect ratios, and the model selects for sort edge size.
from neko_sdk.LSCT.masked_trdg_driver import neko_masked_sample_generator

class character_proto_task_loader:
    def setmvn(this):
        this.sample_mean=0;
        this.sample_var=255;

        this.proto_mean = 1;
        this.proto_var=127.5;

        this.msk_mean=0;
        this.msk_var=255;
    def set_drivers(this,bglist):
        this.bgims = glob.glob(os.path.join(bglist, "*.jpg"));
        this.driver=neko_masked_sample_generator(None);
    def set_batch_size(this,batch_size):
        this.batch_size=this.pcnt+this.ucnt;
    def set_dsize(this,dsize):
        this.dsize = tuple(dsize);

    def __init__(this,charset_root,chars_to_avoid,bglist,batch_size,pcnt,ucnt,ccnt=0,dsize=[32,32],prosize=[32,32],msize=[32.32],chars_to_include=None):
        meta=torch.load(os.path.join(charset_root,"meta.pt"));
        this.setmvn();
        this.meta={};
        this.set_dsize(dsize)
        this.prosize = tuple(prosize);

        this.keyz=[];
        this.pcnt=pcnt;
        this.ucnt=ucnt;
        # grab a chunk for fine-grind classification.
        this.ccnt=ccnt;
        # some "characters" can be stupidly loong.
        this.db=neko_ocr_lmdb_mgmt(charset_root,True,10);
        if(chars_to_include is None):
            for c in meta:
                if(c not in chars_to_avoid):
                    this.meta[c]= meta[c];
                    this.keyz.append(c);
        else:
            for c in chars_to_include:
                if (c in meta):
                    this.meta[c] = meta[c];
                    this.keyz.append(c);
        if (ucnt + pcnt > len(this.keyz)):
            this.ucnt = int(len(this.keyz) * 0.2);
            this.pcnt = len(this.keyz) - this.ucnt;
        this.set_drivers(bglist);
        this.set_batch_size(batch_size);
    def get_random(this,ch):
        name = this.meta[ch];
        item = random.choice(name);
        font = this.db.get_encoded_im_by_name(item);
        return cv2.resize( np.array(font), this.prosize);
    def set_up(this):
        if (this.ccnt != 0):
            sta = random.randint(0, len(this.keyz) - this.ccnt - 1);
            end = sta + this.ccnt;
            rem = this.keyz[:sta] + this.keyz[end:]
            bch = this.keyz[sta:end] + random.sample(rem, this.ucnt + this.pcnt - this.ccnt);
        else:
            bch = random.sample(this.keyz, this.ucnt + this.pcnt);
        random.shuffle(bch);
        tdict={this.pcnt:"[-]","[UNK]":this.pcnt};
        protos=[];
        plabels=torch.arange(this.pcnt+1);
        for i in range(this.pcnt):
            ch = bch[this.ucnt + i];
            protos.append(this.get_random(ch));
            tdict[ch] = i;
            tdict[i] = ch;
        try:
            bgim = random.choice(this.bgims)
            bgim = Image.open(bgim);
        except:
            bgim = "None";
        return bch,protos,plabels,tdict,bgim;
    def batch_charset(this,shots):
        # find where we sample
        bch, protos, plabels, tdict, bgim=this.set_up();
        samples=[];
        labels=[];
        sample_masks=[];
        for i in range(this.pcnt+this.ucnt):
            ch=bch[i];
            # use different part of image as background to reduce io load.
            for t in range(shots):
                f=this.get_random(ch);
                s,m=this.driver.get_samplewm(f,bgim)

                s=cv2.resize(np.array(s),this.dsize);
                samples.append(s);

                m = cv2.resize(np.array(m), this.prosize);
                sample_masks.append(m);

                if(ch in tdict):
                    labels.append(tdict[ch]);
                else:
                    labels.append(this.pcnt);
        return {
            "samples": torch.tensor(np.array(samples)).permute(0,3,1,2)/this.sample_var-this.sample_mean,
            "labels": torch.tensor(labels),
            "protos":torch.tensor(np.array(protos)).permute(0,3,1,2)/this.proto_var-this.proto_mean,
            "tdict":tdict,
            "smsk": torch.tensor(np.array(sample_masks)).permute(0,3,1,2)/this.msk_var-this.msk_mean,
        }
# we unify the normalization process and what the loader provides
# we also provide global labels for semantic branch.
class character_proto_task_loaderg2(character_proto_task_loader):
    def batch_charset(this,shots):
        bch=random.sample(this.keyz,this.ucnt+this.pcnt);
        random.shuffle(bch);
        protos=[];
        samples=[];
        labels=[];
        sample_masks=[];
        tdict={this.pcnt:"[-]","[UNK]":this.pcnt};
        plabels=[];
        for i in range(this.pcnt):
            ch=bch[this.ucnt+i];
            protos.append(this.get_random(ch));
            plabels.append(i);
            tdict[ch]=i;
            tdict[i]=ch;
        plabels.append(this.pcnt);
        try:
            bgim=random.choice(this.bgims)
            bgim = Image.open(bgim);
        except:
            bgim="None";
        for i in range(this.batch_size):
            ch=bch[i];
            # use different part of image as background to reduce io load.
            for t in range(shots):
                f=this.get_random(ch);
                s,m=this.driver.get_samplewm(f,bgim)

                s=cv2.resize(np.array(s),this.dsize);
                samples.append(s);

                m = cv2.resize(np.array(m), this.prosize);
                sample_masks.append(m);

                if(ch in tdict):
                    labels.append(tdict[ch]);
                else:
                    labels.append(this.pcnt);
        return {
            "samples": torch.tensor(samples).permute(0,3,1,2)/127.5-1,
            "labels": torch.tensor(labels),
            "protos":torch.tensor(np.array(protos)).permute(0,3,1,2)/127.5-1,
            "plabels":torch.tensor(plabels),
            "tdict":tdict,
            "smsk": torch.tensor(sample_masks).permute(0,3,1,2)/127.5-1,
        }

class character_proto_task_loader_incl(character_proto_task_loader):
    def __init__(this,charset_root,chars_to_incl,bglist,batch_size,pcnt,ucnt,dsize=[32,32],prosize=[32,32]):
        meta=torch.load(os.path.join(charset_root,"meta.pt"));
        this.meta={};
        this.dsize=tuple(dsize);
        this.prosize = tuple(prosize);

        this.keyz=[];
        this.pcnt=pcnt;
        this.ucnt=ucnt;
        # some "characters" can be stupidly loong.
        this.db=neko_ocr_lmdb_mgmt(charset_root,True,10);
        for c in meta:
            if(c in chars_to_incl):
                this.meta[c]= meta[c];
                this.keyz.append(c);
        if (ucnt + pcnt > len(this.keyz)):
            this.ucnt = int(len(this.keyz) * 0.2);
            this.pcnt = len(this.keyz) - this.ucnt;
        this.driver=neko_masked_sample_generator(bglist);
        this.batch_size=this.pcnt+this.ucnt;


class character_proto_task_dataset_incl(Dataset):
    def __init__(this,charset_root,chars_to_incl,bglist,batch_size,pcnt,ucnt,dsize,prosize,shots,vlen=20000):
        this.shots=shots;
        this.vlen=vlen;
        this.core=character_proto_task_loader_incl(charset_root,chars_to_incl,bglist,batch_size,pcnt,ucnt,dsize,prosize);
    def __len__(this):
        return 2147483647;
    def __getitem__(this,idx) :
        return this.core.batch_charset(this.shots);

class character_proto_task_dataset(Dataset):
    def __init__(this,charset_root,chars_to_avoid,bglist,batch_size,pcnt,ucnt,ccnt,dsize,prosize,shots,msize,chars_to_include=None):
        this.shots=shots;
        this.core=character_proto_task_loader(charset_root,chars_to_avoid,bglist,batch_size,pcnt,ucnt,ccnt,dsize,prosize,msize,chars_to_include=chars_to_include);
    def __len__(this):
        return 2147483647;
    def __getitem__(this,idx) :
        return this.core.batch_charset(this.shots);
class character_proto_task_datasetg2(Dataset):
    def __init__(this,charset_root,chars_to_avoid,bglist,batch_size,pcnt,ucnt,dsize,prosize,shots):
        this.shots=shots;
        this.core=character_proto_task_loaderg2(charset_root,chars_to_avoid,bglist,batch_size,pcnt,ucnt,dsize,prosize);
    def __len__(this):
        return 2147483647;
    def __getitem__(this,idx) :
        return this.core.batch_charset(this.shots);

def show_batch(sample,label,proto,tdict,smsk):
    sp=(sample.permute(0,2,3,1)+1)*127.5;
    mp=(smsk.permute(0,2,3,1))*255;
    pp=(proto.permute(0,2,3,1)+1)*127.5;
    mps=[];
    sps=[];
    pps=[];
    for i in range(len(sp)):
        mps.append(mp[i]);
        sps.append(sp[i]);
        l=label[i].item();
        if(l < len(proto)):
            pps.append(pp[l]);
        else:
            pps.append(torch.zeros_like(pp[0]));
    v=torch.cat([torch.cat(mps,1),torch.cat(sps,1),torch.cat(pps,1)],0);
    v=v.detach().cpu().numpy()[:,:,::-1].astype(np.uint8);

    print([tdict[i.item()] for i in label])
    cv2.namedWindow("w",0);
    cv2.imshow("w",v);
    cv2.waitKey(0)
def collate_fn(d):
    return d[0]
if __name__ == '__main__':
    l = character_proto_task_dataset("/home/lasercat/ssddata/charset_lmdb/", "",
                                    "/home/lasercat/ssddata/synth_data/bgim", 16, 32, 4,[32,32],2);
    dl=DataLoader(l,collate_fn=collate_fn,num_workers=0);
    for d in dl:
        show_batch(d["samples"], d["labels"], d["protos"], d["tdict"],d["smsk"])
        pass;


#
# if __name__ == '__main__':
#     #
#
#     l=character_proto_task_loader("/home/lasercat/ssddata/charset_lmdb/","","/home/lasercat/ssddata/synth_data/bgim",16,32,4);
#     for i in range(19):
#         d=l.batch_charset(1);
#
#         pass;


