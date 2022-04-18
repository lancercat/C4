import torch;
from torch.nn import functional as trnf;
from neko_sdk.torchtools.ts import scatter_mean;
from neko_2021_mjt.routines.ocr_routines.mk5.osdan_routine_mk5_ga_va6 import padding_tensor
from neko_2021_mjt.routines.subroutines.fe_seq.GTA import dump_att_im

import cv2
import numpy as np


def debugva(nim):
    a=(nim.permute(1,2,0).reshape(32,128,3).detach().cpu().numpy()*255).astype(np.uint8);
    return a;
def vashow(dbgprfx,vim,GA,TA,p_len,label,pred):
    for i in range(vim.shape[0]):
        ti = dump_att_im(vim[i], GA[i], TA[i], p_len[i], None);
        mi = debugva(vim[i]);
        cv2.namedWindow("vim" + dbgprfx, 0);
        # cv2.namedWindow("a" + dbgprfx, 0);
        # cv2.namedWindow("b" + dbgprfx, 0);
        cv2.imshow("vim" + dbgprfx, ti);
        # cv2.imshow("a" + dbgprfx, mi);
        # cv2.imshow("b" + dbgprfx, mi2);
        print(label[i],"->",pred[i]);
        cv2.waitKey(30);

def detail_va8(dbgprfx,GA,TA,tdict,vim,vlogits,vlabel_flatten,vlength):
    vga, vta = torch.cat([GA, GA]), torch.cat([TA, TA])
    pred = [tdict[i.item()] for i in vlogits.max(dim=1)[1]];
    gt = [tdict[i.item()] for i in vlabel_flatten];
    for i in range(len(pred)):
        print(gt[i], "->", pred[i]);
    lvlen = [i.item() for i in vlength];
    sta = 0;
    pws, gws = [], [];
    for i in range(len(lvlen)):
        beg = sta;
        end = sta + lvlen[i];
        sta = end;
        pw = "".join(pred[beg:end]);
        gw = "".join(gt[beg:end]);
        pws.append(pw);
        gws.append(gw);
    vashow(dbgprfx, vim, vga, vta, vlength, gws,pws);
def vflatten_label_idx(target,length):
    label_flatten = []
    label_length = []
    src=[];
    for i in range(0, target.size()[0]):
        l=length[i].item();
        cur_label = target[i].tolist()
        label_flatten += cur_label[:l]
        label_length.append(l)
        src +=[i for _ in range(l)]
    label_flatten = torch.LongTensor(label_flatten)
    label_length = torch.IntTensor(label_length)
    src=torch.LongTensor(src);
    return (label_flatten, label_length,src)

class neko_validated_attention_mk8_f:
    def __init__(this,vcnt,inflater,lossname="losses"):
        this.inflater=inflater;
        this.vcnt=vcnt;
    def loss(this,module_dict,proto,vlogits,vlabel_flatten):
        loss_, terms_ = module_dict["valoss"][0](proto, vlogits,vlabel_flatten);
        return loss_,terms_;
    def relabel(this, A, target,acr, length, mlabel=-1):
        T=target.shape[1];
        A_=A[:,:T];
        # well, we first hack the label to remove a part
        # lmask = (torch.arange(A_.shape[1])[None,] < length.unsqueeze(-1)).float();
        # We cannot be messing with the EOS
        amask = (torch.arange(A_.shape[1],device=acr.device)[None,] < (length).unsqueeze(-1)).float();
        # we do not mess with error predictions, either.
        camask=amask*acr;
         # mask a few parts from the image
        split=(torch.rand_like(amask) > 0.5);
        maskedoutA=camask * split;
        maskedoutB=amask -maskedoutA
        # leave EOS alone.
        Apartvisible=(amask-maskedoutA);
        Bpartvisible =(amask-maskedoutB);
        # ign
        neg=torch.zeros_like(target) + mlabel
        ntarA = (neg* maskedoutA +
                 target * Apartvisible).long();
        ntarB = (neg* maskedoutB +
                 target * Bpartvisible).long();

        DAA = A_ * (Apartvisible).float().unsqueeze(-1).unsqueeze(-1);
        DAB= A_ * (Bpartvisible).float().unsqueeze(-1).unsqueeze(-1);
        DTA=A_*(amask).float().unsqueeze(-1).unsqueeze(-1);

        # instead mapping attention backward, we transform the image.
        # hard masking to improve location accuracy.
        return DAA,DAB,DTA, ntarA,ntarB;

        pass;

    def split(this,DAA,DAB,clips,DTA):
        if (DAA.shape[-1] != clips.shape[-1]):
            DAA = trnf.interpolate(DAA, [clips.shape[-2], clips.shape[-1]]);
            # DAB = trnf.interpolate(DAB, [clips.shape[-2], clips.shape[-1]]);
        Avim = (DAA.max(1, keepdim=True)[0]) * clips;
        Bvim = (1 - DAA.max(1, keepdim=True)[0]) * clips;
        return Avim,Bvim;

    def va_nt(this, module_dict, GA, TA, clips, target, acr, p_len, tdict, ign_msk=False,dbgprfx=None):
        DAA,DAB,DTA, ntarA, ntarB = this.relabel(TA, target, acr, p_len, tdict["[UNK]"])
        Avim,Bvim=this.split(DAA,DAB,clips,DTA);
        vim=torch.cat([Avim,Bvim]);

        # set the blocked characters to ignore label....
        # nim = (1-DA.max(1, keepdim=True)[0]) * clips;

        if(dbgprfx is not None):
            for i in range(DAA.shape[0]):
                ti = dump_att_im(clips[i], GA[i], TA[i], p_len[i], None);
                mi = debugva(Avim[i]);
                mi2 = debugva(Bvim[i]);

                cv2.namedWindow("im"+dbgprfx, 0);
                cv2.namedWindow("a"+dbgprfx, 0);
                cv2.namedWindow("b"+dbgprfx, 0);

                cv2.imshow("im"+dbgprfx, ti);
                cv2.imshow("a"+dbgprfx, mi);
                cv2.imshow("b"+dbgprfx, mi2);
            print("A",[tdict[i.item()] for i in ntarA[-1]]);
            print("B",[tdict[i.item()] for i in ntarB[-1]]);

            cv2.waitKey(30);
        # as you can see, these these two process are pretty likely to bite the poor BNs
        module_dict["feature_extractor"].model.freezebn()
        f = module_dict["feature_extractor"](vim);
        module_dict["feature_extractor"].model.unfreezebn()

        # this.debug(AA,mappeda,f,clips[:this.vabs]);
        # as masked image may drastically change the behaviour of lensnet(tpt), we disable it for validation.
        return vim,f, torch.cat([ntarA, ntarB]),torch.cat([p_len,p_len]);
    def seq(this,module_dict,vfs,A,length,vtar,vtar_len,proto,plabel):
        vout_emb = module_dict["seq"](vfs[-1], torch.cat([A, A]), length);
        vlabel_flatten, vlength, vidx = vflatten_label_idx(vtar, vtar_len)
        vfout_emb, _ = this.inflater.inflate(vout_emb, vlength)
        vlogits = module_dict["preds"][0](vfout_emb, proto, plabel);
        return vlogits,vlabel_flatten,vlength;

    def accsort(this, logits, label_flatten, idx):
        with torch.no_grad():
            acr = (logits.max(dim=1)[1] == label_flatten);
            qua = scatter_mean(acr.float(), idx.cuda());
            ord = torch.argsort(-qua)[:this.vcnt];
        return acr, qua, ord

    def engage(this, module_dict, proto, plabel,
               label_flatten, target, idx, length, culength,
               clips, GA, TA, A, logits, tdict, dbgprfx=None):

        acr, qua, ord = this.accsort(logits, label_flatten, idx);
        targetmod = target[ord]

        if ("[UNK]" in tdict):
            acrmod = logits.max(dim=1)[1] == label_flatten;
            tdict[-1] = "🐈"
        else:
            acrmod = acr;
        if GA is None:
            ga=None;
        else:
            ga= GA[ord];
        if (qua[ord[-1]] > 0.5 or dbgprfx is not None):
            acrm, _ = padding_tensor(torch.split(acrmod, length.tolist()))
            vim, vfs, vtar, vtar_len = this.va_nt(module_dict,ga, TA[ord], clips[ord], targetmod,
                                                  acrm[ord], culength[ord], tdict, dbgprfx=dbgprfx);
            vlogits, vlabel_flatten, vlength = this.seq(module_dict, vfs, A[ord].detach(), length[ord], vtar, vtar_len,
                                                        proto, plabel);
            loss_, terms_ = this.loss(module_dict, proto, vlogits, vlabel_flatten.cuda());
            if (GA is not None and dbgprfx is not None):
                detail_va8(dbgprfx, GA[ord], TA[ord], tdict, vim, vlogits, vlabel_flatten, vlength);
            return loss_;
        else:
            return torch.tensor(0.);
