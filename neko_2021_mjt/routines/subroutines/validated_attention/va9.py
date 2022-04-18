import torch;
from torch.nn import functional as trnf;
from neko_2021_mjt.routines.subroutines.validated_attention.va8f import neko_validated_attention_mk8_f,debugva,dump_att_im
import cv2
import numpy as np
from neko_2021_mjt.routines.ocr_routines.mk5.osdan_routine_mk5_ga_va6 import padding_tensor
from neko_2021_mjt.routines.subroutines.fe_seq.GTA import dump_att_im

class neko_validated_attention_mk9(neko_validated_attention_mk8_f):
    def va_nt(this, module_dict, GA, TA, clips, target, acr, p_len, tdict, ign_msk=False,dbgprfx=None):
        DAA,DAB,DTA, ntarA, ntarB = this.relabel(TA, target, acr, p_len, tdict["[UNK]"])
        Avim,Bvim=this.split(DAA,DAB,clips,DTA);
        vim=torch.cat([Avim,Bvim]);

        # set the blocked characters to ignore label....
        # nim = (1-DA.max(1, keepdim=True)[0]) * clips;

        if(dbgprfx is not None):
            for i in range(DAA.shape[0]):
                if(GA is None):
                    ti = dump_att_im(clips[i], None, TA[i], p_len[i], None);
                else:
                    ti = dump_att_im(clips[i], GA[i], TA[i], p_len[i], None);
                mi = debugva(Avim[i]);
                mi2 = debugva(Bvim[i]);

                cv2.namedWindow("im"+dbgprfx, 0);
                cv2.namedWindow("a"+dbgprfx, 0);
                cv2.namedWindow("b"+dbgprfx, 0);

                cv2.imshow("im"+dbgprfx, ti);
                cv2.imshow("a"+dbgprfx, mi);
                cv2.imshow("b"+dbgprfx, mi2);
                cv2.waitKey(0);

                print("A",[tdict[c.item()] for c in ntarA[i]]);
                print("B",[tdict[c.item()] for c in ntarB[i]]);

        # as you can see, these these two process are pretty likely to bite the poor BNs
        # well, freezing BN is no good either, so...
        f = module_dict["feature_extractor_va"](vim);
        # this.debug(AA,mappeda,f,clips[:this.vabs]);
        # as masked image may drastically change the behaviour of lensnet(tpt), we disable it for validation.
        return vim,f, torch.cat([ntarA, ntarB]),torch.cat([p_len,p_len]);

    def engage(this, module_dict, proto, plabel,
               label_flatten, target, idx, length, culength,
               clips, fs, GA, TA, A, logits, tdict, dbgprfx=None):

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
            return loss_,torch.tensor(0.);
        else:
            return torch.tensor(0.),torch.tensor(0.);
class neko_validated_attention_mk9r(neko_validated_attention_mk9):
    def split(this,DAA,DAB,clips,DTA):
        if (DAA.shape[-1] != clips.shape[-1]):
            # DAA = trnf.interpolate(DAA, [clips.shape[-2], clips.shape[-1]]);
            DAB = trnf.interpolate(DAB, [clips.shape[-2], clips.shape[-1]]);
        Avim = (1 - DAB.max(1, keepdim=True)[0]) * clips;
        Bvim = (DAB.max(1, keepdim=True)[0]) * clips;
        return Avim,Bvim;
