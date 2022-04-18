import random

import torch

from neko_2021_mjt.routines.neko_abstract_routines import neko_abstract_routine;
from neko_2020nocr.dan.utils import Loss_counter,neko_os_ACR_counter;

import cv2
import numpy as np
class neko_OSFSL_routine(neko_abstract_routine):

    def set_loggers(this,log_path,log_each,name):
        this.logger_dict={
            "accr": neko_os_ACR_counter("["+name+"]"+"train_accr"),
            "loss": Loss_counter("[" + name + "]" + "train_accr"),
        };

    def set_etc(this, args):
        this.rot=args["rot"];
        # this.freeze_bn=args["freeze_bn"]
        if(this.rot):
            this.possiblerots=[0,1,2,3];

        pass;

    def fe_seq(this, clips, module_dict, length):
        clips = clips.cuda()
        seq = module_dict["seq"];
        features = module_dict["feature_extractor"](clips);

        A = module_dict["CAM"](features)
        # A=torch.ones_like(A);
        out_emb = seq(features[-1], A, length);
        return out_emb, A;
    def show_clip(this,clip,label,proto,rang):
        all=[]
        proto_=list(proto)+[torch.zeros_like(proto[0])]
        for i in rang:
            all.append(torch.cat([clip[i],proto_[label[i]]],1)+1);
        im=(torch.cat(all,-1).detach()*127).permute(1,2,0).cpu().numpy().astype(np.uint8)
        cv2.imshow("debug",im);
        cv2.waitKey(0);
    def fp_impl(this, input_dict, module_dict,logger_dict,nEpoch,batch_idx):
        target=input_dict["labels"].cuda();
        clips=input_dict["samples"].cuda();
        suppports=input_dict["protos"].cuda();
        if(this.rot):
            k=random.choice(this.possiblerots);
            clips=torch.rot90(clips, dims=[2, 3], k=k)
            suppports=torch.rot90(suppports, dims=[2, 3], k=k);

        prototyper=module_dict["prototyper"];
        # do not make injected [s]
        proto=prototyper([suppports],use_sp=False);
        preds=module_dict["preds"];

        # patch up censored label
        # net forward
        length=torch.ones_like(target);
        out_emb,A=this.fe_seq(clips,module_dict,length)
        # net forward
        # Length dring training is known.
        fout_emb=out_emb.reshape(clips.shape[0],-1)
        beams = [];
        terms = [];
        loss = 0;
        for i in range(len(preds)):
            logits = preds[i](fout_emb, proto, None);
            res=logits.max(dim=-1)[1];
            beams.append(res);
            loss_, terms_ = module_dict["losses"][i](proto, logits, target);
            loss = loss_ + loss;
            terms.append(terms_);
        # this.show_clip(clips, target,suppports, range(128+16))
        logger_dict["accr"].add_iter(beams[0],target)
        logger_dict["loss"].add_iter(loss, terms[0])
        return loss;

