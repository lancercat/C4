
# using mk5 DTD
import torch.nn.functional
from neko_2021_mjt.modulars.neko_inflater import neko_inflater
from torch.nn import functional as trnf
from neko_2020nocr.dan.common.common import flatten_label
from neko_2021_mjt.routines.subroutines.validated_attention.va9 import neko_validated_attention_mk9r
from neko_2021_mjt.routines.ocr_routines.mk7.osdan_routine_mk7 import neko_HDOS2C_routine_CFmk7
from neko_2021_mjt.routines.subroutines.fe_seq.TA import temporal_attention_v2_va9
from neko_2021_mjt.routines.subroutines.mk7common import mklabel_mk7

class neko_HDOS2C_routine_CFmk7_va9(neko_HDOS2C_routine_CFmk7):
    def set_etc(this, args):
        this.maxT = args["maxT"];
        this.inflater = neko_inflater();
        #this.vaengine = neko_validated_attention_mk9(inflater=this.inflater, vcnt=18);
        this.attf = temporal_attention_v2_va9;
    def mk_proto(this,label,sampler,prototyper):
        normprotos, plabel, tdict=sampler.model.sample_charset_by_text(label,use_sp=False)
        # im=(torch.cat(normprotos,3)*127+128)[0][0].numpy().astype(np.uint8);
        # cv2.imshow("alphabets",im);
        # print([tdict[label.item()] for label in plabel]);
        # cv2.waitKey(0);
        proto=prototyper(normprotos,use_sp=False);

        semb=None
        return proto, semb, plabel, tdict

    def fp_impl(this, input_dict, module_dict, logger_dict, nEpoch, batch_idx):
        label = input_dict["label"];
        clips = input_dict["image"];
        prototyper = module_dict["prototyper"]
        sampler = module_dict["sampler"];
        preds = module_dict["preds"];
        if("debug_path" in input_dict):
            DBGKEY="meow"
        else:
            DBGKEY=None;
        proto, semb, plabel, tdict = this.mk_proto(label, sampler, prototyper);
        label_flatten, target, length, lengthl, idx = \
            mklabel_mk7(module_dict, proto, plabel, tdict, label);

        clips,target, label_flatten, culength = clips.cuda(),target.cuda(), label_flatten.cuda(), length.cuda().long()
        out_emb,GA,TA,A, pred_length,features = this.attf(clips, module_dict, length)
        # net forward
        # Length dring training is known.
        fout_emb, _ = this.inflater.inflate(out_emb, length)
        lossess = []
        beams = [];
        probs = [];
        terms = [];
        loss = torch.nn.functional.cross_entropy(pred_length, culength);

        logits = preds[0](fout_emb, proto, plabel);
        choutput, prdt_prob = sampler.model.decode(logits, length, proto, plabel, tdict);
        loss_, terms_ = module_dict["losses"][0](proto, logits, label_flatten);
        valoss,_ = this.vaengine.engage(module_dict, proto, plabel, label_flatten,
                                      target, idx, length, culength, clips,features,
                                      GA, TA, A, logits, tdict, dbgprfx=DBGKEY);

        loss = loss_ + loss+valoss*0.3;
        beams.append(choutput);
        terms.append(terms_);

        # computing accuracy and loss
        tarswunk = ["".join([tdict[i.item()] for i in target[j]]).replace('[s]', "") for j in
                    range(len(target))];
        logger_dict["accr"].add_iter(beams[0], length, tarswunk)
        logger_dict["loss"].add_iter(loss, terms[0])
        return loss;


class neko_HDOS2C_routine_CFmk7_va9r(neko_HDOS2C_routine_CFmk7_va9):
    def set_etc(this, args):
        this.maxT = args["maxT"];
        this.inflater = neko_inflater();
        this.vaengine = neko_validated_attention_mk9r(inflater=this.inflater, vcnt=18);
        this.attf = temporal_attention_v2_va9;
