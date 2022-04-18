from neko_2021_mjt.configs.loadouts.mk7.base_module_set_mk7 import \
    config_bogo_resbinorm,arm_module_set_r45pttpttrinorm_orig_dsa3hGTAnp_mk7_va9,arm_module_set_r45trinorm_orig_dsa3hGTAnp_mk7,arm_module_set_r45trinorm_orig_dsa3hGTAnp_mk7_va9

from neko_2021_mjt.configs.common_subs.arm_post_fe_shared_prototyper import arm_shared_prototyper_np
def arm_lsctsp_set(srcdst,prefix,capacity,feat_ch):
    srcdst[prefix + "LSCT_FE"] = config_bogo_resbinorm(prefix + "feature_extractor_container", "res3");
    srcdst[prefix + "LSCT_PFE"] = config_bogo_resbinorm(prefix + "feature_extractor_container", "res4");
    srcdst = arm_shared_prototyper_np(
        srcdst, prefix, capacity, feat_ch,
        prefix + "LSCT_PFE",
        prefix + "GA",
        nameoverride=prefix + "lsct_prototyper",
        use_sp=False
    );
    return srcdst;

def arm_lsct_mk7_va9_routine(srcdst, prefix, base_routine_type,maxT,lsct_routine_type, log_path, log_each,dsprefix,lsct_proto_name="prototyper",valoss_name="loss_cls_emb"):
    srcdst[prefix + "mjst"] = base_routine_type(
        prototyper_name=prefix + "prototyper",
        sampler_name=prefix + "Latin_62_sampler",
        feature_extractor_name=prefix + "feature_extractor_cco",
        feature_extractor_va_name=prefix + "feature_extractor_va",
        CAMname=prefix + "TA",
        seq_name=prefix + "DTD",
        pred_name=[prefix + "pred"],
        loss_name=[prefix + "loss_cls_emb"],
        va_loss_name=[prefix+valoss_name],
        image_name=dsprefix + "image",
        label_name=dsprefix + "label",
        log_path=log_path,
        log_each=log_each,
        name=prefix + "mjst",
        maxT=maxT,
    );
    srcdst[prefix + "lscsr"] = lsct_routine_type(
        prototyper_name=prefix + lsct_proto_name,
        feature_extractor_name=prefix + "LSCT_FE",
        CAMname=prefix + "GA",
        seq_name=prefix + "DTD",
        pred_name=[prefix + "pred"],
        loss_name=[prefix + "loss_cls_emb"],
        image_name="lscs_samples",
        label_name="lscs_labels",
        proto_name="lscs_protos",
        mask_name="lscs_smsk",
        log_path=log_path,
        log_each=log_each,
        name=prefix + "lscsr",
    )
    return srcdst;

def arm_lsctsp_module_set_r45pt_trinorm_orig_dsa3hGTAnp_va9_mk7(srcdst,prefix,maxT,capacity,feat_ch,tr_meta_path,wemb,expf=1):
    srcdst=arm_module_set_r45pttpttrinorm_orig_dsa3hGTAnp_mk7_va9(srcdst,prefix,maxT,capacity,feat_ch,tr_meta_path,expf=expf,wemb=wemb,fecnt=5);
    srcdst[prefix+"LSCT_FE"]= config_bogo_resbinorm(prefix + "feature_extractor_container", "res4");
    srcdst[prefix+"LSCT_PFE"]= config_bogo_resbinorm(prefix + "feature_extractor_container", "res5");
    srcdst = arm_shared_prototyper_np(
        srcdst, prefix, capacity, feat_ch,
        prefix + "LSCT_PFE",
        prefix + "GA",
        nameoverride=prefix+"lsct_prototyper",
        use_sp=False
    );
    # srcdst[prefix + "LSCT_cam"] = config_sa_mk3(feat_ch=32);
    return srcdst;


def arm_lsctsp_module_set_r45trinorm_orig_dsa3hGTAnp_mk7(srcdst,prefix,maxT,capacity,feat_ch,tr_meta_path,wemb,expf=1):
    srcdst=arm_module_set_r45trinorm_orig_dsa3hGTAnp_mk7(srcdst,prefix,maxT,capacity,feat_ch,tr_meta_path,expf,wemb=wemb,fecnt=4);
    srcdst=arm_lsctsp_set(srcdst,prefix,capacity,feat_ch);
    # srcdst[prefix + "LSCT_cam"] = config_sa_mk3(feat_ch=32);
    return srcdst;

def arm_lsctsp_module_set_r45_trinorm_orig_dsa3hGTAnp_va9_mk7(srcdst,prefix,maxT,capacity,feat_ch,tr_meta_path,wemb,expf=1):
    srcdst=arm_module_set_r45trinorm_orig_dsa3hGTAnp_mk7_va9(srcdst,prefix,maxT,capacity,feat_ch,tr_meta_path,expf=expf,wemb=wemb,fecnt=5);
    srcdst[prefix+"LSCT_FE"]= config_bogo_resbinorm(prefix + "feature_extractor_container", "res4");
    srcdst[prefix+"LSCT_PFE"]= config_bogo_resbinorm(prefix + "feature_extractor_container", "res5");
    srcdst = arm_shared_prototyper_np(
        srcdst, prefix, capacity, feat_ch,
        prefix + "LSCT_PFE",
        prefix + "GA",
        nameoverride=prefix+"lsct_prototyper",
        use_sp=False
    );
    # srcdst[prefix + "LSCT_cam"] = config_sa_mk3(feat_ch=32);
    return srcdst;

def arm_lsct_mk7_routine(srcdst, prefix, base_routine_type, maxT, lsct_routine_type, log_path, log_each, dsprefix,
                         lsct_proto_name="prototyper"):
    srcdst[prefix + "word"] = base_routine_type(
        prototyper_name=prefix + "prototyper",
        sampler_name=prefix + "Latin_62_sampler",
        feature_extractor_name=prefix + "feature_extractor_cco",
        CAMname=prefix + "TA",
        seq_name=prefix + "DTD",
        pred_name=[prefix + "pred"],
        loss_name=[prefix + "loss_cls_emb"],
        image_name=dsprefix + "image",
        label_name=dsprefix + "label",
        log_path=log_path,
        log_each=log_each,
        name=prefix + "word",
        maxT=maxT,
    );
    srcdst[prefix + "lscsr"] = lsct_routine_type(
        prototyper_name=prefix + lsct_proto_name,
        feature_extractor_name=prefix + "LSCT_FE",
        CAMname=prefix + "GA",
        seq_name=prefix + "DTD",
        pred_name=[prefix + "pred"],
        loss_name=[prefix + "loss_cls_emb"],
        image_name="lscs_samples",
        label_name="lscs_labels",
        proto_name="lscs_protos",
        mask_name="lscs_smsk",
        log_path=log_path,
        log_each=log_each,
        name=prefix + "lscsr",
    )
    return srcdst;
