from neko_2021_mjt.routines.fsl_routines.osfsl_routine import neko_OSFSL_routine

def osfsl_ocr_routine(prototyper_name,feature_extractor_name,seq_name,
                         CAMname,pred_name,loss_name,image_name,label_name,proto_name,mask_name,log_path,log_each,name,rot=False):
    return \
    {

        "name":name,
        "routine":neko_OSFSL_routine,
        "rot":rot,
        "freeze_bn":False,
        "mod_cvt_dicts":
        {
            "prototyper":prototyper_name,
            "feature_extractor":feature_extractor_name,
            "CAM":CAMname,
            "seq": seq_name,
            "preds":pred_name,
            "losses":loss_name,
        },
        "inp_cvt_dicts":
        {
            "labels":label_name,
            "samples":image_name,
            "protos": proto_name,
        },
        "log_path":log_path,
        "log_each":log_each,
    };
