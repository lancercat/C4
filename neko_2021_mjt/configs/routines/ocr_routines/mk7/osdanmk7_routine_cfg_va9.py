from neko_2021_mjt.routines.ocr_routines.mk7.osdan_routine_mk7_va9 import neko_HDOS2C_routine_CFmk7_va9r

def osdanmk7_va9_ocr_routine_like(RTYPE,sampler_name,prototyper_name,feature_extractor_name,feature_extractor_va_name,seq_name,
                         CAMname,pred_name,loss_name,va_loss_name,label_name,image_name,log_path,log_each,name,maxT):
    return {

        "maxT": maxT,
        "name": name,
        "routine": RTYPE,
        "mod_cvt_dicts":
            {
                "sampler": sampler_name,
                "prototyper": prototyper_name,
                "feature_extractor": feature_extractor_name,
                "feature_extractor_va": feature_extractor_va_name,
                "TA": CAMname,
                "seq": seq_name,
                "preds": pred_name,
                "losses": loss_name,
                "valoss":va_loss_name,
            },
        "inp_cvt_dicts":
            {
                "label": label_name,
                "image": image_name,
            },
        "log_path": log_path,
        "log_each": log_each,
    }
def osdanmk7_va9r_ocr_routine(sampler_name,prototyper_name,feature_extractor_name,feature_extractor_va_name,seq_name,
                         CAMname,pred_name,loss_name,va_loss_name,label_name,image_name,log_path,log_each,name,maxT):
    return osdanmk7_va9_ocr_routine_like(neko_HDOS2C_routine_CFmk7_va9r,sampler_name,prototyper_name,feature_extractor_name,feature_extractor_va_name,seq_name,
                         CAMname,pred_name,loss_name,va_loss_name,label_name,image_name,log_path,log_each,name,maxT);

