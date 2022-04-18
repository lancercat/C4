import glob
import os.path
import random
from neko_sdk.ocr_modules.trdg_driver.corpus_data_generator_driver import neko_abstract_string_generator

class neko_masked_sample_generator(neko_abstract_string_generator):
    def __init__(this,bgimp,seed=9):
        # background images can be handled by caller.

        if(bgimp is not None):
            this.bgims = glob.glob(os.path.join(bgimp,"*.jpg"));
        else:
            this.bgims=["To_be_replaced_by_caller"];
        # background image list
        this.set_genpara();
        random.seed(seed)
        pass;

    def drive_mask(this,bgtype,bgim,mask):
        size = this.size;
        skewing_angle = random.choice(this.skewing_angle);
        random_skew = random.choice(this.random_skew);
        blur = random.choice(this.blur);
        random_blur = this.random_blur;
        background_type = bgtype;
        distorsion_type = random.choice(this.distorsion_type);
        distorsion_orientation = random.choice(this.distorsion_orientation);
        is_handwritten = this.is_handwritten;
        width = this.width;
        alignment = this.alignment;
        if(bgim is not None):
            text_color = "#"+''.join([random.choice('0123456789ABCDEF') for j in range(6)]);
        else:
            text_color = "#" + ''.join([random.choice('0123456789') for j in range(6)]);

        orientation = this.orientation;
        margins = this.margins;
        final_image,final_mask=this.gen.generate_from_mask_core(
            mask,
            size,
            skewing_angle,
            random_skew,
            blur,
            random_blur,
            background_type,
            distorsion_type,
            distorsion_orientation,
            width,
            alignment,
            text_color,
            orientation,
            margins,
            True,
            bgim,
        );
        return final_image,final_mask;

    def get_sample(this,mask,bgim=None):
        bgtype, bgm = this.random_bgm();
        if(bgim is not None):
            bgm=bgim;
        final_image, final_mask = this.drive_mask(bgtype, bgm, mask);
        return final_image;

    def get_samplewm(this, mask, bgim=None):
        bgtype, bgm = this.random_bgm();
        if (bgim is not None):
            bgm = bgim;
        final_image, final_mask = this.drive_mask(bgtype, bgm, mask);
        return final_image,final_mask;
class neko_masked_sample_generator_lite(neko_masked_sample_generator):
    def set_genpara(this):
        # generator
        super(neko_masked_sample_generator_lite,this).set_genpara();
        this.background_types = [0, 1, 3, 3, 3, 3];
