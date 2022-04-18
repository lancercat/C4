from builtins import list

from neko_sdk.thirdparty.trdg.data_generator import FakeTextDataGenerator
import random;
import lmdb;
import pickle;

class neko_abstract_string_generator:
    def mount_meta(this,meta):
        pass;
    def set_genpara(this):
        # generator
        this.gen = FakeTextDataGenerator();
        # hyper parameter
        this.size = 64;
        this.skewing_angle = [0, 0, 0, 2, 2, 5, 5, 10];
        this.random_skew = [False, False, False, True];
        this.blur = [0, 1];
        this.random_blur = [True, True, False];
        this.background_types = [0, 1, 2, 3, 3, 3, 3];
        this.distorsion_type = [False, False, 0, 1, 2];
        this.distorsion_orientation = [0, 1, 2];
        this.is_handwritten = False;
        this.width = -1;
        this.alignment = 0;
        this.text_color = "#010101";
        this.orientation = 0;
        this.space_width = 1;
        this.character_spacing = [0, 64, 32];
        this.margins = (5, 5, 5, 5);
        this.fit = 0;
        this.output_mask = 0;
        this.word_split = [True, False];

    def __init__(this, meta,bgims):
        # meta info
        this.mount_meta(meta);
        this.bgims = bgims;
        # background image list
        this.set_genpara();
        pass;
    def get_content(this):
        return None,None;

    # Drive the vehicle
    def drive(this,bgtype,bgim,font,content):
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
        space_width = this.space_width;
        character_spacing = random.choice(this.character_spacing);
        margins = this.margins;
        fit = this.fit;
        if(len(content)<=5):
            word_split = random.choice(this.word_split);
        else:
            word_split = True;
        final_image,final_mask=this.gen.generate_core(
            content,
            font,
            size,
            skewing_angle,
            random_skew,
            blur,
            random_blur,
            background_type,
            distorsion_type,
            distorsion_orientation,
            is_handwritten,
            width,
            alignment,
            text_color,
            orientation,
            space_width,
            character_spacing,
            margins,
            fit,
            word_split,
            bgim,
        );
        return final_image,content,final_mask;
    def random_bgm(this):
        btype=random.choice(this.background_types);
        bgim=None;
        if(btype==3):
            try:
                bgim = random.choice(this.bgims);
            except:
                return this.random_bgm();
            if(bgim is None):
                return this.random_bgm();
        return btype,bgim;
    def random_clip(this):
        fnt,content=this.get_content();
        bgtype,bgm=this.random_bgm();
        final_image,content,final_mask=this.drive(bgtype,bgm,fnt,content);
        return final_image,content;


class neko_random_string_generator(neko_abstract_string_generator):
    def __init__(this, meta,bgims,max_len):
        super(neko_random_string_generator, this).__init__(meta,bgims,max_len);
        this.max_len=max_len;
        pass;

    def mount_meta(this,meta):
        this.fnt_charset=meta["fnt_charset"];
        this.fnt_grp=meta["fnt_grp"];
        this.fnt_grp_keys=list(this.fnt_grp.keys());
        this.segments=meta["grpseg"];
        this.segment_length=meta["segment_length"];
        this.spaces=meta["spaces"];

    def random_str(this, charset, l):
        return ''.join(random.choice(charset) for _ in range(l));

    def compose_content(this, segment_lens, charset, space):
        str = "".join(this.random_str(charset, l) + space for l in segment_lens);
        return str.strip(space)[:this.max_len];

    def get_content(this):
        fntg = random.choice(this.fnt_grp_keys);
        fnt = random.choice(this.fnt_grp[fntg]);
        charset = this.fnt_charset[fnt];
        segment_cnt = random.choice(this.segments[fntg]);
        segments_length = [random.choice(this.segment_length[fntg]) for _ in range(segment_cnt)];
        content = this.compose_content(segments_length, charset, space=this.spaces[fntg]);
        if (len(content) == 0):
            return this.get_content();
        return fnt, content;


class neko_skip_missing_string_generator(neko_abstract_string_generator):
    def __init__(this, meta, bgims, max_len):
        super(neko_skip_missing_string_generator, this).__init__(meta, bgims, max_len);
        this.max_len = max_len;
        pass;

    def mount_meta(this,meta):
        this.fnt_charset=meta["fnt_charset"];
        this.fnt_grp=meta["fnt_grp"];
        this.fnt_grp_keys=list(this.fnt_grp.keys());
        corpusdb=meta["corpus_db"];
        this.env = lmdb.open(corpusdb,  readonly=True, lock=False, readahead=False, meminit=False)
        this.txn=this.env.begin(write=False);
        this.nSamples = int(this.txn.get('num-samples'.encode()));
        this.cntr=random.randint(0,this.nSamples);
        # this.segments=meta["grpseg"];
        # this.segment_length=meta["segment_length"];
        # this.spaces=meta["spaces"];

    def compose_content(this,length, charset):
        compatKey = 'content-%09d'.encode() % this.cntr;

        rawstr=this.txn.get(compatKey).decode();
        ret = "".join( c if c in charset else "" for c in rawstr);
        return ret[:length];

    def get_content(this):
        this.cntr+=1;
        this.cntr%=this.nSamples;
        fntg = random.choice(this.fnt_grp_keys);
        fnt = random.choice(this.fnt_grp[fntg]);
        charset = this.fnt_charset[fnt];
        # segment_cnt = random.choice(this.segments[fntg]);
        # segments_length = [random.choice(this.segment_length[fntg]) for _ in range(segment_cnt)];
        # segments_length =random.choice(this.segment_length[fntg]);
        content = this.compose_content(this.max_len, charset);
        if (len(content) == 0):
            return this.get_content();
        return fnt, content;


class neko_random_corpus_generator(neko_abstract_string_generator):
    def mount_meta(this,meta,corpusdb):
        this.fonts=meta["fonts"];
        this.env = lmdb.open(corpusdb,  readonly=True, lock=False, readahead=False, meminit=False)
        this.txn=this.env.begin(write=False);
        this.nSamples = int(this.txn.get('num-samples'.encode()));
        this.cntr=0;

    def __init__(this, meta,corpusdb,bgims):
        # meta info
        this.mount_meta(meta,corpusdb);
        this.bgims = bgims;
        # background image list
        this.set_genpara();
        pass;
    def get_content_idx(this,idx):
        contentKey = 'content-%09d'.encode() % idx;
        compatKey = 'compatible-%09d'.encode() % idx;
        content=this.txn.get(contentKey).decode();
        compat_list = pickle.loads(this.txn.get(compatKey));
        fnt=this.fonts[random.choice(compat_list)];
        return fnt,content
    def _get_content(this):
        this.cntr += 1;
        this.cntr %= this.nSamples;
        fnt, content = this.get_content_idx(this.cntr);
        # try:
        #
        # except:
        #     return this.get_content();
        return fnt,content[:25];
    def get_content(this):
        try:
            return this._get_content();
        except:
           return this._get_content();

