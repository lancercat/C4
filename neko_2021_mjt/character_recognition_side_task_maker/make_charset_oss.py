import cv2
import torch

from neko_sdk.ocr_modules.fontkit.fntmgmt import fntmgmt;
from neko_sdk.renderlite.lib_render import render_lite
from random import shuffle
import glob
import os
RDST="/run/media/lasercat/thirdeye/cache/patch/"


def grab_char_from_fnt(file):
    return fntmgmt.get_charset_gen2(file)
def mine_char_form_corpus(file):
    pass;

def grab_character(fonts):
    fl = []
    dd = {};
    chff = [];
    d = {};

    for i in fonts:
        s, c = grab_char_from_fnt(i);
        chff += s;
        chff += c;
        chff = list(set(chff));

        for j in s.union(c):
            if (j not in d):
                d[j] = [];
            d[j].append(i);

        print(len(chff))


    for k in d:
        if (len(d[k]) <= 1):
            print(len(k), "friendless", k, d[k]);
            fl.append([k, d[k]]);
            dd[k]=d[k];
            # del d[k]
        else:
            dd[k] = d[k]
    torch.save(dd, "compat.pt")
    print(len(dd))
    pass;


def render_clips(i):
    d=torch.load("compat.pt");
    a=list(d.keys());
    r=render_lite(fos=64);
    try:
        # if the character has been fully rendered, go bother something else.
        with open(os.path.join(RDST, str(i)+".txt"),"r") as _:
            pass;
        return;
    except:
        pass;

    os.makedirs(os.path.join(RDST, str(i)), exist_ok=True);
    for j in range(len(d[a[i]])):
        try:
            name=os.path.join(RDST,str(i),str(j)+".png");
            im=r.center_draw(128,a[i],d[a[i]][j]);
            if( im.max() < 13):
                pass;
                # print("offending_fnt:",d[a[i]][j],"character",a[i]);
            else:
                cv2.imwrite(name,im)
        except:
            pass;
            # print("offending_fnt:",d[a[i]][j],"character",a[i]);
    # print("here")
    with open(os.path.join(RDST, str(i) + ".txt"), "w+") as _:
        pass;
        # print("done");


from neko_sdk.lmdb_wrappers.im_lmdb_wrapper import im_lmdb_wrapper;
from tqdm import tqdm
import multiprocessing

if __name__ == '__main__':
    FROOT="/run/media/lasercat/portable/ssddata/synth_lsct/fonts-main/";
    DST="/home/lasercat/ssddata/charset_lmdb_oss/";

    with open(os.path.join(FROOT,"all.txt"),"r") as fp:
        fonts=[os.path.join(FROOT,i.strip()) for i in fp]
    # fonts=fonts[:20];
    grab_character(fonts);
    d = torch.load("compat.pt");
    a=list(d.keys());
    i_s = list(range(len(d)));
    shuffle(i_s);
    p = multiprocessing.Pool(12)
    p.map(render_clips,i_s);
    meta = {};
    wrapper = im_lmdb_wrapper(DST);
    for i in tqdm(range(len(a))):
        f = os.path.join(RDST, str(i), "*.png");
        f = glob.glob(f);
        if (len(f) < 2):
            print(a[i], "too few samples", i);
            continue;
        meta[a[i]] = [];
        for imp in f:
            ina = 'image-%09d'.encode() % wrapper.load
            meta[a[i]].append(ina);
            im = cv2.imread(imp)
            wrapper.add_data_utf(im, a[i], "");
    torch.save(meta, os.path.join(DST,"meta.pt"));
    print("done");

