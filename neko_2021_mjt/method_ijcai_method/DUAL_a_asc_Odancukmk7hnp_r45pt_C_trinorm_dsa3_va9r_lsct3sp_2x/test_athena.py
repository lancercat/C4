# coding:utf-8

from __future__ import print_function

from dicted_eval_configs import dan_mjst_dict_eval_cfg
from neko_2021_mjt.lanuch_std_test import testready
import os,glob,cv2,pylcs,torch
from neko_2020nocr.result_renderer import render_word
from neko_sdk.root import find_data_root
import regex
from neko_2020nocr.tasks.dscs import makept
from neko_2021_mjt.open_test import img_test;
from neko_sdk.root import find_export_root,find_model_root


def prepare_pt(root,lang):
    with open(os.path.join(root,lang,"meta","alphabets.txt"))as fp:
        chars=[l.strip() for l in fp];
        allch=[];
        masters=[];
        servants=[];
        for ch in chars:
            if(len(ch)):
                l= regex.findall(r'\X', ch, regex.U);
                allch+=l;
                for i in range(1,len(l)):
                    masters.append(l[0]);
                    servants.append(l[i]);
    allch=list(set(allch));
    fntp=os.path.join(root,lang,"meta","notofont.ttf");
    dictp=os.path.join(root,lang,"meta","dict.pt");
    makept(None, [fntp],
           dictp,
           allch, {}, masters=masters, servants=servants);
    return dictp;

def run_athena_folder(root,dst,lang,argv):
    ptfile=prepare_pt(root,lang);
    runner,globalcache,mdict=testready(argv,dan_mjst_dict_eval_cfg,ptfile);
    sfolder=os.path.join(root,lang);
    dfolder=os.path.join(dst,os.path.basename(lang),"results");
    os.makedirs(dfolder,exist_ok=True);
    files = glob.glob(os.path.join(sfolder, "*.jpg"));
    for i in range(len(files)):
        res = img_test(files[i],
                       runner,
                       globalcache);
        base = os.path.basename(files[i]);
        dstt = os.path.join(dfolder, base.replace("jpg", "txt"));
        dsti = os.path.join(dfolder, base);
        dim, _ = render_word(mdict, {}, cv2.imread(files[i]), None, res, 0);
        cv2.imwrite(dsti, dim);
        print(res);
        with open(dstt, "w+") as fp:
            fp.writelines(res);


def run_athena(root,dst,argv):
    os.makedirs(dst,exist_ok=True)
    langs=glob.glob(os.path.join(root,"lang_*"));
    for lang in langs:
        run_athena_folder(root,dst,lang,argv);
    # args = runner.testready();
    # mdict = torch.load(ptfile)





if __name__ == '__main__':
    import sys

    if (len(sys.argv) < 2):
        argv = ["",
                find_export_root(),
                "_E0",
                find_model_root()+"DUAL_a_asc_Odancukmk7hnp_r45pt_C_trinorm_dsa3_va9r_lsct3sp_2x",
                ]
    else:
        argv = sys.argv;

    run_athena(find_data_root()+"/Athena/","/run/media/lasercat/data/c491/c4-out/",argv)
