from neko_2021_mjt.lanuch_std_test import testready
import os,glob,cv2,pylcs,torch
from neko_2020nocr.result_renderer import render_word
from neko_sdk.root import find_model_root,find_data_root
import regex
from neko_2020nocr.tasks.dscs import makept

def img_test(img,runner,args):
    res=runner.test_img(0,img,args);
    return  res;
