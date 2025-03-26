
from fontTools.ttLib import TTFont
from fontTools.unicode import Unicode
import unicodedata

from fontTools.otlLib.builder import buildCoverage
import torch;
from itertools import chain
class fntmgmt:
    NORMAL_CHARACTER=["Lo","Lu","Ll","So","Sm","Nd","Nl","No",]
    @classmethod
    def parse_sub_4(cls,sub):
        ret=[];
        for k in sub.ligatures.keys():
            for l in sub.ligatures[k]:
                magic=[k]+l.Component;
                ret.append(magic);
        return ret;
    @classmethod
    def parse_sub_1(cls,sub):
        return sub.mapping.keys();

    @classmethod
    def parse_sub_2(cls, sub):
        return sub.mapping.values();

    @classmethod
    def parse_sub(cls,sub):
        if(sub.LookupType==1):
            return cls.parse_sub_1(sub);
        elif(sub.LookupType==2):
            return cls.parse_sub_2(sub);
        elif(sub.LookupType==3):
            pass;

        elif(sub.LookupType==4):
            return cls.parse_sub_4(sub);
        else:
            # print(sub.LookupType);
            return [];
    @classmethod
    def get_charset(cls, fp):
        ttf = TTFont(fp, 0,fontNumber=0, verbose=0, allowVID=0,
                     ignoreDecompileErrors=True)
        chars = chain.from_iterable([chr(y[0]) for y in x.cmap.items()] for x in ttf["cmap"].tables)
        return set(chars);

    # supports gsub. With respect, this is bloodily  over-complex
    @classmethod
    def get_charset_gen2(cls, fp):



        ttf = TTFont(fp, 0, fontNumber=0, verbose=0, allowVID=0,
                     ignoreDecompileErrors=True)
        dic={};
        rdic={}
        def handy(y):
            dic[y[1]]=chr(y[0]);
            rdic[chr(y[0])]=dic[y[1]];
            return chr(y[0])
        schars_ = list(chain.from_iterable([handy(y) for y in x.cmap.items()] for x in ttf["cmap"].tables))
        schars=[];
        for c in schars_:
            if unicodedata.category(c) in cls.NORMAL_CHARACTER:
                schars.append(c);
        cchars=[];
        flag=True;
        try:
            _=ttf["GSUB"].table.LookupList;
        except:
            flag=False;
        if(flag and "GSUB" in ttf and ttf["GSUB"].table.LookupList is not None):
            for sub in ttf["GSUB"].table.LookupList.Lookup:
                for t in sub.SubTable:
                    mag=cls.parse_sub(t);
                    if(mag is None ):
                        continue;

                    codes=list(mag);
                    for c in codes:
                        try:
                            if(type(c)==list):
                                rc="".join(dic[c_.split(".")[0].split("_")[0]] for c_ in c);
                            else:
                                rc=dic[c];
                            cchars.append(rc);
                        except:
                            # print(c);
                            for c_ in c:
                                if(c_.split(".")[0].split("_")[0] not in dic):
                                    pass;
                                    # print("what is ",c_);
        return set(cchars),set(schars);
    @classmethod
    
    def init_charset(this,fnt_d):
        for k in fnt_d:
            this.charset_d[k]=this.get_charset(fnt_d[k]);
            torch.save(this.charset_d,"charset_d.pt")
            print(k,"scanned");
    def load_charset(this):
        this.charset_d=torch.load("charset_d.pt");
    def __init__(this,fnt_d):
        this.charset_d={};
        if(fnt_d is None):
            this.load_charset();
        else:
            this.init_charset(fnt_d);
        pass;


class make_meta:
    def __init__(this,all_chars,blacklisted_chars):
        this.meta={}
        this.meta["fnt_charset"] = {};
        this.meta["fnt_grp"]={};
        this.meta["grpseg"]={};
        this.meta["segment_length"]={};
        this.meta["spaces"]={};
        this.valid=set(all_chars)-set(blacklisted_chars);
    def addgrp(this,name,space,segment_cnt,segment_length):
        this.meta["fnt_grp"][name]=[];
        this.meta["grpseg"][name] = segment_cnt;
        this.meta["segment_length"][name] = segment_length;
        this.meta["spaces"][name] = space;
    def add_fnt(this,fnt,grp_name):
        fnt_cs=fntmgmt.get_charset(fnt);
        this.meta["fnt_grp"][grp_name].append(fnt);
        this.meta["fnt_charset"][fnt]=list(fnt_cs.intersection(this.valid));


