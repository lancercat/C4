


def refactor_meta(meta,unk=0):
    meta["master"]={};
    meta["servants"] = {};
    meta["foes"]={};
    meta["relationships"]={};
    label_dict = {}
    label_dict["[UNK]"] = unk;
    # if the sp_tokens does not provide an specific unk token, set it to -1;
    character=meta["sp_tokens"]+meta["chars"];

    meta["achars"]=character;
    for i, char in enumerate(character):
        # print(i, char)
        label_dict[char] = i;
    meta["label_dict"]=label_dict;
    for ch in character:
        chid=meta["label_dict"][ch];
        meta["master"][chid]=chid;
        meta["servants"][chid] = set();

    for ch in meta["chars"]:
        chid=meta["label_dict"][ch];
        meta["foes"][chid]=set();
    return meta;


def finalize(meta):
    for ch in meta["chars"]:
        chid=meta["label_dict"][ch];
        mid=meta["master"][chid];
        meta["relationships"][chid]=meta["servants"][mid].union({mid}).union(meta["foes"][mid]);
    return meta;

def add_masters(meta,servants,masters):
    for i in range(len(servants)):
        if(masters[i] not in meta["chars"]):
            print("we have rough character", servants[i],"->-",masters[i])
            continue;
        try:
            sid = meta["label_dict"][servants[i]];
            mid = meta["label_dict"][masters[i]];
            meta["master"][sid]=mid;
            meta["servants"][mid].add(sid);
        except:
            pass;

    return meta;
