from neko_sdk.encoders.chunked_resnet.neko_block_fe import \
    init_layer, \
    dan_reslayer


class res45_net:
    def cuda(this):
        pass;
    def set_up_layers(this,layer_dict,bn_dict):
        this.init_layer=init_layer(layer_dict["0"],bn_dict["0"]);
        this.res_layer1 = dan_reslayer(layer_dict["1"], bn_dict["1"]);
        this.res_layer2 = dan_reslayer(layer_dict["2"], bn_dict["2"]);
        this.res_layer3 = dan_reslayer(layer_dict["3"], bn_dict["3"]);
        this.res_layer4 = dan_reslayer(layer_dict["4"], bn_dict["4"]);
        this.res_layer5 = dan_reslayer(layer_dict["5"], bn_dict["5"]);
    def __init__(this,layer_dict,bn_dict):
        this.set_up_layers(layer_dict,bn_dict);

    def __call__(this, x):
        ret=[];
        x=this.init_layer(x.contiguous());
        x = this.res_layer1(x);
        x = this.res_layer2(x);
        ret.append(x);
        x = this.res_layer3(x);
        x = this.res_layer4(x);
        ret.append(x);
        x = this.res_layer5(x);
        ret.append(x);
        return ret;

class res45_net_orig(res45_net):
    def __call__(this, x):
        ret=[];
        x=this.init_layer(x.contiguous());
        tmp_shape = x.size()[2:]
        x = this.res_layer1(x);
        if x.size()[2:] != tmp_shape:
            tmp_shape = x.size()[2:]
            ret.append(x)
        x = this.res_layer2(x);
        if x.size()[2:] != tmp_shape:
            tmp_shape = x.size()[2:];
            ret.append(x);
        x = this.res_layer3(x);
        if x.size()[2:] != tmp_shape:
            tmp_shape = x.size()[2:];
            ret.append(x);
        x = this.res_layer4(x);
        if x.size()[2:] != tmp_shape:
            tmp_shape = x.size()[2:]
            ret.append(x);
        x = this.res_layer5(x);
        ret.append(x);
        return ret;


class res45_net_tpt(res45_net_orig):
    def cuda(this):
        pass;
    def set_up_layers(this,layer_dict,bn_dict):
        this.init_layer=init_layer(layer_dict["0"],bn_dict["0"]);
        this.lens=layer_dict["lens"];
        this.res_layer1 = dan_reslayer(layer_dict["1"], bn_dict["1"]);
        this.res_layer2 = dan_reslayer(layer_dict["2"], bn_dict["2"]);
        this.res_layer3 = dan_reslayer(layer_dict["3"], bn_dict["3"]);
        this.res_layer4 = dan_reslayer(layer_dict["4"], bn_dict["4"]);
        this.res_layer5 = dan_reslayer(layer_dict["5"], bn_dict["5"]);

    def __call__(this, x,debug=False):
        ret=[];
        x=this.init_layer(x.contiguous());
        x,grid=this.lens(x);
        tmp_shape = x.size()[2:]
        x = this.res_layer1(x);
        if x.size()[2:] != tmp_shape:
            tmp_shape = x.size()[2:]
            ret.append(x)
        x = this.res_layer2(x);
        if x.size()[2:] != tmp_shape:
            tmp_shape = x.size()[2:];
            ret.append(x);
        x = this.res_layer3(x);
        if x.size()[2:] != tmp_shape:
            tmp_shape = x.size()[2:];
            ret.append(x);
        x = this.res_layer4(x);
        if x.size()[2:] != tmp_shape:
            tmp_shape = x.size()[2:]
            ret.append(x);
        x = this.res_layer5(x);
        ret.append(x);
        if(debug):
            return x,[grid];
        return ret;
