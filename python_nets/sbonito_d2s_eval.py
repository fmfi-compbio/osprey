from ont_fast5_api.fast5_interface import get_fast5_file
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
from datetime import datetime
import os
import h5py
import deepnano2

def med_mad(x, factor=1.4826):
    """
    Calculate signal median and median absolute deviation
    """
    med = np.median(x)
    mad = np.median(np.absolute(x - med)) * factor
    return med, mad

def rescale_signal(signal):
    signal = signal.astype(np.float32)
    med, mad = med_mad(signal)
    signal -= med
    signal /= mad       
    return np.clip(signal, -2.5, 2.5)

from modelsbd2s import Model
from bonito import model as bmodel
import bonito
import toml


# In[5]:


os.listdir()


# In[6]:


cfgx = toml.load(bonito.__path__[0] + "/models/configs/dna_r9.4.1.toml")
cfgx

common = dict(activation="swish", dropout=0.0, dilation=[1])
block = [
    #C1
    dict(**common, repeat = 1, filters = 80, kernel = [9], stride = [3], residual = False, separable = False,),

    dict(**common, repeat = 5, filters = 80, kernel = [11], stride = [1], residual = True, separable = True, type="BlockX", pool=3, inner_size=160),
    dict(**common, repeat = 5, filters = 80, kernel = [11], stride = [1], residual = True, separable = True, type="BlockX", pool=3, inner_size=160),
    dict(**common, repeat = 5, filters = 80, kernel = [11], stride = [1], residual = True, separable = True, type="BlockX", pool=3, inner_size=160),
    dict(**common, repeat = 5, filters = 80, kernel = [11], stride = [1], residual = True, separable = True, type="BlockX", pool=3, inner_size=160),
    dict(**common, repeat = 5, filters = 80, kernel = [11], stride = [1], residual = True, separable = True, type="BlockX", pool=3, inner_size=160),
    #C2
    dict(**common, repeat = 1, filters = 80, kernel = [11], stride = [1], residual = False, separable = True,),
    #C3
    dict(**common, repeat = 1, filters = 40, kernel = [7], stride = [1], residual = False, separable = False,)
]

cfgx["encoder"]["activation"] = "swish"
cfgx["block"] = block
cfgx["input"]["features"] = 1
cfgx



bmodel.activations["relu6"] = nn.modules.activation.ReLU6


# In[15]:


bmodelx = Model(cfgx)


# In[16]:


C = 5
ls_weights = torch.cat([torch.tensor([0.4]), (0.1 / (C - 1)) * torch.ones(C - 1)]).cuda()

class Net(nn.Module):
    def __init__(self, encoder, oks=40):
        super(Net, self).__init__()
        self.e = encoder
        
        self.out = torch.nn.Linear(oks, 5)
    
    def run_m(self, m):
        def run(*x):
            return m(*x)
        return run
    
    def forward(self, x):
        with torch.cuda.amp.autocast(enabled=True):
            #print("start size", x.shape)
            x = x.permute((0,2,1))
            for i, l in enumerate(self.e.encoder):
                x = l(x)
            #x = self.e(x)
            #print("after c", x.shape)
            x = x.permute((0,2,1))
            out = self.out(x)
            out = torch.nn.functional.log_softmax(out, dim=-1)
            label_smoothing_loss = -((out * ls_weights.to(out.device)).mean())
            return out


    
torch.set_grad_enabled(True)
bmodelx = Model(cfgx)
model = Net(bmodelx.encoder)
model.cuda()

torch.set_grad_enabled(False)

model.load_state_dict(torch.load(sys.argv[1]))

#step = 500
#pad = 10
step = 400
pad = 100

model.eval()
model.cuda()
torch.set_grad_enabled(False)


def decode(signal_piece):
    alph = "NACGT"
    base_state = torch.zeros((1, 1, 5))
    decoder_state = torch.zeros((1, 1, 32))
    decoder_state, _ = model.b.gru(base_state, decoder_state)
    s_enc = signal_piece.unsqueeze(0).cpu()
    out = []

    for i in range(s_enc.shape[1]):
        #print(decoder_state[0,0].shape, s_enc[0,i].shape)
        base = model.j(s_enc[:1,i:i+1], decoder_state[:1,:1])[0][0][0].detach().numpy().argmax()
        if base != 0:
            base_state[:,:,:] = 0
            base_state[0,0,base] = 1
            decoder_state, _ = model.b.gru(base_state, decoder_state)
        out.append(alph[base])

    return "".join(out)

#dir_name = "../../training-data-nobackup/klebsiela/test_data/"
#dir_name = "../../training-data-nobackup/klebsiela/sample_no_restart/"
dir_name = sys.argv[2]
test_files = [os.path.join(dir_name, fn) for fn in os.listdir(dir_name)]

outg = open("%s-g.fasta" % sys.argv[3], "w")
outb5 = open("%s-b5.fasta" % sys.argv[3], "w")


model.cuda()
STEP = 1500
PAD = 100

for find, fn in enumerate(test_files):
    start = datetime.now()
    with h5py.File(fn, "r") as f:
        key = list(f["Raw/Reads"].keys())[0]
        signal_orig = np.array(f["Raw/Reads"][key]["Signal"][()], dtype=np.float32)
        signal = rescale_signal(signal_orig)
        
        print("go", find, len(signal_orig), len(signal), signal.dtype, datetime.now() - start)
        
        outputs = []
        batch = []
        for i in range(0, len(signal), 3*step):
            if i + 3*step + 6*pad > len(signal):
                break
            part = np.array(signal[i:i+3*step+6*pad])
            
            part = np.vstack([part]).T
            batch.append(part)
        
        print("b ready", datetime.now() - start)
            
        for i in range(0, len(batch), 100):
            net_result = F.softmax(model(torch.Tensor(np.stack(batch[i:i+100])).cuda()).detach().cpu(), dim=-1)
            
        #net_result = model_timed(model, torch.Tensor(batch).cpu()).detach().cpu().numpy()
            print("pred read", datetime.now() - start)
            for row in net_result:
#                decoded = decoder.decode(row.numpy())
#                outputs.append(decoded[pad:-pad])
                outputs.append(row[pad:-pad].numpy())

#        seq = []
#        last = 47
#        for o in outputs[:5]:
#            seq.append(o.replace("N", ""))
#        seq = "".join(seq)

        seqg = deepnano2.beam_search_py(np.vstack(outputs), 1, 0.1)
        seqb5 = deepnano2.beam_search_py(np.vstack(outputs), 5, 0.1)
#        seqb10 = deepnano2.beam_search_py(np.vstack(outputs), 10, 0.001)
#        seqb20 = deepnano2.beam_search_py(np.vstack(outputs), 20, 0.001)
        print("seq ready", datetime.now() - start, len(seqg), len(seqb5))
        print(">%d" % find, file=outg)
        print(seqg, file=outg)
        outg.flush()

        print(">%d" % find, file=outb5)
        print(seqb5, file=outb5)
        outb5.flush()

#        print(">%d" % find, file=outb10)
#        print(seqb10, file=outb10)
#        outb10.flush()

#        print(">%d" % find, file=outb20)
#        print(seqb20, file=outb20)
#        outb20.flush()

        print("done", find, fn, len(seqb5), datetime.now() - start)
