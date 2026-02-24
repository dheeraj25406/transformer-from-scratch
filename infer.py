import torch
import sys
import json
from tokenizers import Tokenizer

from transformer import build_transformer

device="mps" if torch.backends.mps.is_available() else "cpu"

# load config
with open("final_model/config.json") as f:
    config=json.load(f)

# load tokenizers
tokenizer_src=Tokenizer.from_file("final_model/tokenizer_en.json")
tokenizer_tgt=Tokenizer.from_file("final_model/tokenizer_it.json")

# build model
model=build_transformer(
    tokenizer_src.get_vocab_size(),
    tokenizer_tgt.get_vocab_size(),
    config["seq_len"],
    config["seq_len"],
    d_model=config["d_model"]
).to(device)

# load weights
ckpt=torch.load("final_model/model.pt",map_location=device)
model.load_state_dict(ckpt["model_state_dict"])
model.eval()


def greedy_decode(model,src,max_len,start_id,end_id):
    src_mask=torch.ones(1,1,1,src.size(1),device=device)

    ys=torch.ones(1,1,dtype=torch.long,device=device).fill_(start_id)

    for i in range(max_len):

        tgt_mask=torch.tril(torch.ones(1,1,ys.size(1),ys.size(1),device=device))

        encoder_output = model.encode(src, src_mask)

        for i in range(max_len):
            out = model.decode(encoder_output, src_mask, ys, tgt_mask)

        prob=model.project(out[:,-1])

        next_word=torch.argmax(prob,dim=1).item()

        ys=torch.cat([ys,torch.ones(1,1,dtype=torch.long,device=device).fill_(next_word)],dim=1)

        if next_word==end_id:
            break

    return ys

# input sentence
sentence=sys.argv[1]

src_tokens=tokenizer_src.encode(sentence).ids
src=torch.tensor(src_tokens).unsqueeze(0).to(device)

start_id=tokenizer_tgt.token_to_id("[SOS]")
end_id=tokenizer_tgt.token_to_id("[EOS]")

out=greedy_decode(model,src,config["seq_len"],start_id,end_id)

print(tokenizer_tgt.decode(out[0].tolist()))