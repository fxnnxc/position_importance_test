import os
from tqdm import tqdm
import time
import argparse
import torch    
import random
import numpy as np
from datetime import datetime
from pit.dataset.mathematical_shapes import MathematicalShapesDataset
from torch.utils.data import DataLoader
from transformers import GPT2LMHeadModel, GPT2Config
from torch.utils.tensorboard import SummaryWriter

#####################################
# function

def load_weights_lm_head(model, path):
    device = model.device()
    # pretrained_dict = original_model.state_dict()
    pretrained_dict = torch.load(path, map_location=device)

    model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    
    return model

def save(model, dir, name):
    os.makedirs(dir, exist_ok=True)
    save_path = os.path.join(dir, f'{name}.pth')
    torch.save(model.state_dict(), save_path)

parser = argparse.ArgumentParser(description='Fine-tune GPT-2 on your custom dataset.',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
parser.add_argument('--device', type=str, default='cuda:0')
# parser.add_argument('--pos_name', type=str, default='sinusoidal')
parser.add_argument('--optimizer', type=str, default='sgd')

parser.add_argument('--num-of-layers', type=int, default='2')
                    

args = parser.parse_args()
print(args)

seed = 888
deterministic = False

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
# if deterministic:
# 	torch.backends.cudnn.deterministic = True
# 	torch.backends.cudnn.benchmark = False

device=args.device

learning_rate=0.00002
batch_size = 128
epoch = 1000
save_every = 100


date = datetime.today()            # 현재 날짜 가져오기

save_dir = os.path.join('outputs/pretrained/', f"{args.num_of_layers}_{args.optimizer}")
log_dir = f"outputs/log_dir/{args.num_of_layers}_{args.optimizer}"
# save_dir = os.path.join('outputs/pretrained/', f'{datetime.today().month}_{datetime.today().day}')
# log_dir = f"outputs/log_dir/{datetime.today().month}_{datetime.today().day}" ### change
writer = SummaryWriter(log_dir)

# dataset 
num_shapes=101
train_dataset = MathematicalShapesDataset( # sum to not 100
                                    train=True,
                                    rule_indices=[0,4], 
                                    num_shapes=num_shapes,
                                    return_rule_label=True)

test_dataset = MathematicalShapesDataset(  # sum to 100
                                    train=False,
                                    rule_indices=[0,4], 
                                    num_shapes=num_shapes,
                                    num_samples=10000, 
                                    return_rule_label=True)

# change number of positions
# load model
config=GPT2Config(n_layer=args.num_of_layers, vocab_size=train_dataset.vocab_len, eos_token_id=train_dataset.eos_token)
model = GPT2LMHeadModel(config=config).to(device)

# select variables to update while training
all_vars = [tensor for tensor in model.parameters()]

# create optimizer
if args.optimizer == 'sgd':
    optimizer = torch.optim.Adam(all_vars, lr=learning_rate)
else:
    optimizer = torch.optim.SGD(all_vars, lr=learning_rate)
optimizer.zero_grad()

# train

# if not os.path.exists(save_dir):
#     os.makedirs(save_dir)

train_loader = DataLoader(train_dataset, batch_size)
pbar = tqdm(train_loader, total=len(train_dataset)//batch_size)

train_loss = []
for i in range(epoch):
    model.train()
    pbar.set_description(f" train - epoch {i+1}")
    avg = 0
    total = 0

    for idx, samples in enumerate(pbar):
        data = {}
        data['input_ids'] = samples['input_ids'].to(device)
        data['labels'] = samples['input_ids'].to(device)
        output = model(**data)
        
        loss = output.loss
        loss.backward()
        optimizer.step()
        
        total += loss
        
    avg = total / (len(train_dataset)//batch_size)
    train_loss.append(avg.item())
    writer.add_scalar(f'Loss/{args.num_of_layers}_{args.optimizer}/Train',avg,i)
    
    if not(i % save_every) or (i == epoch - 1):
        save(model, save_dir, f'{i}')
        
    if i == 0:
        best = avg
    
    if best > avg:
        save(model, save_dir, f'best')
        
        model.eval()
        with open((f'outputs/results/{args.num_of_layers}_{args.optimizer}.txt'), 'a') as fp:        
            fp.write(f"{args.num_of_layers}_{args.optimizer} \n")
            fp.write(f'epoch - {i} \n')
            for idx, d in enumerate(test_dataset):
                data = {}
                data['input_ids'] = d['input_ids'].to(device)
                output = model.generate(data['input_ids'][:5].unsqueeze(0), max_new_tokens=3, min_new_tokens=3)
                fp.write(f"{d['input_ids']}\n")
                fp.write(f"{output.detach().cpu()}\n")

    
    # evaluation
    # model.eval()
    # optimizer.zero_grad()
    # print("Test for 5 random samples")
    # for _ in range(5):
    #     index = np.random.randint(len(test_dataset))
    #     i = test_dataset[index]
    #     i['input_ids'] = i['input_ids'].to(device)
    #     print(i) 
    #     output = model.generate(i['input_ids'][:5].unsqueeze(0), max_new_tokens=3, min_new_tokens=3)
    #     print(output) 
    #         # fp.write('\n'.join(all_text))

    # model.train()
    # optimizer.zero_grad()

    


# -------------
# model.eval()
# model.pad_token_id = train_dataset.eos_token
# # loader = DataLoader(test_dataset, batch_size=1)
# pbar = tqdm(range(len(test_dataset)))
# pbar.set_description(f" test ")
# avg = 0
# for idx in pbar:
#     i = test_dataset[idx]
#     i['input_ids'] = i['input_ids'].to(device)
#     i['labels'] = i['input_ids'].to(device)
#     output = model(**i)
#     loss = output.loss
    
#     avg += loss
# avg /= len(test_dataset)//batch_size
# writer.add_scalar(f'Loss/{args.num_of_layers}_{args.optimizer}/Test',avg,0)

# -------------
# for i in loader:
#     i['input_ids'] = i['input_ids'].to(device)
#     print(i) 
#     output = model.generate(i['input_ids'][:,:5], max_new_tokens=3, min_new_tokens=3)
#     print(output) 
#     break

os.makedirs(f'outputs/loss', exist_ok=True)

with open((f'outputs/loss/{args.num_of_layers}_{args.optimizer}.txt'), 'w') as fp:
    fp.write(f'{args.num_of_layers}_{args.optimizer}\n')
    fp.write(f'{train_loss}\n')
    fp.write(f'{avg}')    

