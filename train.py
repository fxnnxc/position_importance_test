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
    

seed = 888
deterministic = False

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
if deterministic:
	torch.backends.cudnn.deterministic = True
	torch.backends.cudnn.benchmark = False

device='cuda:0'
learning_rate  =0.00002
batch_size = 1
epoch = 100
save_every = 10


date = datetime.today()            # 현재 날짜 가져오기

save_dir = os.path.join('outputs/', f'{datetime.today().month}_{datetime.today().day}')
log_dir = "outputs/log_dir"
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

# load model
config=GPT2Config(n_layer=2, vocab_size=train_dataset.vocab_len, eos_token_id=train_dataset.eos_token)
model = GPT2LMHeadModel(config=config).to(device)

# select variables to update while training
all_vars = [tensor for tensor in model.parameters()]

# create optimizer
optimizer = torch.optim.Adam(all_vars, lr=learning_rate)
optimizer.zero_grad()

# train

# if not os.path.exists(save_dir):
#     os.makedirs(save_dir)

# with open(os.path.join(SAMPLE_DIR, args.run_name, 'samples-{}.txt'.format(counter)), 'w') as fp:
        
for i in range(epoch):
    pbar = tqdm(range(len(train_dataset)))
    pbar.set_description(f" train - epoch {i+1}")
    avg = 0
    for idx in pbar:
        data = train_dataset[idx]
        data['input_ids'] = data['input_ids'].to(device)
        data['labels'] = data['input_ids'].to(device)
        # if idx > 1560:
        #     print(data)
        output = model(**data)
        
        loss = output.loss
        loss.backward()
        optimizer.step()
        
        avg += loss
        avg = avg / (idx + 1)
        
        # writer.add_scalar('Loss/Train',loss,epoch*len(train_dataset) + idx)        
        # pbar.set_postfix(avg_loss = avg.item())
    writer.add_scalar('Loss/Train',avg,i)
    if not(i % save_every) or (i == epoch - 1):
        save(model, save_dir, f'{i}')
    
    # evaluation
    model.eval()
    optimizer.zero_grad()
    print("Test for 5 random samples")
    for _ in range(5):
        index = np.random.randint(len(test_dataset))
        i = test_dataset[index]
        i['input_ids'] = i['input_ids'].to(device)
        print(i) 
        output = model.generate(i['input_ids'][:5].unsqueeze(0), max_new_tokens=3, min_new_tokens=3)
        print(output) 
            # fp.write('\n'.join(all_text))
        

    model.train()
    optimizer.zero_grad()

    


# -------------
model.eval()
model.pad_token_id = train_dataset.eos_token
# loader = DataLoader(test_dataset, batch_size=1)
pbar = tqdm(range(len(test_dataset)))
pbar.set_description(f" test ")
avg = 0
for idx in pbar:
    i = test_dataset[idx]
    i['input_ids'] = i['input_ids'].to(device)
    i['labels'] = i['input_ids'].to(device)
    output = model(**i)
    loss = output.loss
    
    avg += loss
    avg = avg / (idx + 1)

    pbar.set_postfix(avg_loss = avg)
writer.add_scalar('Loss/Test',avg,idx)

# -------------
# for i in loader:
#     i['input_ids'] = i['input_ids'].to(device)
#     print(i) 
#     output = model.generate(i['input_ids'][:,:5], max_new_tokens=3, min_new_tokens=3)
#     print(output) 
#     break


