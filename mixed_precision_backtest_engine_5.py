import numpy as np
import matplotlib.pyplot as plt
import glob
import os
import torch
import time
import transformer_engine.pytorch as te
from transformer_engine.common.recipe import Format, DelayedScaling
import numba as nb
import math

model_revision = 2
checkpoint_epoch = 1
batch_size = 1
time_size = 192 

x_window_size = 192

embedding_size = 128
transformer_size = 1024
transformer_attention_size = 32
state_embedding_size = transformer_size-(embedding_size*4)

sequence_length = time_size+1

start_cash = 100000

desired_reward = 1

actions_list = [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5]
long_rm_multiplier = .1
short_rm_multiplier = .1

dtype = torch.float32
np_dtype = np.float32
tlp_dtype = torch.float16
nlp_dtype = np.float16
fp8_format = Format.HYBRID  # E4M3 during forward pass, E5M2 during backward pass
fp8_recipe = DelayedScaling(fp8_format=fp8_format, amax_history_len=16, amax_compute_algo="max")

# Set environment variables for PyTorch
"""set path to CUDA library"""
os.environ['CUDA_HOME'] = '/usr/local/cuda-12.2'
device = 'cuda:0'

parsed_path = fr'/mnt/drive2/shared_data/unmerged_data/numpy/parsed_order_book/'
merged_array_path = fr'/mnt/drive2/shared_data/merged_data/numpy/merged_arrays/'

#model_state_dict_path = f'{main_path}/models/model_5_{model_revision}/checkpoint_{checkpoint_epoch}.pth'

softmax = torch.nn.Softmax(dim=0)

i = 0
k = 0
#in this function write me code that calculates the sharpe ratio of a list of profits when given the list of profits as well as the risk free rate

class PositionalEncoding(torch.nn.Module):
    def __init__(self, batch_size: int, time: int = 192, features: int = 1024, dropout: float = 0.1):
        super().__init__()
        self.dropout = torch.nn.Dropout(p=dropout)
        position = torch.arange(time).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, features, 2) * (-math.log(10000.0) / time_size))
        pe = torch.zeros((batch_size ,time, features))
        pe[:, :, 0::2] = torch.sin(position * div_term)
        pe[:, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe
        return self.dropout(x)

def sharpe_ratio_calc(arr, rfr):
    rfr = rfr/252
    #arr = np.array(arr)
    profit = arr[-1]
    std = np.std(arr)
    sharpe_ratio = (profit-rfr)/std
    return sharpe_ratio

@nb.njit(cache=True)
def min_max(arr):
    min = np.nanmin(arr)
    max = np.nanmax(arr)
    diff = np.subtract(max, min)
    result = np.divide(np.subtract(arr, min), diff)
    return result


def initialize_normal(layer):
    for name, param in layer.named_parameters():
        if 'weight' in name and param.data.dim() == 2:
            torch.nn.init.normal_(param, mean=0.0, std=.023)

def initialize_kaiming_uniform(layer):
    for name, param in layer.named_parameters():
        if 'weight' in name and param.data.dim() == 2:
            torch.nn.init.kaiming_uniform_(param)

def initialize_kaiming_normal(layer):
    for name, param in layer.named_parameters():
        if 'weight' in name and param.data.dim() == 2:
            torch.nn.init.kaiming_normal_(param)

def initialize_xavier_uniform(layer):
    for name, param in layer.named_parameters():
        if 'weight' in name and param.data.dim() == 2:
            torch.nn.init.xavier_uniform_(param)

def initialize_xavier_normal(layer):
    for name, param in layer.named_parameters():
        if 'weight' in name and param.data.dim() == 2:
            torch.nn.init.xavier_normal_(param)


"""[order_book_state, actions_state, rewards_state, positions_state, cash_state]"""
"""[order_book_state, actions_state, rewards_state, positions_state, cash_state]"""
class ShallowModel(torch.nn.Module):
    def __init__(self, init_method, embedding_size=128, transformer_size=1024, transformer_attention_size=32, batch_size=1024, dropout=0.0):
        super().__init__()
        self.batch_size = batch_size
        self.state_embedding = te.Linear(time_size*208, transformer_size-(embedding_size*4))
        self.action_embedding = te.Linear(time_size, embedding_size)
        self.reward_embedding = te.Linear(time_size, embedding_size)
        self.position_embedding = te.Linear(time_size, embedding_size)
        self.cash_embedding = te.Linear(time_size, embedding_size)
        
        self.transformer_layer = te.TransformerLayer(hidden_size=transformer_size, ffn_hidden_size=transformer_size, 
                                                     num_attention_heads=transformer_attention_size)
        self.transformer_layer.apply(init_method)
        #init_method(self.transformer_layer.weight)
        #self.transformer_layer.apply()
        
        
        self.flatten = torch.nn.Flatten(1, -1)
        self.linear = torch.nn.Linear(transformer_size, 11)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, input_tuple):
        book = self.dropout(self.state_embedding(torch.reshape(input_tuple[0], (self.batch_size, time_size*208))))
        actions = self.dropout(self.action_embedding(input_tuple[1]))
        cash = self.dropout(self.cash_embedding(input_tuple[4]))
        positions = self.dropout(self.position_embedding(input_tuple[3]))
        rewards = self.dropout(self.reward_embedding(input_tuple[2]))

        x = torch.cat((book, actions, cash, positions, rewards), dim=-1)
        x = torch.reshape(x, (self.batch_size, int(x.shape[0]/self.batch_size), x.shape[1]))
        x = self.dropout(self.transformer_layer(x))
        x = self.flatten(x)
        x = torch.reshape(x, (self.batch_size, int((x.shape[-1]/self.batch_size))))
        #x = torch.reshape(x, (batch_size, int((x.shape[-1]/batch_size))))
        x = self.linear(x)
        
        return x

class NormalModel(torch.nn.Module):
    def __init__(self, init_method, embedding_size=128, transformer_size=1024, transformer_attention_size=32, batch_size=1024, dropout=0.0):
        super().__init__()
        self.batch_size = batch_size
        self.state_embedding = te.Linear(time_size*208, transformer_size-(embedding_size*4))
        self.action_embedding = te.Linear(time_size, embedding_size)
        self.reward_embedding = te.Linear(time_size, embedding_size)
        self.position_embedding = te.Linear(time_size, embedding_size)
        self.cash_embedding = te.Linear(time_size, embedding_size)
        
        self.transformer_layer_1 = te.TransformerLayer(hidden_size=transformer_size, ffn_hidden_size=transformer_size, 
                                                       num_attention_heads=transformer_attention_size)
        self.transformer_layer_1.apply(init_method)
        self.transformer_layer_2 = te.TransformerLayer(hidden_size=transformer_size, ffn_hidden_size=transformer_size, 
                                                       num_attention_heads=transformer_attention_size)
        self.transformer_layer_2.apply(init_method)
        self.transformer_layer_3 = te.TransformerLayer(hidden_size=transformer_size, ffn_hidden_size=transformer_size, 
                                                       num_attention_heads=transformer_attention_size)
        self.transformer_layer_3.apply(init_method)

        self.flatten = torch.nn.Flatten(1, -1)
        self.linear = torch.nn.Linear(transformer_size, 11)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, input_tuple):
        book = self.dropout(self.state_embedding(torch.reshape(input_tuple[0], (self.batch_size, time_size*208))))
        actions = self.dropout(self.action_embedding(input_tuple[1]))
        cash = self.dropout(self.cash_embedding(input_tuple[4]))
        positions = self.dropout(self.position_embedding(input_tuple[3]))
        rewards = self.dropout(self.reward_embedding(input_tuple[2]))

        x = torch.cat((book, actions, cash, positions, rewards), dim=-1)
        x = torch.reshape(x, (self.batch_size, int(x.shape[0]/self.batch_size), x.shape[1]))
        
        x = self.dropout(self.transformer_layer_1(x))
        x = self.dropout(self.transformer_layer_2(x))
        x = self.dropout(self.transformer_layer_3(x))

        x = self.flatten(x)
        x = torch.reshape(x, (self.batch_size, int((x.shape[-1]/self.batch_size))))
        #x = torch.reshape(x, (batch_size, int((x.shape[-1]/batch_size))))
        x = self.linear(x)
        
        return x

class DeepModel(torch.nn.Module):
    def __init__(self, init_method, embedding_size=128, transformer_size=1024, transformer_attention_size=32, batch_size=1024, dropout=0.0):
        super().__init__()
        self.transformer_size = transformer_size
        self.batch_size = batch_size
        self.state_embedding = te.Linear(208, transformer_size-(embedding_size*4))
        self.action_embedding = torch.nn.Linear(1, embedding_size)
        self.reward_embedding = torch.nn.Linear(1, embedding_size)
        self.positions_embedding = torch.nn.Linear(1, embedding_size)
        self.cash_embedding = torch.nn.Linear(1, embedding_size)
        
        self.position_embedding = PositionalEncoding(batch_size, time_size, embedding_size, dropout)
        self.state_position_embedding = PositionalEncoding(batch_size, time_size, transformer_size-(embedding_size*4), dropout)

        self.transformer_layer_1 = te.TransformerLayer(hidden_size=transformer_size, ffn_hidden_size=transformer_size, 
                                                       num_attention_heads=transformer_attention_size)
        self.transformer_layer_1.apply(init_method)
        self.transformer_layer_2 = te.TransformerLayer(hidden_size=transformer_size, ffn_hidden_size=transformer_size, 
                                                       num_attention_heads=transformer_attention_size)
        self.transformer_layer_2.apply(init_method)
        self.transformer_layer_3 = te.TransformerLayer(hidden_size=transformer_size, ffn_hidden_size=transformer_size, 
                                                       num_attention_heads=transformer_attention_size)
        self.transformer_layer_3.apply(init_method)
        self.transformer_layer_4 = te.TransformerLayer(hidden_size=transformer_size, ffn_hidden_size=transformer_size, 
                                                       num_attention_heads=transformer_attention_size)
        self.transformer_layer_4.apply(init_method)
        
        self.flatten = torch.nn.Flatten(1, -1)
        self.linear = torch.nn.Linear(transformer_size*time_size, 11)
        self.dropout = torch.nn.Dropout(dropout)
        self.softmax = torch.nn.Softmax(dim=-1)
    def forward(self, input_tuple):
        state_input = torch.reshape(input_tuple[0], (self.batch_size, time_size, 208))
        embedded_state = self.state_embedding(state_input)
        book = self.state_position_embedding(embedded_state)
        actions = self.position_embedding(self.action_embedding(torch.unsqueeze(input_tuple[1], -1)))
        rewards = self.position_embedding(self.reward_embedding(torch.unsqueeze(input_tuple[2], -1)))
        positions = self.position_embedding(self.positions_embedding(torch.unsqueeze(input_tuple[3], -1)))
        cash = self.position_embedding(self.cash_embedding(torch.unsqueeze(input_tuple[4], -1)))

        x = torch.cat((book, actions, cash, positions, rewards), dim=-1)
        x = self.dropout(self.transformer_layer_1(x))
        x = self.dropout(self.transformer_layer_2(x))
        x = self.dropout(self.transformer_layer_3(x))
        x = self.dropout(self.transformer_layer_4(x))
        x = torch.reshape(x, (self.batch_size, int((x.shape[1]*x.shape[2]))))
        x = self.linear(x)

        return x
    
"""model.load_state_dict(torch.load(f'models/model_5_{model_revision}/checkpoint_{checkpoint_epoch}.pth')['model'])"""

load_path = '/mnt/drive2/models/hyperparameter_search/model_5_2/'
load_val_loss = 2.135453
load_batch_size = 256
config = {
    'embedding_size': 16,
    #'transformer_size': ([2048, 4096, 8192]),
    'transformer_size': 128,
    
    'transformer_attention_size': 32,
    "epochs": 500,
    "lr": 7.5e-7,
    #"lr": ([5e-4]),
    "batch_size": 1,
    'prefetch': 512,
    'num_workers': 6,
    'use_scheduler': False,
    'model_depth': 'deep',
    'dropout': 0.5,
    'transformer_init_method': initialize_normal,
    'optimizer': 'SGD'
}
model_name = f'/media/qhawkins/Archive/hyperparameter_models/model_5_{model_revision}/model_{model_revision}_{load_val_loss}_{config["model_depth"]}_{config["embedding_size"]}_{config["transformer_size"]}_{config["transformer_attention_size"]}_{load_batch_size}.pth'
    
if config['model_depth'] == 'shallow':
    model = ShallowModel(config['transformer_init_method'], config['embedding_size'], config['transformer_size'], config['transformer_attention_size'], batch_size, 
                        config['dropout'])
elif config['model_depth'] == 'normal':
    model = NormalModel(config['transformer_init_method'], config['embedding_size'], config['transformer_size'], config['transformer_attention_size'], batch_size, 
                        config['dropout'])
elif config['model_depth'] == 'deep':
    model = DeepModel(config['transformer_init_method'], config['embedding_size'], config['transformer_size'], config['transformer_attention_size'], batch_size, 
                    config['dropout'])


model.load_state_dict(torch.load(model_name))

#model.load_state_dict(torch.load("models/model_5_47/checkpoint__tv_137.pth")['model'])

model.eval()
model.to(device=device)

print('model loaded')

timeframe = f'{x_window_size}_{sequence_length-x_window_size}'

length_list = np.load("/mnt/drive2/shared_data/merged_data/numpy/consolidated_length_list/consolidated_length_list.npy")


def path_function(path):
    if not os.path.exists(path):
        os.makedirs(path)
        print(f'{path} created successfully')
    else:
        print(f'{path} already exists')


@nb.njit(cache=True)
def jit_z_score(x):
    mean_x = np.nanmean(x)
    std_x = np.nanstd(x)
    diff_x = np.subtract(x, mean_x)
    if (np.isnan(std_x) == False and np.isinf(std_x) == False) or (np.isnan(mean_x) == False and np.isinf(mean_x) == False):
        if std_x != 0:
            result = np.divide(diff_x, std_x)
        else:
            result = np.zeros_like(diff_x)
    else:
        print('nans or infs in z_score')
    
    #result.reshape((result.shape[0], result.shape[1], 1))
    return result

day = 0
    
cash_balance = start_cash
start_cash_balance = start_cash
total_profit_list = []
mid_price_list = []
action_storage = []
compounded_profit_list = []

log_path = f"backtests/model_{model_revision}_{load_val_loss}"
path_function(log_path)

log_file_path = f'{log_path}/model_{model_revision}_{load_val_loss}_backtesting_log.txt'

trades_file_path = f'{log_path}/model_{model_revision}_{load_val_loss}_trades_log.npy'
action_probabilities_file_path = f'{log_path}/model_{model_revision}_{load_val_loss}_action_probabilities_log.npy'
profit_file_path = f'{log_path}/model_{model_revision}_{load_val_loss}_profit_log.npy'
position_file_path = f'{log_path}/model_{model_revision}_{load_val_loss}_position_log.npy'
cash_file_path = f'{log_path}/model_{model_revision}_{load_val_loss}_cash_log.npy'
account_value_file_path = f'{log_path}/model_{model_revision}_{load_val_loss}_account_value_log.npy'

trades_log = np.zeros((length_list[-1]), dtype=np_dtype)
action_probabilities_log = np.zeros((length_list[-1], 11), dtype=np_dtype)
profit_log = np.zeros((length_list[-1]), dtype=np_dtype)
position_log = np.zeros((length_list[-1]), dtype=np_dtype)
cash_log = np.zeros((length_list[-1]), dtype=np_dtype)
account_value_log = np.zeros((length_list[-1]), dtype=np_dtype)

j = 0

rewards_state = torch.ones((batch_size, sequence_length-1), dtype=dtype, device=device)*desired_reward
cash = torch.zeros((1, sequence_length-1))
rewards = torch.zeros((1, sequence_length-1))


with torch.no_grad():
    for filename in sorted(glob.glob(os.path.join(parsed_path, '*.npy'))):
        arr = np.load(filename)

        arr[:, :, 1] = np.divide(arr[:, :, 1], 5000, out=np.zeros_like(arr[:, :, 1]), dtype=np_dtype, where=arr[:, :, 1]!=0)
        actions = torch.zeros((batch_size, sequence_length-1), dtype=dtype, device=device)
        positions = torch.zeros((batch_size, sequence_length-1), dtype=dtype, device=device)
        actions_state = torch.zeros((batch_size, sequence_length-1), dtype=dtype, device=device)
        cash = torch.zeros((batch_size, sequence_length-1), dtype=dtype, device=device)

        rewards = torch.zeros((batch_size, sequence_length-1), dtype=dtype, device=device)        
        cash[:, :] = start_cash
        rewards[:, :] = desired_reward
        day_profit_list = []
        
        position = 0
        action = 0
        reward = 0
        initial_j = j
        time_start = time.perf_counter()
        order_book_state = torch.zeros((batch_size, sequence_length-1, 104, 2), dtype=dtype, device=device)
        for i in range(sequence_length, len(arr)):
            
            order_book_state[0, :, :-4, 0] = torch.from_numpy(jit_z_score(arr[i-sequence_length+1:i, :, 1]).astype(np_dtype))
            order_book_state[0, :, :-4, 1] = torch.from_numpy(jit_z_score(arr[i-sequence_length+1:i, :, 1]).astype(np_dtype))
            
            #padded = torch.zeros((order_book_state.shape[0], time_size, 4, 2)).to(device)
            #order_book_state = torch.concat((order_book_state, padded), dim=2).to(dtype)
            
            mid_price = (arr[i, 50, 0]+arr[i, 49, 0])/2
            mid_price_list.append(mid_price)
            
            '''actions_state = z_score(actions), shape
            cash_state = torch.from_numpy(cash_state) (time, 1)'''
            actions_state = actions/5       
            """positions_state = z_score(positions), shape (time, 1))"""
            positions_state = min_max(positions)
            cash_state = min_max(cash)
            """rewards_state = z_score(rewards), shape (time, 1)"""
            
            

            output = model([order_book_state, actions_state, rewards_state, positions_state, cash_state])
            

            output = output[0, :]
            output = softmax(output)
            action = torch.argmax(output, dim=0)
            action = actions_list[action.item()]
            action_storage.append(action)

            current_exposure = position * mid_price

            """if action is positive and cost of creating new position is smaller than the current cash balance """
            if (action > 0 and (mid_price * action) < cash_balance and current_exposure < start_cash*long_rm_multiplier) or (action < 0 and current_exposure > -start_cash*short_rm_multiplier):
                cash_balance = cash_balance - (mid_price * action) - (.0035 * abs(action))
                position += action
            
            actions[:, :-1] = actions[:, 1:].clone()
            actions[0, -1] = action
            positions[:, :-1] = positions[:, 1:].clone()
            positions[0, -1] = position
            cash[:, :-1] = cash[:, 1:].clone()
            cash[0, -1] = cash_balance




            account_value = (position * mid_price) + cash_balance
            total_profit = ((account_value - start_cash)/start_cash)*100
            day_profit = ((account_value - start_cash_balance)/start_cash_balance)*100

            trades_log[j] = action
            action_probabilities_log[j] = output.cpu().numpy()
            profit_log[j] = total_profit
            position_log[j] = position
            cash_log[j] = cash_balance
            account_value_log[j] = account_value

            total_profit_list.append(total_profit)
            compounded_profit_list.append(total_profit)
            day_profit_list.append(day_profit)
            #mid_price_list.append(mid_price)
            #print(position, cash_balance, mid_price, action)
            j += 1
        
        cash_balance = cash_balance + (position * mid_price) - (.0035 * abs(position))
        start_cash_balance = cash_balance
        time_end = time.perf_counter()
        day+=1
        print(f'minimum position for day {day}: {np.min(position_log[initial_j:j])}')
        print(f'maximum position for day {day}: {np.max(position_log[initial_j:j])}')
        print(f'average position for day {day}: {np.mean(position_log[initial_j:j])}')
        day_info = f'day: {day}, time for day: {round(time_end-time_start, 4)}, remaining time: {round(((time_end-time_start)*(len(length_list)-day))/3600, 4)} hours, total_profit: {round(total_profit, 4)}%, sharpe ratio: {sharpe_ratio_calc(day_profit_list, .0552)}, average action: {round(np.mean(action_storage), 4)}\n'
        with open(log_file_path, "a") as f:
            f.write(day_info)

        print(day_info)
        action_storage = []
        total_profit_list = []

trades_log = trades_log[:j]
action_probabilities_log = action_probabilities_log[:j]
profit_log = profit_log[:j]
position_log = position_log[:j]
cash_log = cash_log[:j]
account_value_log = account_value_log[:j]

np.save(trades_file_path, trades_log)
np.save(action_probabilities_file_path, action_probabilities_log)
np.save(profit_file_path, profit_log)
np.save(position_file_path, position_log)
np.save(cash_file_path, cash_log)
np.save(account_value_file_path, account_value_log)

plt.plot(compounded_profit_list)
#plt.plot(mid_price_list)
plt.show() 
