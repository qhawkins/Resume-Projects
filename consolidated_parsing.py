import numpy as np
import glob
import os
import torch
import numba as nb

precision = np.float16
torch_precision = torch.float16
start_offset = 193
pick_frequency = 1
start_delay = 0
csv_action_path = '/mnt/drive2/shared_data/unmerged_data/csv/actions'
csv_reward_path = '/mnt/drive2/shared_data/unmerged_data/csv/rewards'
csv_position_path = '/mnt/drive2/shared_data/unmerged_data/csv/positions/'
csv_cash_path = '/mnt/drive2/shared_data/unmerged_data/csv/cash/'

pytorch_shared_folder = "/mnt/drive2/shared_data/unmerged_data/pytorch"
numpy_raw_data_folder = "/mnt/drive2/shared_data/unmerged_data/numpy/raw_order_book"
numpy_parsed_order_book = "/mnt/drive2/shared_data/unmerged_data/numpy/parsed_order_book"
numpy_merged_array_path = "/mnt/drive2/shared_data/merged_data/numpy/merged_arrays"
numpy_consolidated_length_list_path = "/mnt/drive2/shared_data/merged_data/numpy/consolidated_length_list"
path_list = [csv_action_path, csv_reward_path, csv_position_path, csv_cash_path]

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
    
    return result

def parser_func():
    filename_list = []
    length_list = []
    length = 0
    i = 0
    k = 0

    for filename in sorted(glob.glob(os.path.join(numpy_raw_data_folder, '*.npy'))):
        arr = np.load(filename, mmap_mode='r')
        length += arr.shape[2]-start_delay
        length_list.append(length)    
    
    storage_arr = np.zeros((100, 2, length_list[-1])).astype(np.float32)
    
    for filename in sorted(glob.glob(os.path.join(numpy_raw_data_folder, '*.npy'))):
        arr = np.load(filename)
        filename_list.append(filename)
        if k == 0:
            storage_arr[:, :2, :length_list[0]] = arr[:, :2, start_delay:]
            k += 1
        else:
            storage_arr[:, :2, length_list[k - 1]:length_list[k]] = arr[:, :2, start_delay:]
            k += 1

        print(f'loaded {filename}')


    categorical = np.split(storage_arr, length_list[:-1], axis=2)
    
    for arr in categorical:
        arr = np.transpose(arr, (2, 0, 1))
        arr = arr.astype(np.float32)

        print(f'saving {filename_list[i]}')
        np.save(file=filename_list[i].replace("raw_order_book", "parsed_order_book"), arr=arr)
        print(f'saved {filename_list[i]}')
        print(arr.shape)
        print(f'{filename} has {np.sum(np.isnan(arr))} nans')
        categorical[i] = arr
        i += 1
        
    np.save(file=rf'{numpy_consolidated_length_list_path}/consolidated_length_list.npy', arr=length_list)


    merged_array = np.concatenate(categorical, axis=0)
    print(f'merged array shape: {merged_array.shape}')
    np.save(file=fr'{numpy_merged_array_path}/merged_array.npy', arr=merged_array)
    print('merged array saved')
    return length_list

consolidated_order_book = np.zeros((25000, start_offset-1, 100, 2), dtype=precision, order='C')
consolidated_actions = np.zeros((25000, start_offset-1), dtype=precision, order='C')
consolidated_labels = np.zeros((25000, 1), dtype=precision, order='C')
consolidated_rewards = np.zeros((25000, start_offset-1), dtype=precision, order='C')
consolidated_positions = np.zeros((25000, start_offset-1), dtype=precision, order='C')
consolidated_cash = np.zeros((25000, start_offset-1), dtype=precision, order='C')

order_book_i = 0
actions_i = 0
rewards_i = 0
positions_i = 0
cash_i = 0

def path_function(path):
    split = path.split("_")[-1].split(".")[0]
    return int(split)

shape_list = []
length = 0
#shape_list_1 = []
order_book_day = 0

@nb.njit(cache=True)
def min_max(arr):
    min = np.nanmin(arr)
    max = np.nanmax(arr)
    diff = np.subtract(max, min)
    result = np.divide(np.subtract(arr, min), diff)
    return result


if input("create .pt files from csv? (y/n)").lower() == 'y':
    """KEEP IN MIND THAT THE ORDER BOOK DATA IS ALWAYS N LARGER THAN THE DATA GENERATED FROM C++ BECAUSE OF THE OFFSET, IN THIS CASE ITS 200 OFF OF THE END OF THE DATA, SHOULD BE INDEXED AS ARR[:-200, :, :, :]"""
    for path in path_list:
        for filename in sorted(glob.glob(os.path.join(path, '*.csv')), key=path_function):
            arr = np.loadtxt(filename, delimiter=',')
            storage_arr = np.zeros((arr.shape[0], start_offset-1)).astype(precision, copy=False)
            print(f'{filename} has {np.sum(np.isnan(arr))} nans')
            k = 0

            if path == csv_action_path:
                storage_labels = np.zeros((arr.shape[0], 1)).astype(precision, copy=False)
                for i in range(start_offset, arr.shape[0]):
                    if i % pick_frequency == 0:    
                        #storage_arr[k, :] = z_score(arr[i-start_offset:i-1])
                        
                        """removed normalization from actions and switched to dividing by 5"""

                        storage_arr[k, :] = arr[i-start_offset:i-1]/5
                        
                        storage_labels[k, :] = arr[i]
                        k+=1
                        len_storage = i
                storage_arr = torch.tensor(storage_arr[:k-1, :], dtype=torch_precision)
                storage_labels = torch.tensor(storage_labels[:k-1, :], dtype=torch_precision)
                torch.save(storage_arr, filename.replace('/csv/', '/pytorch/').replace('.csv', '.pt'))
                torch.save(storage_labels, filename.replace('actions', 'labels').replace('/csv/', '/pytorch/').replace('.csv', '.pt'))
                print(f'actions storage_arr shape: {storage_arr.shape}')
                print(f'storage_labels shape: {storage_labels.shape}')        

            elif path == csv_reward_path:
                for i in range(start_offset, arr.shape[0]):
                    if i % pick_frequency == 0:
                        
                        """undid normalization, leaving it raw"""
                        """normalizing rewards to be between 0-1, min max scaling"""
                        
                        storage_arr[k, :] = min_max(arr[i-start_offset:i-1])

                        
                        k+=1
                        len_storage = i
                
                storage_arr = torch.tensor(storage_arr[:k-1, :], dtype=torch_precision)
                torch.save(storage_arr, filename.replace('/csv/', '/pytorch/').replace('.csv', '.pt'))
                print(f'rewards storage_arr shape: {storage_arr.shape}')

            elif path == csv_position_path:
                for i in range(start_offset, arr.shape[0]):
                    if i % pick_frequency == 0:        
                        """used to be normalizaed by just dividing by 500"""
                        storage_arr[k, :] = min_max(arr[i-start_offset:i-1])
                        """
                        storage_arr[k, :] = z_score(arr[i-start_offset:i-1])
                        """
                        k+=1
                        len_storage = i
                storage_arr = torch.tensor(storage_arr[:k-1, :], dtype=torch_precision)
                torch.save(storage_arr, filename.replace('/csv/', '/pytorch/').replace('.csv', '.pt'))
                print(f'positions storage_arr shape: {storage_arr.shape}')

            elif path == csv_cash_path:
                for i in range(start_offset, arr.shape[0]):
                    if i % pick_frequency == 0:        
                        """
                        storage_arr[k, :] = jit_z_score(arr[i-start_offset:i-1]).astype(precision)
                        """
                        """used to be scaled by dividing by 100000"""
                        storage_arr[k, :] = min_max(arr[i-start_offset:i-1])
                        k+=1
                        len_storage = i
                storage_arr = torch.tensor(storage_arr[:k-1, :], dtype=torch_precision)
                torch.save(storage_arr, filename.replace('/csv/', '/pytorch/').replace('.csv', '.pt'))
                length += len_storage
                print(f'cash storage_arr shape: {storage_arr.shape}')
                shape_list.append(length)


if input("create consolidated order book? (y/n): ").lower() == 'y':
    if input("parse arrays? (y/n): ").lower() == 'y':
        shape_list = parser_func()
    
    if input('convert length list to csv? (y/n)') == 'y':
        np.savetxt(fr'{numpy_merged_array_path}/length_list.csv', shape_list, delimiter=',')
        print('length list saved as csv')

    if input('convert merged array to mid price csv? (y/n)') == 'y':
        merged_array = np.load(fr'{numpy_merged_array_path}/merged_array.npy')
        mid_price = (merged_array[:, 49, 0] + merged_array[:, 50, 0]) / 2
        mid_price = np.reshape(mid_price, (1, mid_price.shape[0]))
        print(mid_price.shape)
        np.savetxt(fr'{numpy_merged_array_path}/mid_prices.csv', mid_price, delimiter=',')
        print('mid price saved as csv')

    for filename in sorted(glob.glob(os.path.join(numpy_parsed_order_book, '*.npy'))):
        arr = np.load(filename)
        
        """KEEP IN MIND THAT THE ORDER BOOK DATA IS ALWAYS N LARGER THAN THE DATA GENERATED FROM C++ BECAUSE OF THE OFFSET, IN THIS CASE ITS 200 OFF OF THE END OF THE DATA, SHOULD BE INDEXED AS ARR[:-200, :, :]"""
        arr = arr[:-200, :, :]
        order_book_i = 0
        consolidated_order_book = np.zeros((25000, start_offset-1, 100, 2), dtype=precision, order='C')
        arr_0 = np.zeros((start_offset-1, 100, 1), dtype=precision)
        arr_1 = np.zeros((start_offset-1, 100, 1), dtype=precision)

        shape_list.append(arr.shape[0])
        print(f'{filename} has {np.sum(np.isnan(arr))} nans')
        for i in range(start_offset, arr.shape[0]):
            if i % pick_frequency == 0:
                arr_0 = jit_z_score(arr[i-start_offset:i-1, :, 0]).astype(precision, copy=False)
                arr_1 = jit_z_score(arr[i-start_offset:i-1, :, 1]).astype(precision, copy=False)
                arr_2 = np.stack((arr_0, arr_1), -1)
                consolidated_order_book[order_book_i, :, :, :] = arr_2.astype(precision, copy=False)
                order_book_i += 1

        consolidated_order_book = consolidated_order_book[:order_book_i-1, :, :, :]
        print(f'consolidated order book shape: {consolidated_order_book.shape}')
        torch.save(torch.tensor(consolidated_order_book, dtype=torch_precision), f"{pytorch_shared_folder}/order_book/order_book_day_{order_book_day}.pt")
        order_book_day += 1
