#include "torch/torch.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <map>
#include <numeric>
#include <functional>
#include <algorithm>
#include <span>
#include <chrono>
#include <memory>
#include <torch/script.h>
#include <filesystem>
#include <cmath>
#include <tuple>
#include <utility>
#include <chrono>
#include <random>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <queue>
#include <future>
#include "ThreadPool.h"
#include "ThreadPool.cpp"


std::vector<std::vector<float>> load_csv(const std::string& file_path) {
    std::vector<std::vector<float>> data;
    std::ifstream file(file_path);

    if (file.is_open()) {
        std::string line;
        while (std::getline(file, line)) {
            std::vector<float> row;
            std::istringstream line_stream(line);
            std::string cell;

            while (std::getline(line_stream, cell, ',')) {
                row.push_back(std::stof(cell));
            }

            data.push_back(row);
        }
        file.close();
    }
    else {
        std::cerr << "Unable to open the file: " << file_path << std::endl;
    }
    return data;
}


std::vector<std::vector<float>> old_load_csv(const std::string& filename) {
    std::vector<std::vector<float>> data;
    std::ifstream file(filename);

    if (!file.is_open()) {
        std::cerr << "Error opening file: " << filename << std::endl;
        return data;
    }

    std::string line;
    while (std::getline(file, line)) {
        std::vector<float> row;
        std::istringstream line_stream(line);

        std::string cell;
        while (std::getline(line_stream, cell, ',')) {
            float value = std::stod(cell);
            row.push_back(value);
        }

        data.push_back(row);
    }

    file.close();
    return data;
}


class Environment {
public:
    //int timesetp_offset = 0;
    Environment() {}

    Environment(std::vector<float>& prices, std::vector<float>& predictions, std::vector<int>& length_list, int random_index, int num_steps, int offset_init, float gamma_init, int time) {
        //std::vector<std::vector<float>> price_vec = prices;
        //std::vector<std::vector<float>> predictions_vec = predictions;
        prices_v = std::vector<float>(prices.begin(), prices.end());
        predictions_v = std::vector<float>(predictions.begin(), predictions.end());
        length_list_v = length_list;
        //std::cout << "after length list" << std::endl;
        position_history.resize(prices_v.size());
        cash_history.resize(prices_v.size());
        total_profit_history.resize(prices_v.size());
        st_profit_history.resize(prices_v.size());
        action_history.resize(prices_v.size());
        w_short = 0.5;
        w_long = 0.5;
        offset = offset_init;
        gamma = gamma_init;
        time_dim = time;
        //int timestep_offset = 0;
        //int buffer = end_of_ep_buffer;
        //std::vector<int> position_history(prices_v.size());
        //std::cout << "position history dimensions" << position_history.size() << std::endl;
        //delete(&price_vec);
        //delete(*predictions_vec);

    }
    //step
    void reset(std::vector<float>& prices, std::vector<float>& predictions, std::vector<int>& length_list, int random_index, int num_steps) {
        std::vector<float> prices_v = std::vector<float>(prices.begin(), prices.end());
        std::vector<float> predictions_v = std::vector<float>(predictions.begin(), predictions.end());
        prediction = 0;
        vec_mean = 0;
        vec_std = 0;
        done = false;
        last_trade_tick = time_dim;
        current_tick = time_dim;
        cash = 100000;
        start_cash = 100000;
        position = 0;
        account_value = 0;
        profit_list.clear();
        step_reward = 0;
        total_reward = 0;
        total_profit = 0;
        action_taken = 0;
        past_profit = 0;
        st_profit = 0;
        trade = false;
        //position_history.clear();
        //std::vector<float> state = torch::zeros(40);
        std::pair<std::vector<std::vector<float>>, float> temp_pair; // = get_state(20);
        temp_pair.first = std::vector<std::vector<float>>(time_dim, std::vector<float>(7));
        temp_pair.second = 0;
        state = temp_pair.first;
        price = temp_pair.second;
        int action = 0;
        int previous_action = 0;
        cash_history.clear();
        cash_history.resize(prices_v.size());
        total_profit_history.clear();
        total_profit_history.resize(prices_v.size());
        st_profit_history.clear();
        st_profit_history.resize(prices_v.size());
        position_history.clear();
        position_history.resize(prices_v.size());
        action_history.clear();
        action_history.resize(prices_v.size());
        sharpe_history = 0;
        position_penalty_history = 0;
        weighted_profit_history = 0;
        total_profit_reward = 0;
        omega_ratio_history = 0;
        previous_action_reward = 0;
        int timestep_offset = 0;
    }

    void timestep_offset_update(int offset) {
        timestep_offset += offset + time_dim;
    }
    std::vector<std::vector<float>> get_state_env() {
        return state;
    }

    int get_timestep_offset() {
		return timestep_offset;
	}

    float get_price() {
        return price;
    }

    std::vector<int> get_position_history() {
        return position_history;
    }

    std::vector<float> get_cash_history() {
        return cash_history;
    }

    std::vector<float> get_total_profit_history() {
        return total_profit_history;
    }

    float get_step_reward() {
        return step_reward;
    }

    float get_previous_action_reward() {
        return previous_action_reward;
    }

    float get_total_reward() {
        return total_reward;
    }

    float get_total_profit() {
        return total_profit;
    }

    int get_position() {
        return position;
    }

    float get_account_value() {
        return account_value;
    }

    float get_sharpe_history() {
        return sharpe_history;
    }

    float get_position_penalty() {
        return position_penalty_history;
    }

    float get_weighted_profit_history() {
        return weighted_profit_history;
    }

    float get_total_profit_reward() {
        return total_profit_reward;
    }

    float get_omega_ratio() {
        return omega_ratio_history;
    }

    float get_cash() {
        return cash;
    }

    bool step(int action, int timestep, int random_index, int num_steps, int end_of_ep_buffer) {
        current_tick = timestep + timestep_offset;
        bool ep_buffer_hit = false;
        //std::cout << "step action: " << action << std::endl;
        if (binary_search(length_list_v.begin(), length_list_v.end(), current_tick + random_index + end_of_ep_buffer)) {
            bool ep_buffer_hit = true;

            if (position > 0) {
                cash = cash + (position * price) - (.0035 * abs(position));
                position = 0;
                //std::cout << "long close" << std::endl;
            }
            else if (position < 0) {
                cash = cash + (position * price) - (.0035 * abs(position));
                position = 0;
                //std::cout << "short close" << std::endl;
            }
        }


        temp_pair.first = std::vector<std::vector<float>>(time_dim, std::vector<float>(7));
        temp_pair.second = 0;
        temp_pair = get_state(current_tick);


        //std::cout <<  << std::endl;
        state = temp_pair.first;
        //std::cout << "step state" << std::endl;
        price = temp_pair.second;
        //std::cout << "price" << std::endl;
        past_profit = total_profit;
        total_profit = ((position * price) + cash) / start_cash;
        //std::cout << "total profit: " << total_profit << std::endl;
        st_profit = total_profit - past_profit;
        st_profit_history[current_tick] = st_profit;
        //std::cout << "calculate_reward" << std::endl;
        trade = false;
        //std::cout << "action: " << action << std::endl;
        if ((action > 0 && (price * action) < cash) || (action < 0 && ((position * price) + (action * price) > -100000))) {
            trade = true;
            //std::cout << "trade=true" << std::endl;
        }
        if (trade == true) {
            last_trade_tick = current_tick;
            if (action > 0) {
                cash = cash - (price * action) - (.0035 * action);
                position += action;
                //std::cout << "buy" << std::endl;
                //std::cout << "position: " << position << std::endl;

            }
            else if (action < 0) {
                cash = cash - (price * action) + (.0035 * action);
                position += action;
                //std::cout << "sell" << std::endl;
                //std::cout << "position: " << position << std::endl;
            }
        }
        position_history[current_tick] = position;
        account_value = (position * price) + cash;
        //std::cout << "price: " << price << " tick: " << current_tick << std::endl;
        total_profit = account_value / start_cash;
        //std::cout << "total profit: " << total_profit << std::endl;
        cash_history[current_tick] = cash;
        total_profit_history[current_tick] = total_profit;
        action_history[current_tick] = action;


        if (current_tick == time_dim) {
            for (int i = 0; i < time_dim; i++) {
                position_history[i] = 0;
                cash_history[i] = 100000;
            }
        }



        //weighted_profit_history[current_tick] = weighted_profit;
        step_reward = calculate_reward(previous_action, action);
        previous_action = action;
        return ep_buffer_hit;
    }

    float calculate_profit(int current_tick, int n) {
        int end_tick = current_tick + n;
        float start_value = (position * price) + cash;
        //std::cout << "start value: " << start_value << std::endl;
        float end_value = (position_history[end_tick] * prices_v[end_tick]) + cash_history[end_tick];
        //std::cout << "position history: " << position_history[start_tick] << std::endl;
        //std::cout << "prices_v: " << prices_v[start_tick] << std::endl;
        //std::cout << "cash_history: " << cash_history[start_tick] << std::endl;
        //std::cout << "end val - start val: " << end_value - start_value << std::endl;
        //exit(8280);
        return (end_value - start_value) / start_value;
    }



private:
    std::vector<int> position_history;
    std::vector<std::vector<float>> state;
    std::vector<float> predictions_v;
    std::vector<float> prices_v;
    std::vector<int> length_list_v;
    int timestep_offset = 0;
    float step_reward;
    float total_profit;
    bool done;
    float total_reward;
    std::tuple<std::vector<float>, float, int, float, float> interim_info;
    int item;
    float price;
    float prediction;
    float vec_mean;
    float vec_std;
    int last_trade_tick;
    int current_tick;
    int end_tick = prices_v.size();
    float start_cash = 100000;
    float cash = 100000;
    int position;
    float account_value;
    std::vector<float> profit_list;
    int action_taken;
    float past_profit;
    float st_profit;
    bool trade;
    int n_short = 5;
    int n_long = 10;
    float w_short = .5;
    float w_long = .5;
    std::vector<float> cash_history;
    std::vector<float> total_profit_history;
    std::vector<float> st_profit_history;
    std::vector<float> action_history;
    float sharpe_history;
    float position_penalty_history;
    float weighted_profit_history;
    float total_profit_reward;
    float omega_ratio_history;
    float position_penalty;
    float previous_action_reward;
    std::pair<std::vector<std::vector<float>>, float> temp_pair;
    int previous_action;
    int offset;
    float gamma;
    int time_dim;

    std::pair<std::vector<std::vector<float>>, float> get_state(int tick) {
        std::vector<float> price_s(prices_v.begin() + tick - time_dim, prices_v.begin() + tick);
        std::vector<float> predictions_s(predictions_v.begin() + tick - time_dim, predictions_v.begin() + tick);
        std::vector<float> position_s(position_history.begin() + tick - time_dim, position_history.begin() + tick);
        std::vector<float> cash_s(cash_history.begin() + tick - time_dim, cash_history.begin() + tick);
        std::vector<float> st_profit_s(st_profit_history.begin() + tick - time_dim, st_profit_history.begin() + tick);
        std::vector<float> total_profit_s(total_profit_history.begin() + tick - time_dim, total_profit_history.begin() + tick);
        std::vector<float> action_s(action_history.begin() + tick - time_dim, action_history.begin() + tick);
        //std::cout << "price_s: " << price_s.size() << std::endl;

        price = price_s[price_s.size() - 1];
        std::vector<float> n_price = normalize_data(price_s);
        std::vector<float> n_predictions = normalize_data(predictions_s);

        std::vector<float> n_position = normalize_data(position_s);
        std::vector<float> n_cash = normalize_data(cash_s);
        std::vector<float> n_st_profit = normalize_data(st_profit_s);
        std::vector<float> n_total_profit = normalize_data(total_profit_s);
        //std::vector<float> n_action = normalize_data(action_s);
        std::vector<float> n_action(action_s.size());
        for (int i = 0; i < action_s.size(); ++i) {
            n_action[i] = action_s[i] / 5;
        }

        std::vector<std::vector<float>> state(time_dim, std::vector<float>(7));

        for (int i = 0; i < time_dim; ++i) {
            state[i][0] = n_price[i];
            state[i][1] = n_predictions[i];
            state[i][2] = n_position[i];
            state[i][3] = n_cash[i];
            state[i][4] = n_st_profit[i];
            state[i][5] = n_total_profit[i];
            state[i][6] = n_action[i];
        }

        return make_pair(state, price);

    }

    std::vector<float> normalize_data(std::vector<float>& data) {
        float sum = 0.0;
        float variance = 0.0;

        // Find sum and set very small absolute values to 0.
        for (float& x : data) {
            if (abs(x) < 1e-6) {
                x = 0.0;
            }
            sum += x;
        }

        float mean = sum / data.size();

        // Calculate variance in separate loop
        for (float& x : data) {
            variance += pow(x - mean, 2);
        }

        variance /= data.size();
        float stdDeviation = sqrt(variance);

        // Normalize data
        for (float& x : data) {
            x = (x - mean) / (stdDeviation + 1e-5);
        }

        return data;
    }


    std::vector<float> future_profits(int buffer_len, int position, int current_tick) {
        std::vector<float> fut_profit(buffer_len);
        float initial_basis = (position * prices_v[current_tick]) + cash;
        for (int i = 0; i < buffer_len; ++i) {
            fut_profit[i] = (5000 * (((position * prices_v[current_tick + i]) + cash) - initial_basis)) / initial_basis;
        }

        return fut_profit;
    }

    std::vector<float> future_positions(int buffer_len, int position, int current_tick) {
        std::vector<float> fut_position(buffer_len);
        for (int i = current_tick; i < current_tick + buffer_len; ++i) {
            float position_reward;

            if (position > 50) {
                position_reward = -.5;
            }
            else {
                position_reward = 0.1;
            }
            fut_position[i - current_tick] = position_reward;
        }
        return fut_position;
    }

    float weighted_future_rewards(std::vector<float> unweighted_vector, float gamma) {
        for (int i = 0; i < unweighted_vector.size(); ++i) {
            unweighted_vector[i] = unweighted_vector[i] * std::pow(gamma, i);
            //std::cout << unweighted_vector[i] * std::pow(gamma, i) << std::endl;
        }
        return std::accumulate(unweighted_vector.begin(), unweighted_vector.end(), 0.0);
    }

    float calculate_reward(int previous_action, int action) {
        //float f_position = position;
        float step_reward = 0.0;
        std::vector<float> profit_vec = future_profits(offset, position, current_tick);
        step_reward += weighted_future_rewards(profit_vec, gamma);
        weighted_profit_history += step_reward;
        //if (current_tick > 120) {
        //    step_reward = sharpe_ratio() * .1;

            //std::cout << "sharpe ratio: " << step_reward << std::endl;
            //std::cout << "sharpe step reward: " << step_reward << std::endl;
        //}
        //else
        //{
        //    step_reward = 0.0;
            //std::cout << "sharpe ratio: " << step_reward << std::endl;
            //std::cout << "sharpe step reward: " << step_reward << std::endl;
        ///}
        //sharpe_history += step_reward;
        if (abs(position) > 50) {
            //std::cout << "abs position: " << abs(f_position)/1000 << std::endl;
            step_reward -= (abs(float(position)) / 1000);
            //std::cout << "step reward 1: " << step_reward << std::endl;
            position_penalty_history -= (abs(float(position)) / 1000);
            //std::cout << "position: " << f_position << std::endl;
            //std::cout << "position step reward: " << step_reward << std::endl;
        }
        //else {
        //    step_reward += .5;
            //std::cout << "step reward 2: " << step_reward << std::endl;
            //position_penalty_history += .5;
            //std::cout << "position step reward: " << step_reward << std::endl;
        //}
        //std::cout << "step reward 2: " << step_reward << std::endl;
        //if (current_tick > 120) {
            //float omega_ratio = calculateOmegaRatio();
            //step_reward += std::sqrt(omega_ratio);
            //std::cout << "step reward 3: " << step_reward << std::endl;
            //omega_ratio_history += std::sqrt(omega_ratio);
            //std::cout << "omega ratio: " << omega_ratio << std::endl;
            //std::cout << "step: " << current_tick << std::endl;
        //}
        //else {
            //step_reward += 0.0;
            //std::cout << "step reward 4: " << step_reward << std::endl;
            //omega_ratio_history += 0.0;
        //}

        //step_reward += weighted_profit;
        //std::cout << "step reward 5: " << step_reward << std::endl;
        //weighted_profit_history += weighted_profit;
        //std::cout << "weighted profit: " << weighted_profit << std::endl;
        //std::cout << "weighted profit: " << weighted_profit << std::endl;
        //std::cout << "weighted profit step reward: " << step_reward << std::endl;
        //step_reward += (total_profit - 1) * 125;
        //std::cout << "total profit: " << total_profit << std::endl;
        //std::cout << "step reward 6: " << step_reward << std::endl;
        //total_profit_reward += (total_profit - 1) * 125;

        //if (previous_action != action) {
        //    step_reward += .5;
        //    previous_action_reward += .5;
        //}
        //else {
        //    step_reward -= .5;
        //    previous_action_reward -= .5;
        //}
        //std::cout << "total profit: " << total_profit << std::endl;
        //std::cout << "total profit step reward: " << step_reward << std::endl;
        //std::cout << "timestep: " << current_tick << std::endl;
            //std::cout << "weighted profit: " << weighted_profit << std::endl;
        //std::cout << current_tick << " step reward: " << step_reward << std::endl;
        //std::cout << "============================" << std::endl;

        total_reward += step_reward;
        //std::cout << "total reward: " << total_reward << std::endl;
        //std::cout << step_reward << std::endl;
        return step_reward;
    }

};

class ActorCritic : public torch::nn::Module {
public:
    int action_dim;
    torch::Tensor action_var;
    torch::nn::Sequential actor, critic;

    ActorCritic() {}

    ActorCritic(int state_dim, int action_dim, bool has_continuous_action_space, float action_std_init, torch::nn::Sequential actor_model, torch::nn::Sequential critic_model)
        : action_var(torch::full({ action_dim }, action_std_init* action_std_init).to(torch::Device(torch::kCUDA))) {
        this->action_dim = action_dim;
        this->actor = actor_model;
        this->critic = critic_model;
    }

    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> act(torch::Tensor state, int env_num, float epsilon, float random_value) {
        torch::Tensor action_probs = actor->forward(state);
        //std::cout << "state device: " << state.device() << std::endl;
        //std::cout << "action probs created" << std::endl;
        torch::Tensor action;
        if (random_value > epsilon) {
            action = torch::argmax(action_probs, 1);
        }
        else {
            action = torch::multinomial(action_probs, 1, true);
        }
        //std::cout << "action created" << std::endl;
        torch::Tensor action_logprobs = torch::zeros(env_num, torch::device(torch::kCUDA));
        // std::cout << "action log probs created" << std::endl;
        for (int i = 0; i < action_probs.size(0); i++) {
            action_logprobs[i] = (action_probs[i][action[i].item<int>()] + 1e-8).log();
        }
        //std::cout << "action log probs looped" << std::endl;
        //std::cout << "state device: " << state.device() << std::endl;
        torch::Tensor state_val = critic->forward(state);
        //std::cout << "state val created" << std::endl;

        
        return std::make_tuple(action, action_logprobs, state_val);
    }

};

void write_vector_to_csv(const std::vector<float>& data, const std::string& file_name) {
    std::ofstream output_file(file_name);

    if (!output_file.is_open()) {
        std::cerr << "Error: Could not open the file." << std::endl;
        return;
    }

    for (size_t i = 0; i < data.size(); ++i) {
        output_file << data[i];

        if (i < data.size() - 1) {
            output_file << ",";
        }
    }
    output_file.close();
}

std::vector<torch::Tensor> normalize_tensor(const std::vector<torch::Tensor>& data) {
    std::vector<torch::Tensor> normalized_data = data;
    torch::Tensor mean = torch::mean(torch::stack(data));
    torch::Tensor std = torch::std(torch::stack(data));
    for (torch::Tensor x : normalized_data) {
        x = (x - mean) / (std + 1e-5);
    }
    return normalized_data;
}

//int random_index_selection(int index_start, int num_steps, int vector_size) {
//    int seed = std::chrono::system_clock::now().time_since_epoch().count();
//    srand(seed);
//    int randomIndex = index_start + (rand() % (vector_size - num_steps - 20));
//    std::cout << "random index: " << randomIndex << std::endl;
//    return randomIndex;
//}

int random_index_selection(int index_start, int num_steps, int vector_size, int time_dim) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> distrib(index_start, vector_size - num_steps - time_dim);

    int randomIndex = distrib(gen);
    //std::cout << "random index: " << randomIndex << std::endl;
    return randomIndex;
}

struct GruToLinear : torch::nn::Module {
    GruToLinear() {}

    torch::Tensor forward(std::tuple<at::Tensor, at::Tensor> x) {
        std::get<1>(x) = torch::reshape(std::get<1>(x), { std::get<1>(x).size(1), std::get<1>(x).size(0), std::get<1>(x).size(2) });
        return torch::concat({ std::get<0>(x), std::get<1>(x) }, 1);
    }
};


void concurrent_environments(
    std::vector<torch::Tensor>& action_batch,
    torch::Tensor& return_batch, torch::Tensor& action_logprob_batch,
    std::vector<std::vector<int>>& actions_storage, torch::Tensor& state_value_batch,
    std::vector<std::vector<Environment>>& envs, std::vector<std::vector<int>>& random_index_vec, int time_dim,
    int batch_size, int num_envs, int thread, int num_steps, int end_of_ep_buffer,
    int state_dim, float epsilon, std::vector<ActorCritic> actor_critic_vec, std::vector<int> actions, int num_threads, std::vector<std::vector<float>>& profit_sum_vec,
    std::vector<std::vector<float>>& step_reward_sum_vec, std::vector<std::vector<float>>& price_sum_vec,
    std::vector<std::vector<float>>& action_sum_vec, std::vector<std::vector<float>>& position_sum_vec,
    std::vector<std::vector<float>>& position_penalty_sum_vec, std::vector<std::vector<float>>& weighted_profit_sum_vec,
    std::vector<std::vector<float>>& omega_ratio_sum_vec, std::vector<std::vector<float>>& previous_action_sum_vec, int init_time) {

    //std::cout << "starting initialization of thread" << std::endl;
    torch::Tensor current_state_storage = torch::zeros({ num_envs, time_dim, state_dim });
    //td::cout << "1" << std::endl;
    torch::Tensor returns = torch::zeros({ num_envs, batch_size });
    //std::cout << "2" << std::endl;
    //std::cout << "thread: " << thread << std::endl;
    ActorCritic actor_critic = actor_critic_vec[thread];
    //std::cout << "tensors initialized" << std::endl;
    auto device = torch::Device(torch::kCUDA);
    float profit_sum = 0;
    float step_reward_sum = 0;
    float price_sum = 0;
    float action_sum = 0;
    float position_sum = 0;
    float position_penalty_sum = 0;
    float weighted_profit_sum = 0;
    float omega_ratio_sum = 0;
    float previous_action_sum = 0;
    //std::cout << "rest of variables initialized" << std::endl;
    //std::cout << "initialization of thread finished, starting workload" << std::endl;


    for (int timestep = init_time; timestep < (init_time + batch_size); ++timestep) {
        int tick = timestep - init_time;
        //std::cout << "environment iterator starter" << std::endl;
        for (int i = 0; i < num_envs; ++i) {
            auto& env = envs[thread][i];
            //std::cout << "env acquired" << std::endl;
            bool end_ep_hit = env.step(actions_storage[thread][i], timestep, random_index_vec[thread][i], num_steps, end_of_ep_buffer);
            //std::cout << "step " << timestep << " thread " << thread << " env " << i << "tick: " << tick << std::endl;
            std::vector<float> flattened_state;
            auto state_env = env.get_state_env();
            //std::cout << "state env size: " << state_env.size() << std::endl;
            for (const auto& inner : state_env) {
                flattened_state.insert(flattened_state.end(), inner.begin(), inner.end());
            }
            //std::cout << "flattened state done" << std::endl;
            torch::Tensor flattened_state_tensor = torch::tensor(flattened_state);
            //std::cout << "flattened state tensor done" << std::endl;
            torch::Tensor state_tensor = flattened_state_tensor.reshape({ time_dim, state_dim });
            //std::cout << "state tensor done" << std::endl;
            current_state_storage[i] = state_tensor;
            //std::cout << "current state storage done" << std::endl;
            //std::cout << env.get_step_reward() << std::endl;

            returns[i][tick] = env.get_step_reward();
            
            //std::cout << "returns done" << std::endl;
            if (end_ep_hit) {
                env.timestep_offset_update(end_of_ep_buffer);
                //end_ep_hit_counter += 1;
            }
            end_ep_hit = false;
            //std::cout << "end ep hit done" << std::endl;

        }
        //std::cout << "environment iterator finished" << std::endl;
        //std::cout << "thread " << thread << " env loop finished" << std::endl;

        torch::Tensor current_state = current_state_storage.to(device);
        //std::cout << "current state moved to device" << std::endl;
        //std::cout << "current state sent to device" << std::endl;
        //auto rd_start = std::chrono::high_resolution_clock::now();
        std::random_device rd;
        std::default_random_engine generator(rd());
        std::uniform_real_distribution<float> distribution(0.0, 1.0);

        //std::cout << "random value initialized" << std::endl;
        float random_value = distribution(generator);
        //std::cout << "random value done" << std::endl;
         //std::cout << "tensors initialized" << std::endl;
        //std::cout << "thread " << thread << " current state sent to device " << timestep << " timestep" << std::endl;

        auto [action_tensor, action_logprob_tensor, state_value_tensor] = actor_critic.act(current_state, num_envs, epsilon, random_value);
        //std::cout << "action tensor size: " << action_tensor.sizes() << "state value tensor size: " << state_value_tensor.sizes() << std::endl;
        //std::cout << "thread " << thread << " act done" << timestep << " timestep" << std::endl;
        //std::cout << "action tensor done" << std::endl;
        //action_tensor.squeeze();
        //std::cout << "main computation done, time for array assignment" << std::endl;
        //std::cout << "action tensor squeezed" << std::endl;
        //std::cout << state_value_tensor.sizes();
        for (int j = 0; j < envs.size(); ++j) {
            //std::cout << "action tensor j " << j << std::endl;
            actions_storage[thread][j] = actions[action_tensor[j].item<int>()];
        }
        
        //std::cout << "actions storage done" << std::endl;
        //std::cout << "current_state_storage " << current_state_storage.sizes() << std::endl;
        //std::cout << "state_batch sizes " << state_batch.size() << std::endl;
        //std::cout << "state batch size: " << state_batch.size();
        //std::cout << "state batch done" << std::endl;
        action_batch[thread] = action_tensor.squeeze();
        
        //std::cout << "action batch done" << std::endl;
        //std::cout << returns << std::endl;
        //std::cout << returns.squeeze_().sizes() << std::endl;
        return_batch[thread] = returns.squeeze();
        
        //std::cout << "return batch done" << std::endl;

        action_logprob_batch.index_put_({ thread, torch::indexing::Slice(), tick }, action_logprob_tensor);
        //std::cout << "action logprob batch done" << std::endl;
        state_value_batch.index_put_({ thread, torch::indexing::Slice(), tick }, state_value_tensor.squeeze());
        //state_value_batch[thread] = state_value_tensor;
        //std::cout << "state value batch done" << std::endl;
        for (int i = 0; i < num_envs; ++i) {
            auto& env = envs[thread][i];
            profit_sum += env.get_total_profit();
            step_reward_sum += env.get_step_reward();
            price_sum += env.get_price();
            action_sum += float(actions_storage[thread][i]);
            position_sum += env.get_position();
            position_penalty_sum += env.get_position_penalty();
            weighted_profit_sum += env.get_weighted_profit_history();
            omega_ratio_sum += env.get_omega_ratio();
            previous_action_sum += env.get_previous_action_reward();
        }
        //std::cout << "thread: " << thread << " assignment is starting" << std::endl;
        if ((timestep - time_dim) < num_steps) {
            profit_sum_vec[thread][timestep - time_dim] = profit_sum / num_envs;
            step_reward_sum_vec[thread][timestep - time_dim] = step_reward_sum / num_envs;
            price_sum_vec[thread][timestep - time_dim] = price_sum / num_envs;
            action_sum_vec[thread][timestep - time_dim] = action_sum / num_envs;
            position_sum_vec[thread][timestep - time_dim] = position_sum / num_envs;
            position_penalty_sum_vec[thread][timestep - time_dim] = position_penalty_sum / num_envs;
            weighted_profit_sum_vec[thread][timestep - time_dim] = weighted_profit_sum / num_envs;
            omega_ratio_sum_vec[thread][timestep - time_dim] = omega_ratio_sum / num_envs;
            previous_action_sum_vec[thread][timestep - time_dim] = previous_action_sum / num_envs;
        }
        
        //std::cout << profit_sum / num_envs << std::endl;
        profit_sum = 0;
        step_reward_sum = 0;
        price_sum = 0;
        action_sum = 0;
        position_sum = 0;
        position_penalty_sum = 0;
        weighted_profit_sum = 0;
        omega_ratio_sum = 0;
        previous_action_sum = 0;
        //std::cout << "thread timestep done, tick is: " << tick << std::endl;

    }
}

torch::nn::Sequential create_actor_model(int state_dim) {
    return torch::nn::Sequential(
        torch::nn::GRU(torch::nn::GRUOptions(state_dim, 64).batch_first(true).num_layers(8)),
        GruToLinear(),
        torch::nn::Flatten(),
        torch::nn::Linear(4352, 128),
        torch::nn::ReLU(),
        torch::nn::Linear(128, 128),
        torch::nn::ReLU(),
        torch::nn::Linear(128, 11),
        torch::nn::Softmax(1)
    );
}

torch::nn::Sequential create_critic_model(int state_dim) {
    return torch::nn::Sequential(
        torch::nn::GRU(torch::nn::GRUOptions(state_dim, 64).batch_first(true).num_layers(8)),
        GruToLinear(),
        torch::nn::Flatten(),
        torch::nn::Linear(4352, 128),
        torch::nn::ReLU(),
        torch::nn::Linear(128, 128),
        torch::nn::ReLU(),
        torch::nn::Linear(128, 1)
    );
}

void clone_model(torch::nn::Sequential master_model, torch::nn::Sequential model_clone) {
    torch::autograd::GradMode::set_enabled(false);
    auto params = model_clone->named_parameters(true);
    auto buffers = model_clone->named_buffers(true);
    for (auto& val : master_model->named_parameters()) {
        auto name = val.key();
        auto* t = params.find(name);
        if (t != nullptr) {
            t->copy_(val.value());
        }
        else {
            t = buffers.find(name);
            if (t != nullptr) {
                t->copy_(val.value());
            }
        }
    }
    torch::autograd::GradMode::set_enabled(true);
}

int main() {
    //torch::autograd::AnomalyMode::set_enabled(true);
    //torch::autograd::GradMode::set_enabled(true);
    auto device = torch::Device(torch::kCUDA);
    int num_envs = 32;
    int time_dim = 60;
    int state_dim = 7;
    int action_dim = 11;
    float learning_rate = 0.0001;
    int num_episodes = 10000;

    float epsilon = 1.;
    float epsilon_decay = .0005;

    int end_of_ep_buffer = 64;

    float gamma = .95;

    int num_steps = 512;
    int batch_size = 48;
    int vec_dimensions = num_episodes * num_steps;
    int index_start = 0;
    int action_scaling = 1;

    int num_threads = 8;

    if (torch::cuda::is_available()) {
        device = torch::Device(torch::kCUDA);
        std::cout << "Device set to: " << device << std::endl;
    }
    else {
        std::cout << "Device set to: CPU" << std::endl;
    }

    std::vector<int> actions{ -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5 };

    for (int i = 0; i < actions.size(); i++) {
        actions[i] = actions[i] * action_scaling;
    }


    ThreadPool pool;
    pool.SetThreadCount(num_threads);
    pool.Start();


    std::string price_str = "C:/Users/quinn/PycharmProjects/reinforcement_learning/mid_prices.csv";
    std::string predictions_str = "C:/Users/quinn/PycharmProjects/reinforcement_learning/predictions.csv";
    std::string length_list_str = "C:/Users/quinn/PycharmProjects/reinforcement_learning/length_list.csv";
    std::vector<std::vector<float>> price_vec = load_csv(price_str);
    std::vector<std::vector<float>> predictions_vec = load_csv(predictions_str);
    std::vector<std::vector<float>> length_list_vec = load_csv(length_list_str);
    std::vector<float> price_data = price_vec[0];
    std::vector<float> predictions_data = predictions_vec[0];
    int vector_size = price_data.size();
    //std::cout << "std::vector size: " << vector_size << std::endl;

    std::vector<float> actor_losses;
    std::vector<float> profits;
    std::vector<float> positions;
    std::vector<float> actions_taken;
    std::vector<float> prices;
    std::vector<float> rewards;
    std::vector<float> critic_losses;
    std::vector<float> previous_action_rewards;

    critic_losses.resize(vec_dimensions);
    actor_losses.resize(vec_dimensions);
    profits.resize(vec_dimensions);
    positions.resize(vec_dimensions);
    actions_taken.resize(vec_dimensions);
    prices.resize(vec_dimensions);
    rewards.resize(vec_dimensions);
    previous_action_rewards.resize(vec_dimensions);

    std::vector<int> length_list(length_list_vec[0].size());

    for (int i = 0; i < length_list_vec[0].size(); ++i) {
        length_list[i] = int(length_list_vec[0][i]);
    }

    std::vector<std::vector<Environment>> envs(num_threads, std::vector<Environment>(num_envs));
    //envs.reserve(num_envs    
    for (int i = 0; i < num_threads; ++i) {
        for (int j = 0; j < num_envs; ++j) {
            int random_index = random_index_selection(index_start, num_steps, vector_size, time_dim);
            envs[i][j] = Environment(price_data, predictions_data, length_list, random_index, num_steps, end_of_ep_buffer, gamma, time_dim);
        }
    }

    bool has_continuous_action_space = false;
    float action_std_init = 1.0;

    torch::nn::Sequential master_actor_model = create_actor_model(state_dim);
    torch::nn::Sequential master_critic_model = create_critic_model(state_dim);

    auto optimizer_actor = torch::optim::Adam(master_actor_model->parameters(), torch::optim::AdamOptions(learning_rate));
    auto optimizer_critic = torch::optim::Adam(master_critic_model->parameters(), torch::optim::AdamOptions(learning_rate));

    auto actor_lr_decay = torch::optim::StepLR(optimizer_actor, 500, 0.5);
    auto critic_lr_decay = torch::optim::StepLR(optimizer_critic, 500, 0.5);

    torch::Tensor actor_loss;
    torch::Tensor critic_loss;
    float actor_loss_float = 0;
    float critic_loss_float = 0;
    //std::vector<torch::Tensor> returns;
    std::vector<std::vector<int>> actions_storage(num_threads, std::vector<int>(num_envs));

    //std::cout << "vectors initialized" << std::endl;
    //returns.reserve(num_envs);
    //std::cout << "initialization done" << std::endl;
    for (int episode = 0; episode < num_episodes; ++episode) {
        std::vector<std::vector<int>> random_index_vec(num_threads, std::vector<int>(num_envs));
        auto episode_start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < num_threads; ++i) {
            for (int j = 0; j < num_envs; ++j) {
                auto& env = envs[i][j];
                int random_index = random_index_selection(index_start, num_steps, vector_size, time_dim);
                random_index_vec[i][j] = random_index;
                env.reset(price_data, predictions_data, length_list, random_index, num_steps);
            }
        }

        float profit_sum = 0, step_reward_sum = 0, price_sum = 0, action_sum = 0, position_sum = 0, total_reward_sum = 0,
            position_penalty_sum = 0, weighted_profit_sum = 0, total_profit_sum = 0, omega_ratio_sum = 0, total_profit_reward_sum = 0, previous_action_sum = 0;

        //std::vector<torch::Tensor> state_batch(num_threads);
        //torch::Tensor state_batch = torch::zeros({num_threads, num_envs, batch_size, time_dim, state_dim }, device);
        
        for (auto& element : actions_storage) {
            for (auto& element2 : element) {
                element2 = 0;
            }
        }
        epsilon -= epsilon_decay;
        //int end_ep_hit_counter = 0;
        int current_offset = 0;

        std::vector<std::vector<float>> profit_sum_vec(num_threads, std::vector<float>(num_steps));
        std::vector<std::vector<float>> step_reward_sum_vec(num_threads, std::vector<float>(num_steps));
        std::vector<std::vector<float>> price_sum_vec(num_threads, std::vector<float>(num_steps));
        std::vector<std::vector<float>> action_sum_vec(num_threads, std::vector<float>(num_steps));
        std::vector<std::vector<float>> position_sum_vec(num_threads, std::vector<float>(num_steps));
        std::vector<std::vector<float>> position_penalty_sum_vec(num_threads, std::vector<float>(num_steps));
        std::vector<std::vector<float>> weighted_profit_sum_vec(num_threads, std::vector<float>(num_steps));
        std::vector<std::vector<float>> omega_ratio_sum_vec(num_threads, std::vector<float>(num_steps));
        std::vector<std::vector<float>> previous_action_sum_vec(num_threads, std::vector<float>(num_steps));

        for (int timestep = time_dim; timestep < time_dim + num_steps; timestep += batch_size) {
            //std::cout << "timestep: " << timestep << std::endl;
            //int thread_num = 0;
            std::vector<ActorCritic> actor_critic_vec;
            for (int i = 0; i < num_threads; ++i) {
                torch::nn::Sequential actor_model = create_actor_model(state_dim);
                torch::nn::Sequential critic_model = create_critic_model(state_dim);
                clone_model(master_actor_model, actor_model);
                clone_model(master_critic_model, critic_model);
                actor_model->to(device);
                critic_model->to(device);
                actor_critic_vec.push_back(ActorCritic(state_dim, action_dim, has_continuous_action_space, action_std_init, actor_model, critic_model));
            }
            //std::cout << "models cloned" << std::endl;
            for (ActorCritic actorcritic : actor_critic_vec) {
                actorcritic.to(device);
            }
            //std::cout << "a/c sent to device" << std::endl;
            torch::Tensor return_batch = torch::zeros({ num_threads, num_envs, batch_size });
            std::vector<torch::Tensor> action_batch(num_threads);
            torch::Tensor action_logprob_batch = torch::zeros({ num_threads, num_envs, batch_size }, device);
            torch::Tensor state_value_batch = torch::zeros({ num_threads, num_envs, batch_size }, device);
            torch::Tensor current_state_storage = torch::zeros({ num_envs, time_dim, state_dim });
            torch::Tensor returns = torch::zeros(num_envs);

            for (int i = 0; i < num_threads; ++i) {
                pool.QueueJob([i, &action_batch, &return_batch, &action_logprob_batch,
                    &actions_storage, &state_value_batch, &envs, &random_index_vec,
                    &time_dim, &batch_size, &num_envs, &num_steps, &end_of_ep_buffer,
                    &state_dim, &epsilon, &actor_critic_vec, &actions, &num_threads,
                    &profit_sum_vec, &step_reward_sum_vec, &price_sum_vec, &action_sum_vec,
                    &position_sum_vec, &position_penalty_sum_vec, &weighted_profit_sum_vec,
                    &omega_ratio_sum_vec, &previous_action_sum_vec, &timestep]
                    {
                        concurrent_environments(action_batch, return_batch,
                        action_logprob_batch, actions_storage, state_value_batch, envs,
                        random_index_vec, time_dim, batch_size, num_envs, i, num_steps,
                        end_of_ep_buffer, state_dim, epsilon, actor_critic_vec, actions,
                        num_threads, profit_sum_vec, step_reward_sum_vec, price_sum_vec,
                        action_sum_vec, position_sum_vec, position_penalty_sum_vec,
                        weighted_profit_sum_vec, omega_ratio_sum_vec, previous_action_sum_vec,
                        timestep);
                    });
                //std::cout << "thread " << i << " queued" << std::endl;
            }

            while (pool.busy()) {
                //std::cout << "pool busy" << std::endl;
                std::this_thread::sleep_for(std::chrono::microseconds(100));
            }
            
            //std::cout << "threads done" << std::endl;
            //std::cout << "threads joined, timestep is: " << timestep << std::endl;
            try {
                optimizer_actor.zero_grad();
                //std::cout << "optimizer zeroed" << std::endl;
                //std::cout << "state batch size: " << state_batch.size() << std::endl;
                //torch::Tensor states_t;
                //try {
                    //torch::Tensor states_t = torch::stack(state_batch);

                //}
                //catch (std::exception& e) {
                //	std::cout << e.what() << std::endl;
                //}


                //std::cout << "states stacked" << std::endl;
                //orch::Tensor returns_t;
                //try {
                return_batch = return_batch.to(device).squeeze();
                //std::cout << "returns stacked" << std::endl;
                //std::cout << "action logprobs stacked" << std::endl;
                state_value_batch.squeeze_();
                //std::cout << "state values stacked" << std::endl;
                torch::Tensor advantages = torch::sub(return_batch, state_value_batch);
                //std::cout << "advantages calculated" << std::endl;
                //std::cout << "advantages: " << advantages << std::endl;
                //std::cout << "action logprob batch: " << action_logprob_batch << std::endl;
                std::cout << "action logprob batch sizes: " << action_logprob_batch.sizes() << std::endl;
                std::cout << "advantages sizes: " << advantages.sizes() << std::endl;

                torch::Tensor actor_loss = torch::mean(torch::neg_(action_logprob_batch) * advantages.detach());
                //std::cout << "actor loss: " << actor_loss << std::endl;
                actor_loss.backward();
                //std::cout << "actor loss backwarded" << std::endl;
                optimizer_actor.step();
                //std::cout << "actor optimizer stepped" << std::endl;
                optimizer_critic.zero_grad();
                //std::cout << "critic optimizer zeroed" << std::endl;
                //std::cout << "state value batch: " << state_value_batch.sizes() << std::endl;
                //std::cout << "return batch: " << return_batch.sizes() << std::endl;
                torch::Tensor critic_loss = torch::mse_loss(state_value_batch, return_batch);
                //std::cout << "critic loss calculated" << std::endl;
                critic_loss.backward();
                //std::cout << "critic loss backwarded" << std::endl;
                optimizer_critic.step();
                //std::cout << "critic optimizer stepped" << std::endl;
                //state_batch.clear();
                //action_batch.clear();
                //return_batch.clear();
                //action_logprob_batch.clear();
                //state_value_batch.clear();
                //std::cout << "values cleared" << std::endl;
                actor_loss_float = actor_loss.item<float>();
                critic_loss_float = critic_loss.item<float>();
                //std::cout << "actor loss: " << actor_loss_float << std::endl;
                //std::cout << "critic loss: " << critic_loss_float << std::endl;

            }
            catch (std::exception& e) {
				std::cout << e.what() << std::endl;
			}


        }
        //std::cout << profit_sum_vec.size() << profit_sum_vec[0].size() << std::endl;
        for (int j = 0; j < num_steps; ++j) {
            for (int i = 0; i < num_threads; ++i) {
                //std::cout << "i: " << i << " j: " << j << std::endl;
                profit_sum += profit_sum_vec[i][j];
                step_reward_sum += step_reward_sum_vec[i][j];
                price_sum += price_sum_vec[i][j];
                action_sum += action_sum_vec[i][j];
                position_sum += position_sum_vec[i][j];
                position_penalty_sum += position_penalty_sum_vec[i][j];
                weighted_profit_sum += weighted_profit_sum_vec[i][j];
                omega_ratio_sum += omega_ratio_sum_vec[i][j];
                previous_action_sum += previous_action_sum_vec[i][j];
                //std::cout << "i: " << i << " j: " << j << std::endl;
            }
            profits[j + (episode * num_steps)] = profit_sum / num_threads;
            rewards[j + (episode * num_steps)] = step_reward_sum / num_threads;
            positions[j + (episode * num_steps)] = position_sum / num_threads;
            actions_taken[j + (episode * num_steps)] = action_sum / num_threads;
            prices[j + (episode * num_steps)] = price_sum / num_threads;
            actor_losses[j + (episode * num_steps)] = actor_loss_float / num_threads;
            critic_losses[j + (episode * num_steps)] = critic_loss_float / num_threads;
            previous_action_rewards[j + (episode * num_steps)] = previous_action_sum / num_threads;

            step_reward_sum = 0;
            price_sum = 0;
            action_sum = 0;
            position_sum = 0;
            position_penalty_sum = 0;
            omega_ratio_sum = 0;
            profit_sum = 0;
            previous_action_sum = 0;
        }
        //std::cout << "loop finished" << std::endl;
        total_profit_sum = 0;
        for (int i = episode * num_steps; i < (episode * num_steps) + num_steps; ++i) {
            total_profit_sum += profits[i];
        }
        total_profit_sum = total_profit_sum / num_steps;

        total_reward_sum = 0;
        for (int i = episode * num_steps; i < (episode * num_steps) + num_steps; ++i) {
            total_reward_sum += rewards[i];
        }
        total_reward_sum = total_reward_sum / num_steps;

        float total_position_sum = 0;
        for (int i = episode * num_steps; i < (episode * num_steps) + num_steps; ++i) {
            total_position_sum += positions[i];
        }
        total_position_sum = total_position_sum / num_steps;

        float total_action_taken_sum = 0;
        for (int i = episode * num_steps; i < (episode * num_steps) + num_steps; ++i) {
            total_action_taken_sum += actions_taken[i];
        }
        total_action_taken_sum = total_action_taken_sum / num_steps;

        float total_price_sum = 0;
        for (int i = episode * num_steps; i < (episode * num_steps) + num_steps; ++i) {
            total_price_sum += prices[i];
        }
        total_price_sum = total_price_sum / num_steps;

        float total_actor_loss_sum = 0;
        for (int i = episode * num_steps; i < (episode * num_steps) + num_steps; ++i) {
            total_actor_loss_sum += actor_losses[i];
        }
        total_actor_loss_sum = total_actor_loss_sum / num_steps;

        float total_critic_loss_sum = 0;
        for (int i = episode * num_steps; i < (episode * num_steps) + num_steps; ++i) {
            total_critic_loss_sum += critic_losses[i];
        }
        total_critic_loss_sum = total_critic_loss_sum / num_steps;

        //need to print out the average action distribution across episodes
        auto episode_end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(episode_end - episode_start);

        actor_lr_decay.step();
        critic_lr_decay.step();

        std::cout << "episode: " << episode << std::endl;
        std::cout << std::endl;
        std::cout << "average rewards: " << (total_reward_sum / num_steps) << std::endl;
        //std::cout << "average sharpe reward: " << env.sharpe_history / num_steps << std::endl;
        std::cout << "average position penalty: " << (position_penalty_sum / num_steps) << std::endl;
        std::cout << "average weighted profit reward: " << (weighted_profit_sum / num_steps) << std::endl;
        std::cout << "average total profit reward: " << (total_profit_reward_sum / num_steps) << std::endl;
        //std::cout << "average omega reward: " << (omega_ratio_sum / num_steps) / num_envs << std::endl;
        std::cout << "average previous action reward: " << (previous_action_sum / num_steps) << std::endl;
        std::cout << "total rewards: " << total_reward_sum << std::endl;

        std::cout << std::endl;
        //std::cout << "total rewards: " << accumulate(rewards.begin(), rewards.end(), 0.0) << std::endl;
        //for (int i = (num_steps * episode); i < (num_steps * episode) + num_steps; ++i) {
        //    position_sum += positions[i]/num_envs;
        //}
        for (int i = (num_steps * episode); i < (num_steps * episode) + num_steps; ++i) {
            action_sum += actions_taken[i];
            //std::cout << actions_taken[i] << std::endl;
        }
        std::cout << "average actions: " << action_sum / num_steps << std::endl;
        std::cout << "average positions: " << total_position_sum / num_steps << std::endl;

        std::cout << std::endl;
        std::cout << "average profit: " << (total_profit_sum / num_steps) << std::endl;
        //std::cout << "average profits: " << accumulate(profits.begin(), profits.end(), 0.0) / profits.size() << std::endl;
        std::cout << "total profit: " << total_profit_sum << std::endl;
        std::cout << std::endl;
        std::cout << "actor loss: " << actor_loss_float << std::endl;
        std::cout << "critic loss: " << critic_loss_float << std::endl;
        std::cout << "epsilon: " << epsilon << std::endl;
        //std::cout << "learning rate for actor/critic: " << optimizer_actor.parameters() << std::endl;
        std::cout << std::endl;
        std::cout << "elapsed time in seconds: " << duration.count() / 1000.0 << std::endl;
        std::cout << "=============================================================================" << std::endl;

        std::vector<float> previous_action_rewards_save;
        for (int k = 0; k < ((episode + 1) * num_steps); ++k) {
            previous_action_rewards_save.push_back(previous_action_rewards[k]);
        }
        write_vector_to_csv(previous_action_rewards_save, "C:/Users/quinn/PycharmProjects/reinforcement_learning/previous_action_rewards.csv");

        std::vector<float> profit_save;
        for (int k = 0; k < ((episode + 1) * num_steps); ++k) {
            profit_save.push_back(profits[k]);
        }
        write_vector_to_csv(profit_save, "C:/Users/quinn/PycharmProjects/reinforcement_learning/profits.csv");

        std::vector<float> actor_loss_save;
        for (int k = 0; k < ((episode + 1) * num_steps); ++k) {
            actor_loss_save.push_back(actor_losses[k]);
        }
        write_vector_to_csv(actor_loss_save, "C:/Users/quinn/PycharmProjects/reinforcement_learning/actor_losses.csv");

        std::vector<float> critic_loss_save;
        for (int k = 0; k < ((episode + 1) * num_steps); ++k) {
            critic_loss_save.push_back(critic_losses[k]);
        }
        write_vector_to_csv(critic_loss_save, "C:/Users/quinn/PycharmProjects/reinforcement_learning/critic_losses.csv");

        std::vector<float> reward_save;
        for (int k = 0; k < ((episode + 1) * num_steps); ++k) {
            reward_save.push_back(rewards[k]);
        }
        write_vector_to_csv(reward_save, "C:/Users/quinn/PycharmProjects/reinforcement_learning/rewards.csv");

        std::vector<float> position_save;
        for (int k = 0; k < ((episode + 1) * num_steps); ++k) {
            position_save.push_back(positions[k]);
        }
        write_vector_to_csv(position_save, "C:/Users/quinn/PycharmProjects/reinforcement_learning/positions.csv");

        std::vector<float> action_save;
        for (int k = 0; k < ((episode + 1) * num_steps); ++k) {
            action_save.push_back(actions_taken[k]);
        }
        write_vector_to_csv(action_save, "C:/Users/quinn/PycharmProjects/reinforcement_learning/actions.csv");

        std::vector<float> price_save;
        for (int k = 0; k < ((episode + 1) * num_steps); ++k) {
            price_save.push_back(prices[k]);
        }
        write_vector_to_csv(price_save, "C:/Users/quinn/PycharmProjects/reinforcement_learning/prices.csv");
        float rounded_profit = round(total_profit_sum * 100000) / 100000;

        std::string actor_model_string = std::format("C:/Users/quinn/PycharmProjects/reinforcement_learning/trained_actor_models/actor_model_{0}_profit_{1}_episode_{2}_lr_{3}_batch_{4}_steps_threaded.pt", rounded_profit, episode, learning_rate, batch_size, num_steps);
        std::string critic_model_string = std::format("C:/Users/quinn/PycharmProjects/reinforcement_learning/trained_critic_models/critic_model_{0}_profit_{1}_episode_{2}_lr_{3}_batch_{4}_steps_threaded.pt", rounded_profit, episode, learning_rate, batch_size, num_steps);

        if (total_profit_sum > 1) {
            torch::save(master_actor_model, actor_model_string);
            torch::save(master_critic_model, critic_model_string);
        }
        
    }
    pool.Stop();

    return 0;
}