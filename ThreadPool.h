#pragma once

#include <functional>
#include <thread>
#include <vector>
#include <queue>
#include <mutex>

class ThreadPool {
public:
    void Start();
    void QueueJob(const std::function<void()>& job);
    void Stop();
    bool busy();
    void SetThreadCount(int thread_count);

private:
    void ThreadLoop();
    int num_threads;
    bool should_terminate = false;           // Tells threads to stop looking for jobs
    std::mutex queue_mutex;                  // Prevents data races to the job queue
    std::condition_variable mutex_condition; // Allows threads to wait on new jobs or termination 
    std::vector<std::thread> threads;
    std::queue<std::function<void()>> jobs;
    std::atomic<int> active_threads{0};

};