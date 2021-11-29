//#ifndef THREAD_POOL_H
//#define THREAD_POOL_H
//
//#include <iostream>
//#include <vector>
//#include <queue>
//#include <memory>
//#include <thread>
//#include <mutex>
//#include <condition_variable>
//#include <future>
//#include <functional>
//#include <stdexcept>
//
//using namespace std;
//
//class ThreadPool {
//
//public:
//    ThreadPool(size_t);                          //构造函数
//    template<class F, class... Args>             //类模板
//    auto enqueue(F&& f, Args&&... args)->std::future<typename std::result_of<F(Args...)>::type>; //设置任务
//    void Delete();                              //析构函数
//
//private:
//    std::vector< std::thread > workers;            //线程队列，每个元素为一个Thread对象
//    std::queue< std::function<void()> > tasks;     //任务队列，每个元素为一个函数对象    
//
//    std::mutex queue_mutex;                        //互斥量
//    std::condition_variable condition;             //条件变量
//    bool stop;                                     //停止
//};
//
//// 构造函数，把线程插入线程队列，插入时调用embrace_back()，用匿名函数lambda初始化Thread对象
//inline ThreadPool::ThreadPool(size_t threads) : stop(false) {
//
//    for (size_t i = 0; i < threads; ++i)
//        workers.emplace_back(
//            [this]
//            {
//                while(1)
//                {
//                    
//                    std::function<void()> task;
//                    {
//                        
//                        std::unique_lock<std::mutex> mutex(this->queue_mutex);
//
//                        //若返回false时才阻塞线程，阻塞时自动释放锁
//                        //若返回true且受到通知时解阻塞，然后加锁
//                        this->condition.wait(mutex, [this] { return this->stop || !this->tasks.empty(); });
//
//                        if (this->stop && this->tasks.empty())                                               
//                            return;  //用以中断循环
//                        
//                        //从任务队列取出一个任务
//                        task = std::move(this->tasks.front());
//                        this->tasks.pop();
//                    }                            
//                    task();                      // 执行这个任务
//                }
//            }
//            );
//}
//
//// 添加新的任务
//template<class F, class... Args>
//auto ThreadPool::enqueue(F&& f, Args&&... args) -> std::future<typename std::result_of<F(Args...)>::type>
//{
//    // 获取函数返回值类型        
//    using return_type = typename std::result_of<F(Args...)>::type;
//
//    // 创建一个指向任务的智能指针
//    auto task = std::make_shared< std::packaged_task<return_type()> >(
//        std::bind(std::forward<F>(f), std::forward<Args>(args)...)
//        );
//
//    std::future<return_type> res = task->get_future();
//    {
//        std::unique_lock<std::mutex> lock(queue_mutex);  
//        if (stop)
//            throw std::runtime_error("enqueue on stopped ThreadPool");
//
//        tasks.emplace([task]() { (*task)(); });          //把任务加入队列
//    }                                                   //自动解锁
//    condition.notify_one();                             //通知条件变量，唤醒一个线程
//    return res;
//}
//
//// 删除所有线程
//inline void ThreadPool::Delete()
//{
//    {
//        std::unique_lock<std::mutex> lock(queue_mutex);
//        stop = true;
//    }
//    condition.notify_all();
//    for (std::thread& worker : workers)
//        worker.join();
//    
//    cout << "All the threads have been deleted" << endl;
//}
//
//#endif
