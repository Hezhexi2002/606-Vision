//#include <pthread.h>
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
//#include <Windows.h>
//#include <string>
//
//using namespace std;
//
//
//bool operator==(pthread_t x,pthread_t y)
//{
//    if (x == y)
//        return TRUE;
//    else
//        return FALSE;
//}
//
//
//// 定义任务结构体
//using callback = void(*)(void*);
//struct Task
//{
//    Task()
//    {
//        function = nullptr;
//        arg = nullptr;
//    }
//    Task(callback f, void* arg)
//    {
//        function = f;
//        this->arg = arg;
//    }
//    callback function;
//    void* arg;
//};
//
//// 任务队列
//class TaskQueue
//{
//public:
//    TaskQueue();
//    ~TaskQueue();
//
//    // 添加任务
//    void addTask(Task& task);
//    void addTask(callback func, void* arg);
//
//    // 取出一个任务
//    Task takeTask();
//
//    // 获取当前队列中任务个数
//    inline int taskNumber()
//    {
//        return m_queue.size();
//    }
//
//private:
//    pthread_mutex_t m_mutex;    // 互斥锁
//    std::queue<Task> m_queue;   // 任务队列
//};
//
///****************************************************************************************/
//// 线程池
//class ThreadPool
//{
//public:
//    ThreadPool(int min, int max);
//    ~ThreadPool();
//
//    // 添加任务
//    void addTask(Task task);
//    // 获取忙线程的个数
//    int getBusyNumber();
//    // 获取活着的线程个数
//    int getAliveNumber();
//
//private:
//    // 工作的线程的任务函数
//    static void* worker(void* arg);
//    // 管理者线程的任务函数
//    static void* manager(void* arg);
//    void threadExit();
//
//private:
//    pthread_mutex_t m_lock;
//    pthread_cond_t m_notEmpty;
//    pthread_t* m_threadIDs;
//    pthread_t m_managerID;
//    TaskQueue* m_taskQ;
//    int m_minNum;
//    int m_maxNum;
//    int m_busyNum;
//    int m_aliveNum;
//    int m_exitNum;
//    bool m_shutdown = false;
//};
//
//
///****************************************************************************************/
////任务队列的定义
//TaskQueue::TaskQueue()
//{
//    pthread_mutex_init(&m_mutex, NULL);
//}
//
//TaskQueue::~TaskQueue()
//{
//    pthread_mutex_destroy(&m_mutex);
//}
//
//void TaskQueue::addTask(Task& task)
//{
//    pthread_mutex_lock(&m_mutex);
//    m_queue.push(task);
//    pthread_mutex_unlock(&m_mutex);
//}
//
//void TaskQueue::addTask(callback func, void* arg)
//{
//    pthread_mutex_lock(&m_mutex);
//    Task task;
//    task.function = func;
//    task.arg = arg;
//    m_queue.push(task);
//    pthread_mutex_unlock(&m_mutex);
//}
//
//Task TaskQueue::takeTask()
//{
//    Task t;
//    pthread_mutex_lock(&m_mutex);
//    if (m_queue.size() > 0)
//    {
//        t = m_queue.front();
//        m_queue.pop();
//    }
//    pthread_mutex_unlock(&m_mutex);
//    return t;
//}
//
///****************************************************************************************/
//
//ThreadPool::ThreadPool(int minNum, int maxNum)
//{
//    // 实例化任务队列
//    m_taskQ = new TaskQueue;
//    do {
//        // 初始化线程池
//        m_minNum = minNum;
//        m_maxNum = maxNum;
//        m_busyNum = 0;
//        m_aliveNum = minNum;
//
//        // 根据线程的最大上限给线程数组分配内存
//        m_threadIDs = new pthread_t[maxNum];
//        if (m_threadIDs == nullptr)
//        {
//            cout << "分配 thread_t 内存失败...." << endl;;
//            break;
//        }
//        // 初始化
//        memset(m_threadIDs, 0, sizeof(pthread_t) * maxNum);
//        // 初始化互斥锁,条件变量
//        if (pthread_mutex_init(&m_lock, NULL) != 0 ||
//            pthread_cond_init(&m_notEmpty, NULL) != 0)
//        {
//            cout << "init mutex or condition fail..." << endl;
//            break;
//        }
//
//        /////////////////// 创建线程 //////////////////
//        // 根据最小线程个数, 创建线程
//        for (int i = 0; i < minNum; ++i)
//        {
//            pthread_create(&m_threadIDs[i], NULL, worker, this);
//            cout << "创建子线程, ID: " << to_string(m_threadIDs[i]) << endl;
//        }
//        // 创建管理者线程, 1个
//        pthread_create(&m_managerID, NULL, manager, this);
//    } while (0);
//}
//
//ThreadPool::~ThreadPool()
//{
//    m_shutdown = 1;
//    // 销毁管理者线程
//    pthread_join(m_managerID, NULL);
//    // 唤醒所有消费者线程
//    for (int i = 0; i < m_aliveNum; ++i)
//    {
//        pthread_cond_signal(&m_notEmpty);
//    }
//
//    if (m_taskQ) delete m_taskQ;
//    if (m_threadIDs) delete[]m_threadIDs;
//    pthread_mutex_destroy(&m_lock);
//    pthread_cond_destroy(&m_notEmpty);
//}
//
//void ThreadPool::addTask(Task task)
//{
//    if (m_shutdown)
//    {
//        return;
//    }
//    // 添加任务，不需要加锁，任务队列中有锁
//    m_taskQ->addTask(task);
//    // 唤醒工作的线程
//    pthread_cond_signal(&m_notEmpty);
//}
//
//int ThreadPool::getAliveNumber()
//{
//    int threadNum = 0;
//    pthread_mutex_lock(&m_lock);
//    threadNum = m_aliveNum;
//    pthread_mutex_unlock(&m_lock);
//    return threadNum;
//}
//
//int ThreadPool::getBusyNumber()
//{
//    int busyNum = 0;
//    pthread_mutex_lock(&m_lock);
//    busyNum = m_busyNum;
//    pthread_mutex_unlock(&m_lock);
//    return busyNum;
//}
//
//
//// 工作线程任务函数
//void* ThreadPool::worker(void* arg)
//{
//    ThreadPool* pool = static_cast<ThreadPool*>(arg);
//    // 一直不停的工作
//    while (true)
//    {
//        // 访问任务队列(共享资源)加锁
//        pthread_mutex_lock(&pool->m_lock);
//        // 判断任务队列是否为空, 如果为空工作线程阻塞
//        while (pool->m_taskQ->taskNumber() == 0 && !pool->m_shutdown)
//        {
//            printf(" Thread %ld wating...\n", pthread_self());
//            // 阻塞线程
//            pthread_cond_wait(&pool->m_notEmpty, &pool->m_lock);
//
//            // 解除阻塞之后, 判断是否要销毁线程
//            if (pool->m_exitNum > 0)
//            {
//                pool->m_exitNum--;
//                if (pool->m_aliveNum > pool->m_minNum)
//                {
//                    pool->m_aliveNum--;
//                    pthread_mutex_unlock(&pool->m_lock);
//                    pool->threadExit();
//                }
//            }
//        }
//        // 判断线程池是否被关闭了
//        if (pool->m_shutdown)
//        {
//            pthread_mutex_unlock(&pool->m_lock);
//            pool->threadExit();
//        }
//
//        // 从任务队列中取出一个任务
//        Task task = pool->m_taskQ->takeTask();
//        // 工作的线程+1
//        pool->m_busyNum++;
//        // 线程池解锁
//        pthread_mutex_unlock(&pool->m_lock);
//        // 执行任务
//        cout << "thread " << to_string(pthread_self()) << " start working..." << endl;
//        task.function(task.arg);
//        delete task.arg;
//        task.arg = nullptr;
//
//        // 任务处理结束
//        cout << "thread " << to_string(pthread_self()) << " end working...";
//        pthread_mutex_lock(&pool->m_lock);
//        pool->m_busyNum--;
//        pthread_mutex_unlock(&pool->m_lock);
//    }
//
//    return nullptr;
//}
//
//
//// 管理者线程任务函数
//void* ThreadPool::manager(void* arg)
//{
//    ThreadPool* pool = static_cast<ThreadPool*>(arg);
//    // 如果线程池没有关闭, 就一直检测
//    while (!pool->m_shutdown)
//    {
//        // 每隔5s检测一次
//        Sleep(5);
//        // 取出线程池中的任务数和线程数量
//        //  取出工作的线程池数量
//        pthread_mutex_lock(&pool->m_lock);
//        int queueSize = pool->m_taskQ->taskNumber();
//        int liveNum = pool->m_aliveNum;
//        int busyNum = pool->m_busyNum;
//        pthread_mutex_unlock(&pool->m_lock);
//
//        // 创建线程
//        const int NUMBER = 2;
//        // 当前任务个数>存活的线程数 && 存活的线程数<最大线程个数
//        if (queueSize > liveNum && liveNum < pool->m_maxNum)
//        {
//            // 线程池加锁
//            pthread_mutex_lock(&pool->m_lock);
//            int num = 0;
//            for (int i = 0; i < pool->m_maxNum && num < NUMBER
//                && pool->m_aliveNum < pool->m_maxNum; ++i)
//            {
//                if (pool->m_threadIDs[i] == 0)
//                {
//                    pthread_create(&pool->m_threadIDs[i], NULL, worker, pool);
//                    num++;
//                    pool->m_aliveNum++;
//                }
//            }
//            pthread_mutex_unlock(&pool->m_lock);
//        }
//
//        // 销毁多余的线程
//        // 忙线程*2 < 存活的线程数目 && 存活的线程数 > 最小线程数量
//        if (busyNum * 2 < liveNum && liveNum > pool->m_minNum)
//        {
//            pthread_mutex_lock(&pool->m_lock);
//            pool->m_exitNum = NUMBER;
//            pthread_mutex_unlock(&pool->m_lock);
//            for (int i = 0; i < NUMBER; ++i)
//            {
//                pthread_cond_signal(&pool->m_notEmpty);
//            }
//        }
//    }
//    return nullptr;
//}
//
//// 线程退出
//void ThreadPool::threadExit()
//{
//    pthread_t tid = pthread_self();
//    for (int i = 0; i < m_maxNum; ++i)
//    {
//        if (m_threadIDs[i] == tid)
//        {
//            cout << "threadExit() function: thread "
//                << to_string(pthread_self()) << " exiting..." << endl;
//            m_threadIDs[i] = 0;
//            break;
//        }
//    }
//    pthread_exit(NULL);
//}
//
//
