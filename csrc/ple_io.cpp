// ple_io.cpp -- the host row reader of the PLE n-gram table (docs/history/qwen4exp_plan.md section 5.5): a pool of threads
// reads n rows of 160 bytes at given byte offsets of an open file into a host buffer. Each decode cycle needs 16 rows per
// token, at random offsets of a 51.2 GB table on disk, so the reads must run in parallel to reach the IOPS of the drive.
//
//   plain   pread(fd, out row, 160, offset): the kernel's page cache serves hot rows (each miss reads at least a page).
//   direct  the fd was opened with O_DIRECT: pread of the 4 KiB-aligned page(s) that hold the row into an aligned buffer
//           of the thread, then a copy of the 160 bytes. The page cache keeps nothing.
//
// The pool is created at the first call, keeps its threads, and grows to the largest thread count that a call asks for.
// A call wakes only the workers it asks for (threads - 1, at most one per row beyond the first): a wakeup costs ~10 us,
// so a small batch from the page cache wants few threads, and cold rows from the drive want many. The calling thread
// works too, and returns when all rows are read. The bound function releases the GIL. Python: ops().ple_gather (bound in
// qsa_bind.cpp). For plain reads, the caller sets posix_fadvise(fd, POSIX_FADV_RANDOM): readahead would read 128 KB per
// row. Tests: tests/test_ple_io.py.
#include <errno.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace {

constexpr int64_t PAGE = 4096;

struct Job {
    int fd = -1;
    const int64_t* offsets = nullptr;
    uint8_t* out = nullptr;
    int64_t n = 0, row = 0;
    bool direct = false;
    std::atomic<int64_t> next{0};
    std::atomic<int> err{0};  // the first errno (or EIO for a short read)
};

class Pool {
public:
    ~Pool() {
        {
            std::lock_guard<std::mutex> g(mu_);
            stop_ = true;
        }
        for (auto& w : workers_) w->cv.notify_one();
        for (auto& w : workers_) w->th.join();
    }

    // adds workers up to n; waits for a running job first (a new worker must not join it)
    void grow(int n) {
        std::lock_guard<std::mutex> one(run_mu_);
        while ((int)workers_.size() < n) {
            workers_.push_back(std::make_unique<Worker>());
            Worker* w = workers_.back().get();
            w->th = std::thread([this, w] { loop(w); });
        }
    }

    // runs the job on the workers 0 .. helpers - 1 (helpers <= the pool size) and the calling thread; one job at a time.
    // Each worker has its own condition variable, so a call wakes exactly the workers it uses.
    int run(Job& job, int helpers) {
        std::lock_guard<std::mutex> one(run_mu_);
        {
            std::lock_guard<std::mutex> g(mu_);
            job_ = &job;
            active_ = helpers;
            for (int i = 0; i < helpers; ++i) workers_[i]->go = true;
        }
        for (int i = 0; i < helpers; ++i) workers_[i]->cv.notify_one();
        work(job);
        std::unique_lock<std::mutex> g(mu_);
        done_cv_.wait(g, [this] { return active_ == 0; });
        job_ = nullptr;
        return job.err.load();
    }

private:
    struct Worker {
        std::thread th;
        std::condition_variable cv;
        bool go = false;  // a job waits for this worker (guarded by mu_)
    };

    static void work(Job& job) {
        // the aligned page buffer of this thread (two pages: a row may straddle a page boundary)
        thread_local uint8_t* buf = nullptr;
        if (job.direct && buf == nullptr && posix_memalign(reinterpret_cast<void**>(&buf), PAGE, 2 * PAGE) != 0) {
            buf = nullptr;
            int z = 0;
            job.err.compare_exchange_strong(z, ENOMEM);
            return;
        }
        for (int64_t i; (i = job.next.fetch_add(1, std::memory_order_relaxed)) < job.n;) {
            const int64_t off = job.offsets[i];
            uint8_t* dst = job.out + i * job.row;
            ssize_t got;
            if (job.direct) {
                const int64_t base = off & ~(PAGE - 1), len = ((off + job.row - base + PAGE - 1) / PAGE) * PAGE;
                got = pread(job.fd, buf, len, base);
                if (got >= off + job.row - base) memcpy(dst, buf + (off - base), job.row);
                else if (got >= 0) got = -2;
            } else {
                got = pread(job.fd, dst, job.row, off);
                if (got >= 0 && got != job.row) got = -2;
            }
            if (got < 0) {
                int z = 0;
                job.err.compare_exchange_strong(z, got == -2 ? EIO : errno);
            }
        }
    }

    void loop(Worker* w) {
        for (;;) {
            Job* job;
            {
                std::unique_lock<std::mutex> g(mu_);
                w->cv.wait(g, [&] { return stop_ || w->go; });
                if (stop_) return;
                w->go = false;
                job = job_;
            }
            work(*job);
            {
                std::lock_guard<std::mutex> g(mu_);
                if (--active_ == 0) done_cv_.notify_all();
            }
        }
    }

    std::vector<std::unique_ptr<Worker>> workers_;
    std::mutex mu_, run_mu_;
    std::condition_variable done_cv_;
    Job* job_ = nullptr;
    int active_ = 0;  // the helpers still running on the current job
    bool stop_ = false;
};

std::mutex pool_mu;
Pool* pool = nullptr;  // never freed: joining threads in a static destructor at interpreter exit is not safe

}  // namespace

// Reads n rows of `row` bytes at the byte offsets offsets[0 .. n) of fd into out (n * row bytes), with at least `threads`
// threads (the pool keeps the largest count asked). direct: fd is open with O_DIRECT. Returns 0, or the first errno.
int ple_read_rows(int fd, const int64_t* offsets, uint8_t* out, int64_t n, int64_t row, bool direct, int threads) {
    if (n <= 0) return 0;
    if (row <= 0 || row > PAGE || threads < 1) return EINVAL;
    {
        std::lock_guard<std::mutex> g(pool_mu);
        if (pool == nullptr) pool = new Pool();
    }
    const int helpers = (int)std::min<int64_t>(threads - 1, n - 1);  // the caller is the last thread
    pool->grow(helpers);
    Job job;
    job.fd = fd;
    job.offsets = offsets;
    job.out = out;
    job.n = n;
    job.row = row;
    job.direct = direct;
    return pool->run(job, helpers);
}
