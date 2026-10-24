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
//
// The row cache (ple_cache_*): a set-associative cache of rows in host memory, with a fixed capacity in rows. Each set
// holds WAYS rows; a row id maps to one set by a hash, and a miss replaces the least recently used way of the set. The
// rows live in one arena (capacity x row bytes), the ids and the use stamps in two arrays beside it. One mutex guards
// a cache: a lookup or an insert of n rows takes it once. The feeder of engine/weights/ple.py (PleFeeder) looks up
// the rows of a chunk, reads the misses with ple_read_rows and inserts them. Python: ops().ple_cache_new and friends
// (bound in qsa_bind.cpp; the handle frees the cache when Python drops it). Tests: tests/test_ple_feeder.py.
#include <errno.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include <algorithm>
#include <cstdint>
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

// ------------------------------------------------------------------------------------------------------------------
// The row cache.
struct PleRowCache {
    int64_t nsets = 0, ways = 0, row = 0;
    std::vector<int64_t> ids;     // [nsets * ways], -1 = empty
    std::vector<uint64_t> stamp;  // [nsets * ways], the last use
    std::vector<uint8_t> data;    // [nsets * ways * row]
    uint64_t clock = 0;
    int64_t hits = 0, misses = 0, filled = 0;
    std::mutex mu;
};

namespace {
inline uint64_t mix64(uint64_t x) {  // splitmix64's finalizer: the set of an id
    x ^= x >> 30;
    x *= 0xbf58476d1ce4e5b9ull;
    x ^= x >> 27;
    x *= 0x94d049bb133111ebull;
    return x ^ (x >> 31);
}
}  // namespace

// A cache of at least `capacity` rows of `row` bytes in sets of `ways` (the capacity rounds up to whole sets).
PleRowCache* ple_cache_create(int64_t capacity, int64_t row, int64_t ways) {
    auto* c = new PleRowCache();
    c->ways = std::max<int64_t>(1, ways);
    c->nsets = std::max<int64_t>(1, (capacity + c->ways - 1) / c->ways);
    c->row = row;
    const size_t n = (size_t)(c->nsets * c->ways);
    c->ids.assign(n, -1);
    c->stamp.assign(n, 0);
    c->data.resize(n * (size_t)row);
    return c;
}

void ple_cache_destroy(PleRowCache* c) { delete c; }

// out[i] = the row of ids[i] where the cache holds it (miss[i] = 0); else miss[i] = 1 and out[i] is untouched.
void ple_cache_lookup(PleRowCache* c, const int64_t* ids, int64_t n, uint8_t* out, uint8_t* miss) {
    std::lock_guard<std::mutex> g(c->mu);
    int64_t h = 0;
    for (int64_t i = 0; i < n; ++i) {
        const int64_t id = ids[i];
        const int64_t base = (int64_t)(mix64((uint64_t)id) % (uint64_t)c->nsets) * c->ways;
        int64_t hit = -1;
        for (int64_t w = 0; w < c->ways; ++w)
            if (c->ids[base + w] == id) {
                hit = base + w;
                break;
            }
        if (hit >= 0 && id >= 0) {
            memcpy(out + i * c->row, c->data.data() + hit * c->row, c->row);
            c->stamp[hit] = ++c->clock;
            miss[i] = 0;
            ++h;
        } else {
            miss[i] = 1;
        }
    }
    c->hits += h;
    c->misses += n - h;
}

// Stores the rows of ids (ids < 0 are skipped; an id the cache holds keeps its row): each into an empty way of its set,
// else in place of the least recently used one.
void ple_cache_insert(PleRowCache* c, const int64_t* ids, int64_t n, const uint8_t* rows) {
    std::lock_guard<std::mutex> g(c->mu);
    for (int64_t i = 0; i < n; ++i) {
        const int64_t id = ids[i];
        if (id < 0) continue;
        const int64_t base = (int64_t)(mix64((uint64_t)id) % (uint64_t)c->nsets) * c->ways;
        int64_t slot = -1;
        uint64_t oldest = UINT64_MAX;
        for (int64_t w = 0; w < c->ways; ++w) {
            const int64_t k = base + w;
            if (c->ids[k] == id) {
                slot = -2;
                break;
            }
            const uint64_t s = c->ids[k] < 0 ? 0 : c->stamp[k];  // an empty way is the oldest
            if (s < oldest) {
                oldest = s;
                slot = k;
            }
        }
        if (slot < 0) continue;
        if (c->ids[slot] < 0) ++c->filled;
        c->ids[slot] = id;
        c->stamp[slot] = ++c->clock;
        memcpy(c->data.data() + slot * c->row, rows + i * c->row, c->row);
    }
}

// (hits, misses, rows held, capacity in rows)
void ple_cache_counters(PleRowCache* c, int64_t* out) {
    std::lock_guard<std::mutex> g(c->mu);
    out[0] = c->hits;
    out[1] = c->misses;
    out[2] = c->filled;
    out[3] = c->nsets * c->ways;
}

void ple_cache_reset_counters(PleRowCache* c) {
    std::lock_guard<std::mutex> g(c->mu);
    c->hits = c->misses = 0;
}
