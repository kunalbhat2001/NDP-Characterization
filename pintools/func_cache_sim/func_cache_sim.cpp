/*
 * func_cache_sim.cpp
 *
 * Intel Pin Tool: Function-Level Cache & DRAM Access Simulator
 *
 * Simulates a configurable 3-level cache hierarchy (thread-private L1,
 * thread-private L2, shared LLC) and estimates the number of DRAM accesses
 * (LLC misses + dirty writebacks) caused by each function call, categorized
 * by function name and argument values.
 *
 * Enhancements:
 *   1. -startFF knob: Fast-forward simulation until magic instruction
 *      (xchg %rbx, %rbx with RCX=0x2 for ROI Start, RCX=0x4 for ROI End).
 *   2. Thread-private L1/L2 caches to eliminate lock contention on hits.
 *   3. -csv <filename> knob for structured data output.
 *   4. Function filtering (include, exclude, boilerplate suppression, exact/substring).
 */

#include "pin.H"
#include <iostream>
#include <fstream>
#include <iomanip>
#include <string>
#include <vector>
#include <unordered_map>
#include <map>
#include <algorithm>
#include <sstream>
#include <cmath>
#include <sys/stat.h>

#define CACHELINE_SIZE 64
#define CACHELINE_MASK (~((ADDRINT)(CACHELINE_SIZE - 1)))
#define PAGE_SHIFT 12
#ifndef PAGE_SIZE
#define PAGE_SIZE ((ADDRINT)1 << PAGE_SHIFT)
#endif

// =====================================================================
// Command Line Knobs
// =====================================================================

KNOB<double> KnobL1SizeMB(KNOB_MODE_WRITEONCE, "pintool",
    "l1_mb", "0.0625", "L1 Data Cache size in MB per thread (default: 0.0625 = 64KB)");

KNOB<double> KnobL2SizeMB(KNOB_MODE_WRITEONCE, "pintool",
    "l2_mb", "0.5", "L2 Cache size in MB per thread (default: 0.5 = 512KB)");

KNOB<double> KnobLlcSizeMB(KNOB_MODE_WRITEONCE, "pintool",
    "llc_mb", "16.0", "Shared LLC (Last Level Cache) size in MB (default: 16.0)");

KNOB<UINT32> KnobL1Assoc(KNOB_MODE_WRITEONCE, "pintool",
    "l1_assoc", "8", "L1 Cache associativity (ways, default: 8)");

KNOB<UINT32> KnobL2Assoc(KNOB_MODE_WRITEONCE, "pintool",
    "l2_assoc", "8", "L2 Cache associativity (ways, default: 8)");

KNOB<UINT32> KnobLlcAssoc(KNOB_MODE_WRITEONCE, "pintool",
    "llc_assoc", "16", "LLC Cache associativity (ways, default: 16)");

KNOB<BOOL> KnobHeapOnly(KNOB_MODE_WRITEONCE, "pintool",
    "heap_only", "0", "Only simulate heap memory accesses (1=Heap only, 0=All data accesses)");

KNOB<std::string> KnobOutputFile(KNOB_MODE_WRITEONCE, "pintool",
    "o", "func_cache_sim_report.txt", "Specify output summary report file name");

KNOB<std::string> KnobCsvFile(KNOB_MODE_WRITEONCE, "pintool",
    "csv", "", "Specify output CSV report file name (optional)");

KNOB<std::string> KnobCsvDir(KNOB_MODE_WRITEONCE, "pintool",
    "csv_dir", "", "Directory to generate per-function CSV outputs under (optional)");

KNOB<BOOL> KnobStartFF(KNOB_MODE_WRITEONCE, "pintool",
    "startFF", "0", "Fast-forward simulation until ROI start magic instruction (xchg %rbx, %rbx)");

KNOB<BOOL> KnobMainExecutableOnly(KNOB_MODE_WRITEONCE, "pintool",
    "main_only", "1", "Only instrument routines in the main executable (1=Yes, 0=All)");

KNOB<UINT32> KnobNumArgs(KNOB_MODE_WRITEONCE, "pintool",
    "num_args", "6", "Number of function arguments to capture and hash (1 to 6, default: 6)");

KNOB<std::string> KnobFuncFilter(KNOB_MODE_WRITEONCE, "pintool",
    "func", "", "Filter by function name(s), comma-separated (e.g. -func pr_pull_step,sssp_relax_edges)");

KNOB<std::string> KnobFilterAlias(KNOB_MODE_WRITEONCE, "pintool",
    "filter", "", "Alias for -func");

KNOB<std::string> KnobExcludeFuncs(KNOB_MODE_WRITEONCE, "pintool",
    "exclude", "", "Comma-separated list of function names to exclude");

KNOB<BOOL> KnobHideBoilerplate(KNOB_MODE_WRITEONCE, "pintool",
    "no_boilerplate", "1", "Automatically hide compiler/runtime boilerplate routines");

KNOB<BOOL> KnobExactMatch(KNOB_MODE_WRITEONCE, "pintool",
    "exact", "0", "Require exact function name match instead of substring match");

KNOB<BOOL> KnobVerbose(KNOB_MODE_WRITEONCE, "pintool",
    "verbose", "0", "Log live function entries/exits to stderr");

// Architectural Analytical Cycle Model Knobs (aligned with StarNUMA & CXL NDP)
KNOB<double> KnobHostBaseCpi(KNOB_MODE_WRITEONCE, "pintool",
    "cpi_host", "0.5", "Host 4-wide OOO core base CPI without memory stalls (default: 0.5)");

KNOB<double> KnobNdpBaseCpi(KNOB_MODE_WRITEONCE, "pintool",
    "cpi_ndp", "1.0", "NDP 2-wide downscaled core base CPI without memory stalls (default: 1.0)");

KNOB<double> KnobL2Latency(KNOB_MODE_WRITEONCE, "pintool",
    "lat_l2", "10.0", "L2 cache hit latency in cycles (default: 10.0)");

KNOB<double> KnobLlcLatency(KNOB_MODE_WRITEONCE, "pintool",
    "lat_llc", "20.0", "LLC cache hit latency in cycles (default: 20.0)");

KNOB<double> KnobDramLocalLatency(KNOB_MODE_WRITEONCE, "pintool",
    "lat_dram", "150.0", "Local DDR5 DRAM access latency in cycles (default: 150.0)");

KNOB<double> KnobCxlLinkLatency(KNOB_MODE_WRITEONCE, "pintool",
    "lat_cxl", "240.0", "Interconnect network CXL round-trip link latency in cycles (default: 240.0)");

// =====================================================================
// Filtering Helpers
// =====================================================================

std::vector<std::string> SplitCommaList(const std::string& str) {
    std::vector<std::string> tokens;
    if (str.empty()) return tokens;
    std::stringstream ss(str);
    std::string item;
    while (std::getline(ss, item, ',')) {
        size_t first = item.find_first_not_of(" \t\r\n");
        size_t last = item.find_last_not_of(" \t\r\n");
        if (first != std::string::npos && last != std::string::npos) {
            tokens.push_back(item.substr(first, last - first + 1));
        }
    }
    return tokens;
}

bool IsBoilerplateRoutine(const std::string& name) {
    if (name.empty()) return true;
    if (name[0] == '.') return true; // .plt, .plt.sec, .plt.got
    if (name == "_init" || name == "_fini" || name == "_start") return true;
    if (name.find("tm_clones") != std::string::npos) return true;
    if (name.find("frame_dummy") != std::string::npos) return true;
    if (name.find("__do_global_dtors") != std::string::npos) return true;
    if (name.find("__libc_") != std::string::npos) return true;
    return false;
}

bool RoutinePassesFilter(const std::string& rtn_name,
                         const std::vector<std::string>& includes,
                         const std::vector<std::string>& excludes,
                         bool hide_boilerplate,
                         bool exact_match) {
    if (hide_boilerplate && IsBoilerplateRoutine(rtn_name)) {
        return false;
    }

    for (const auto& excl : excludes) {
        if (exact_match) {
            if (rtn_name == excl) return false;
        } else {
            if (rtn_name.find(excl) != std::string::npos) return false;
        }
    }

    if (includes.empty()) {
        return true;
    }

    for (const auto& inc : includes) {
        if (exact_match) {
            if (rtn_name == inc) return true;
        } else {
            if (rtn_name.find(inc) != std::string::npos) return true;
        }
    }

    return false;
}

// =====================================================================
// Set-Associative LRU Cache Model
// =====================================================================

struct CacheBlock {
    ADDRINT tag = 0;
    bool valid = false;
    bool dirty = false;
};

class SetAssociativeCache {
public:
    std::string name;
    UINT64 size_bytes = 0;
    UINT32 line_size = CACHELINE_SIZE;
    UINT32 assoc = 1;
    UINT32 num_sets = 1;
    UINT32 set_shift = 6;
    UINT32 set_mask = 0;

    std::vector<std::vector<CacheBlock>> sets;

    VOID Init(const std::string& n, UINT64 size_b, UINT32 a) {
        name = n;
        line_size = CACHELINE_SIZE;
        assoc = (a > 0) ? a : 1;

        if (size_b < (UINT64)assoc * line_size) {
            size_b = (UINT64)assoc * line_size;
        }
        size_bytes = size_b;

        UINT32 raw_sets = size_bytes / (line_size * assoc);
        num_sets = 1;
        while (num_sets * 2 <= raw_sets) {
            num_sets *= 2;
        }

        set_shift = 6;
        set_mask = num_sets - 1;

        sets.resize(num_sets);
        for (auto& s : sets) {
            s.resize(assoc);
        }
    }

    bool Access(ADDRINT addr, bool is_write, bool& evicted_dirty, ADDRINT& evicted_addr) {
        evicted_dirty = false;
        evicted_addr = 0;

        ADDRINT block_addr = addr & CACHELINE_MASK;
        UINT32 set_idx = (block_addr >> set_shift) & set_mask;
        auto& set = sets[set_idx];

        for (UINT32 i = 0; i < assoc; ++i) {
            if (set[i].valid && set[i].tag == block_addr) {
                if (is_write) set[i].dirty = true;
                if (i > 0) {
                    CacheBlock hit_block = set[i];
                    for (UINT32 j = i; j > 0; --j) {
                        set[j] = set[j - 1];
                    }
                    set[0] = hit_block;
                }
                return true;
            }
        }

        CacheBlock victim = set[assoc - 1];
        if (victim.valid && victim.dirty) {
            evicted_dirty = true;
            evicted_addr = victim.tag;
        }

        for (UINT32 j = assoc - 1; j > 0; --j) {
            set[j] = set[j - 1];
        }

        set[0].valid = true;
        set[0].tag = block_addr;
        set[0].dirty = is_write;

        return false;
    }
};

// =====================================================================
// Function Signature & Per-Call Statistics
// =====================================================================

struct CallSignature {
    ADDRINT func_addr = 0;
    std::string func_name;
    std::vector<ADDRINT> args;
    size_t hash_val = 0;

    bool operator==(const CallSignature& other) const {
        return func_addr == other.func_addr && args == other.args;
    }
};

struct FunctionDramStats {
    CallSignature sig;
    UINT64 invocations = 0;
    UINT64 total_instructions = 0;
    UINT64 total_compute_insts = 0;
    UINT64 total_mem_insts = 0;
    UINT64 total_mem_ops = 0;

    UINT64 excl_l1_misses = 0;
    UINT64 excl_l2_misses = 0;
    UINT64 excl_llc_misses = 0;
    UINT64 excl_dram_reads = 0;
    UINT64 excl_dram_writes = 0;
    UINT64 excl_total_dram = 0;
    UINT64 max_excl_dram_single_call = 0;

    UINT64 incl_total_dram = 0;
    UINT64 max_incl_dram_single_call = 0;
    UINT64 incl_instructions = 0;

    UINT32 max_recursion_depth = 0;
    UINT64 recursive_invocations = 0;
};

struct CallFrame {
    ADDRINT func_addr = 0;
    std::string func_name;
    std::vector<ADDRINT> args;
    size_t hash_val = 0;
    UINT32 recursion_depth = 0;

    UINT64 instructions = 0;
    UINT64 compute_insts = 0;
    UINT64 mem_insts = 0;

    UINT64 mem_ops = 0;
    UINT64 excl_l1_misses = 0;
    UINT64 excl_l2_misses = 0;
    UINT64 excl_llc_misses = 0;
    UINT64 excl_dram_reads = 0;
    UINT64 excl_dram_writes = 0;
    UINT64 excl_total_dram = 0;

    UINT64 incl_instructions = 0;
    UINT64 incl_total_dram = 0;
};

struct PageCacheEntry {
    ADDRINT page = (ADDRINT)-1;
    bool is_heap = false;
};

struct ThreadData {
    std::vector<CallFrame> call_stack;
    std::unordered_map<ADDRINT, UINT32> active_depth;

    ADDRINT pending_alloc_size = 0;
    ADDRINT pending_realloc_ptr = 0;

    PageCacheEntry page_cache[64];

    // Thread-private L1 and L2 caches (lock-free hits)
    SetAssociativeCache private_l1;
    SetAssociativeCache private_l2;

    UINT64 thread_instructions = 0;
    UINT64 thread_compute_insts = 0;
    UINT64 thread_mem_insts = 0;

    UINT64 thread_mem_ops = 0;
    UINT64 thread_l1_hits = 0;
    UINT64 thread_l1_misses = 0;
    UINT64 thread_l2_hits = 0;
    UINT64 thread_l2_misses = 0;
};

// =====================================================================
// Global State
// =====================================================================

static volatile bool g_in_roi = true;

// Shared LLC
static SetAssociativeCache g_shared_llc;
static PIN_LOCK g_llc_lock;

// Global stats
static UINT64 g_total_instructions = 0;
static UINT64 g_total_compute_insts = 0;
static UINT64 g_total_mem_insts = 0;
static UINT64 g_total_mem_ops = 0;
static UINT64 g_global_l1_hits = 0;
static UINT64 g_global_l1_misses = 0;
static UINT64 g_global_l2_hits = 0;
static UINT64 g_global_l2_misses = 0;
static UINT64 g_global_llc_hits = 0;
static UINT64 g_global_llc_misses = 0;
static UINT64 g_global_dram_reads = 0;
static UINT64 g_global_dram_writes = 0;

static PIN_LOCK g_stats_lock;
static PIN_LOCK g_threads_list_lock;
static std::vector<ThreadData*> g_active_threads;
static std::unordered_map<size_t, FunctionDramStats> g_call_hash_map;

static PIN_LOCK g_heap_lock;
static std::map<ADDRINT, ADDRINT> g_heap_blocks;
static ADDRINT g_min_heap_addr = (ADDRINT)-1;
static ADDRINT g_max_heap_addr = 0;
static ADDRINT g_proc_heap_start = 0;
static ADDRINT g_proc_heap_end = 0;

static TLS_KEY tls_key;

// =====================================================================
// Hash Functions & Helpers
// =====================================================================

inline size_t HashCombine(size_t seed, ADDRINT v) {
    return seed ^ (std::hash<ADDRINT>{}(v) + 0x9e3779b97f4a7c15ULL + (seed << 6) + (seed >> 2));
}

size_t ComputeCallHash(ADDRINT func_addr, const std::vector<ADDRINT>& args) {
    size_t seed = std::hash<ADDRINT>{}(func_addr);
    for (ADDRINT arg : args) {
        seed = HashCombine(seed, arg);
    }
    return seed;
}

void ParseProcMaps() {
    std::ifstream maps("/proc/self/maps");
    if (!maps.is_open()) return;

    std::string line;
    while (std::getline(maps, line)) {
        if (line.find("[heap]") != std::string::npos) {
            std::stringstream ss(line);
            std::string range;
            ss >> range;
            size_t dash = range.find('-');
            if (dash != std::string::npos) {
                g_proc_heap_start = std::stoull(range.substr(0, dash), nullptr, 16);
                g_proc_heap_end = std::stoull(range.substr(dash + 1), nullptr, 16);
                if (g_proc_heap_start < g_min_heap_addr) g_min_heap_addr = g_proc_heap_start;
                if (g_proc_heap_end > g_max_heap_addr) g_max_heap_addr = g_proc_heap_end;
            }
            break;
        }
    }
    maps.close();
}

inline bool IsAddressInHeap(ThreadData* tdata, ADDRINT addr) {
    if (addr < g_min_heap_addr || addr >= g_max_heap_addr) return false;

    ADDRINT page = addr >> PAGE_SHIFT;
    UINT32 cache_idx = page & 0x3F;
    if (tdata->page_cache[cache_idx].page == page) {
        return tdata->page_cache[cache_idx].is_heap;
    }

    PIN_GetLock(&g_heap_lock, 1);
    bool in_heap = (g_proc_heap_start && addr >= g_proc_heap_start && addr < g_proc_heap_end);
    if (!in_heap && !g_heap_blocks.empty()) {
        auto it = g_heap_blocks.upper_bound(addr);
        if (it != g_heap_blocks.begin()) {
            --it;
            if (addr >= it->first && addr < it->second) in_heap = true;
        }
    }
    PIN_ReleaseLock(&g_heap_lock);

    tdata->page_cache[cache_idx].page = page;
    tdata->page_cache[cache_idx].is_heap = in_heap;
    return in_heap;
}

void RegisterHeapBlock(ADDRINT start, ADDRINT size) {
    if (start == 0 || size == 0) return;
    PIN_GetLock(&g_heap_lock, 1);
    g_heap_blocks[start] = start + size;
    if (start < g_min_heap_addr) g_min_heap_addr = start;
    if (start + size > g_max_heap_addr) g_max_heap_addr = start + size;
    PIN_ReleaseLock(&g_heap_lock);
}

void UnregisterHeapBlock(ADDRINT start) {
    if (start == 0) return;
    PIN_GetLock(&g_heap_lock, 1);
    g_heap_blocks.erase(start);
    PIN_ReleaseLock(&g_heap_lock);
}

// =====================================================================
// Memory Allocator Interceptors
// =====================================================================

VOID BeforeMalloc(THREADID tid, ADDRINT size) {
    ThreadData* tdata = static_cast<ThreadData*>(PIN_GetThreadData(tls_key, tid));
    if (tdata) tdata->pending_alloc_size = size;
}

VOID AfterMalloc(THREADID tid, ADDRINT ret_ptr) {
    ThreadData* tdata = static_cast<ThreadData*>(PIN_GetThreadData(tls_key, tid));
    if (tdata && ret_ptr != 0 && tdata->pending_alloc_size > 0) {
        RegisterHeapBlock(ret_ptr, tdata->pending_alloc_size);
        tdata->pending_alloc_size = 0;
    }
}

VOID BeforeCalloc(THREADID tid, ADDRINT nmemb, ADDRINT size) {
    ThreadData* tdata = static_cast<ThreadData*>(PIN_GetThreadData(tls_key, tid));
    if (tdata) tdata->pending_alloc_size = nmemb * size;
}

VOID BeforeRealloc(THREADID tid, ADDRINT ptr, ADDRINT size) {
    ThreadData* tdata = static_cast<ThreadData*>(PIN_GetThreadData(tls_key, tid));
    if (tdata) {
        tdata->pending_realloc_ptr = ptr;
        tdata->pending_alloc_size = size;
    }
}

VOID AfterRealloc(THREADID tid, ADDRINT ret_ptr) {
    ThreadData* tdata = static_cast<ThreadData*>(PIN_GetThreadData(tls_key, tid));
    if (tdata && ret_ptr != 0 && tdata->pending_alloc_size > 0) {
        if (tdata->pending_realloc_ptr != 0) {
            UnregisterHeapBlock(tdata->pending_realloc_ptr);
        }
        RegisterHeapBlock(ret_ptr, tdata->pending_alloc_size);
        tdata->pending_alloc_size = 0;
        tdata->pending_realloc_ptr = 0;
    }
}

VOID BeforeFree(ADDRINT ptr) {
    if (ptr != 0) UnregisterHeapBlock(ptr);
}

// =====================================================================
// ROI Magic Instruction Interceptor
// =====================================================================

VOID OnMagicInstruction(THREADID tid, ADDRINT field) {
    if (field == 0x2 || field == 0x0 || field == 0x3) {
        g_in_roi = true;
        std::cerr << "[func_cache_sim] ROI Start detected on thread " << tid << "\n";
    } else if (field == 0x4 || field == 0x1) {
        g_in_roi = false;
        std::cerr << "[func_cache_sim] ROI End detected on thread " << tid << "\n";
    }
}

// =====================================================================
// Routine Entry / Exit Analysis
// =====================================================================

VOID OnRoutineEntry(THREADID tid, ADDRINT rtn_addr,
                    ADDRINT arg0, ADDRINT arg1, ADDRINT arg2,
                    ADDRINT arg3, ADDRINT arg4, ADDRINT arg5) {
    if (KnobStartFF.Value() && !g_in_roi) return;

    ThreadData* tdata = static_cast<ThreadData*>(PIN_GetThreadData(tls_key, tid));
    if (!tdata) return;

    UINT32 current_depth = ++(tdata->active_depth[rtn_addr]);

    UINT32 num_args = KnobNumArgs.Value();
    if (num_args > 6) num_args = 6;

    std::vector<ADDRINT> args;
    args.reserve(num_args);
    ADDRINT raw_args[6] = { arg0, arg1, arg2, arg3, arg4, arg5 };
    for (UINT32 i = 0; i < num_args; ++i) {
        args.push_back(raw_args[i]);
    }

    size_t call_hash = ComputeCallHash(rtn_addr, args);

    CallFrame frame;
    frame.func_addr = rtn_addr;
    frame.func_name = RTN_FindNameByAddress(rtn_addr);
    frame.args = args;
    frame.hash_val = call_hash;
    frame.recursion_depth = current_depth;

    if (KnobVerbose.Value()) {
        std::cerr << "[TID " << tid << "] CALL " << frame.func_name
                  << " [Hash: 0x" << std::hex << call_hash << std::dec << "]\n";
    }

    tdata->call_stack.push_back(frame);
}

VOID OnRoutineExit(THREADID tid, ADDRINT rtn_addr) {
    if (KnobStartFF.Value() && !g_in_roi) return;

    ThreadData* tdata = static_cast<ThreadData*>(PIN_GetThreadData(tls_key, tid));
    if (!tdata || tdata->call_stack.empty()) return;

    if (tdata->active_depth[rtn_addr] > 0) {
        tdata->active_depth[rtn_addr]--;
    }

    CallFrame finished_frame;
    bool found = false;

    while (!tdata->call_stack.empty()) {
        CallFrame top = tdata->call_stack.back();
        tdata->call_stack.pop_back();
        if (top.func_addr == rtn_addr) {
            finished_frame = top;
            found = true;
            break;
        } else {
            if (!tdata->call_stack.empty()) {
                CallFrame& parent = tdata->call_stack.back();
                parent.incl_total_dram += top.incl_total_dram;
                parent.incl_instructions += top.incl_instructions;
            }
        }
    }

    if (!found) return;

    finished_frame.incl_instructions += finished_frame.instructions;

    // Aggregate into global HashMap
    PIN_GetLock(&g_stats_lock, tid + 1);
    FunctionDramStats& stats = g_call_hash_map[finished_frame.hash_val];
    if (stats.invocations == 0) {
        stats.sig.func_addr = finished_frame.func_addr;
        stats.sig.func_name = finished_frame.func_name;
        stats.sig.args = finished_frame.args;
        stats.sig.hash_val = finished_frame.hash_val;
    }

    stats.invocations++;
    stats.total_instructions += finished_frame.instructions;
    stats.total_compute_insts += finished_frame.compute_insts;
    stats.total_mem_insts += finished_frame.mem_insts;
    stats.total_mem_ops += finished_frame.mem_ops;
    stats.excl_l1_misses += finished_frame.excl_l1_misses;
    stats.excl_l2_misses += finished_frame.excl_l2_misses;
    stats.excl_llc_misses += finished_frame.excl_llc_misses;
    stats.excl_dram_reads += finished_frame.excl_dram_reads;
    stats.excl_dram_writes += finished_frame.excl_dram_writes;
    stats.excl_total_dram += finished_frame.excl_total_dram;

    if (finished_frame.excl_total_dram > stats.max_excl_dram_single_call) {
        stats.max_excl_dram_single_call = finished_frame.excl_total_dram;
    }

    stats.incl_total_dram += finished_frame.incl_total_dram;
    stats.incl_instructions += finished_frame.incl_instructions;
    if (finished_frame.incl_total_dram > stats.max_incl_dram_single_call) {
        stats.max_incl_dram_single_call = finished_frame.incl_total_dram;
    }

    if (finished_frame.recursion_depth > 1) {
        stats.recursive_invocations++;
        if (finished_frame.recursion_depth > stats.max_recursion_depth) {
            stats.max_recursion_depth = finished_frame.recursion_depth;
        }
    }
    PIN_ReleaseLock(&g_stats_lock);

    if (!tdata->call_stack.empty()) {
        CallFrame& parent = tdata->call_stack.back();
        parent.incl_total_dram += finished_frame.incl_total_dram;
        parent.incl_instructions += finished_frame.incl_instructions;
    }
}

// =====================================================================
// Memory Access Simulation (Thread-Private L1/L2 + Shared LLC)
// =====================================================================

VOID OnMemoryAccess(THREADID tid, ADDRINT addr, UINT32 size, BOOL is_write) {
    if (KnobStartFF.Value() && !g_in_roi) return;

    ThreadData* tdata = static_cast<ThreadData*>(PIN_GetThreadData(tls_key, tid));
    if (!tdata || tdata->call_stack.empty()) return;

    if (KnobHeapOnly.Value() && !IsAddressInHeap(tdata, addr)) {
        return;
    }

    tdata->thread_mem_ops++;
    CallFrame& frame = tdata->call_stack.back();
    frame.mem_ops++;

    // 1. Thread-Private L1 Data Cache (Lock-Free)
    bool l1_evict_dirty = false;
    ADDRINT l1_evict_addr = 0;
    bool l1_hit = tdata->private_l1.Access(addr, is_write, l1_evict_dirty, l1_evict_addr);

    if (l1_hit) {
        tdata->thread_l1_hits++;
        return;
    }
    tdata->thread_l1_misses++;
    frame.excl_l1_misses++;

    // 2. Thread-Private L2 Cache (Lock-Free)
    bool l2_evict_dirty = false;
    ADDRINT l2_evict_addr = 0;
    bool l2_hit = tdata->private_l2.Access(addr, is_write, l2_evict_dirty, l2_evict_addr);

    // Propagate L1 dirty eviction down to L2
    if (l1_evict_dirty) {
        bool dummy_dirty = false;
        ADDRINT dummy_addr = 0;
        tdata->private_l2.Access(l1_evict_addr, true, dummy_dirty, dummy_addr);
        if (dummy_dirty) {
            l2_evict_dirty = true;
            l2_evict_addr = dummy_addr;
        }
    }

    if (l2_hit) {
        tdata->thread_l2_hits++;
        return;
    }
    tdata->thread_l2_misses++;
    frame.excl_l2_misses++;

    // 3. Shared LLC (Protected by g_llc_lock)
    PIN_GetLock(&g_llc_lock, tid + 1);

    // If L2 evicted a dirty block, write it to LLC
    if (l2_evict_dirty) {
        bool llc_wb_evict_dirty = false;
        ADDRINT llc_wb_evict_addr = 0;
        g_shared_llc.Access(l2_evict_addr, true, llc_wb_evict_dirty, llc_wb_evict_addr);
        if (llc_wb_evict_dirty) {
            g_global_dram_writes++;
            frame.excl_dram_writes++;
            frame.excl_total_dram++;
            frame.incl_total_dram++;
        }
    }

    bool llc_evict_dirty = false;
    ADDRINT llc_evict_addr = 0;
    bool llc_hit = g_shared_llc.Access(addr, is_write, llc_evict_dirty, llc_evict_addr);

    if (llc_hit) {
        g_global_llc_hits++;
        PIN_ReleaseLock(&g_llc_lock);
        return;
    }

    // LLC Miss -> DRAM Access
    g_global_llc_misses++;
    g_global_dram_reads++;
    frame.excl_llc_misses++;
    frame.excl_dram_reads++;
    frame.excl_total_dram++;
    frame.incl_total_dram++;

    if (llc_evict_dirty) {
        g_global_dram_writes++;
        frame.excl_dram_writes++;
        frame.excl_total_dram++;
        frame.incl_total_dram++;
    }

    PIN_ReleaseLock(&g_llc_lock);
}

// =====================================================================
// Instrumentation Functions
// =====================================================================

VOID ImageLoad(IMG img, VOID *v) {
    RTN mallocRtn = RTN_FindByName(img, "malloc");
    if (RTN_Valid(mallocRtn)) {
        RTN_Open(mallocRtn);
        RTN_InsertCall(mallocRtn, IPOINT_BEFORE, (AFUNPTR)BeforeMalloc,
                       IARG_THREAD_ID, IARG_FUNCARG_ENTRYPOINT_VALUE, 0, IARG_END);
        RTN_InsertCall(mallocRtn, IPOINT_AFTER, (AFUNPTR)AfterMalloc,
                       IARG_THREAD_ID, IARG_FUNCRET_EXITPOINT_VALUE, IARG_END);
        RTN_Close(mallocRtn);
    }

    RTN callocRtn = RTN_FindByName(img, "calloc");
    if (RTN_Valid(callocRtn)) {
        RTN_Open(callocRtn);
        RTN_InsertCall(callocRtn, IPOINT_BEFORE, (AFUNPTR)BeforeCalloc,
                       IARG_THREAD_ID,
                       IARG_FUNCARG_ENTRYPOINT_VALUE, 0,
                       IARG_FUNCARG_ENTRYPOINT_VALUE, 1,
                       IARG_END);
        RTN_InsertCall(callocRtn, IPOINT_AFTER, (AFUNPTR)AfterMalloc,
                       IARG_THREAD_ID, IARG_FUNCRET_EXITPOINT_VALUE, IARG_END);
        RTN_Close(callocRtn);
    }

    RTN reallocRtn = RTN_FindByName(img, "realloc");
    if (RTN_Valid(reallocRtn)) {
        RTN_Open(reallocRtn);
        RTN_InsertCall(reallocRtn, IPOINT_BEFORE, (AFUNPTR)BeforeRealloc,
                       IARG_THREAD_ID,
                       IARG_FUNCARG_ENTRYPOINT_VALUE, 0,
                       IARG_FUNCARG_ENTRYPOINT_VALUE, 1,
                       IARG_END);
        RTN_InsertCall(reallocRtn, IPOINT_AFTER, (AFUNPTR)AfterRealloc,
                       IARG_THREAD_ID, IARG_FUNCRET_EXITPOINT_VALUE, IARG_END);
        RTN_Close(reallocRtn);
    }

    RTN freeRtn = RTN_FindByName(img, "free");
    if (RTN_Valid(freeRtn)) {
        RTN_Open(freeRtn);
        RTN_InsertCall(freeRtn, IPOINT_BEFORE, (AFUNPTR)BeforeFree,
                       IARG_FUNCARG_ENTRYPOINT_VALUE, 0, IARG_END);
        RTN_Close(freeRtn);
    }

    if (KnobMainExecutableOnly.Value() && !IMG_IsMainExecutable(img)) {
        return;
    }

    for (SEC sec = IMG_SecHead(img); SEC_Valid(sec); sec = SEC_Next(sec)) {
        for (RTN rtn = SEC_RtnHead(sec); RTN_Valid(rtn); rtn = RTN_Next(rtn)) {
            RTN_Open(rtn);

            ADDRINT rtn_addr = RTN_Address(rtn);

            RTN_InsertCall(rtn, IPOINT_BEFORE, (AFUNPTR)OnRoutineEntry,
                           IARG_THREAD_ID,
                           IARG_ADDRINT, rtn_addr,
                           IARG_FUNCARG_ENTRYPOINT_VALUE, 0,
                           IARG_FUNCARG_ENTRYPOINT_VALUE, 1,
                           IARG_FUNCARG_ENTRYPOINT_VALUE, 2,
                           IARG_FUNCARG_ENTRYPOINT_VALUE, 3,
                           IARG_FUNCARG_ENTRYPOINT_VALUE, 4,
                           IARG_FUNCARG_ENTRYPOINT_VALUE, 5,
                           IARG_END);

            RTN_InsertCall(rtn, IPOINT_AFTER, (AFUNPTR)OnRoutineExit,
                           IARG_THREAD_ID,
                           IARG_ADDRINT, rtn_addr,
                           IARG_END);

            RTN_Close(rtn);
        }
    }
}

VOID OnBblExecution(THREADID tid, UINT32 num_ins, UINT32 compute_ins, UINT32 mem_ins) {
    if (KnobStartFF.Value() && !g_in_roi) return;

    ThreadData* tdata = static_cast<ThreadData*>(PIN_GetThreadData(tls_key, tid));
    if (!tdata) return;

    tdata->thread_instructions += num_ins;
    tdata->thread_compute_insts += compute_ins;
    tdata->thread_mem_insts += mem_ins;

    if (!tdata->call_stack.empty()) {
        CallFrame& frame = tdata->call_stack.back();
        frame.instructions += num_ins;
        frame.compute_insts += compute_ins;
        frame.mem_insts += mem_ins;
    }
}

VOID Trace(TRACE trace, VOID *v) {
    for (BBL bbl = TRACE_BblHead(trace); BBL_Valid(bbl); bbl = BBL_Next(bbl)) {
        UINT32 num_ins = BBL_NumIns(bbl);
        UINT32 mem_ins = 0;
        for (INS ins = BBL_InsHead(bbl); INS_Valid(ins); ins = INS_Next(ins)) {
            if (INS_IsMemoryRead(ins) || INS_IsMemoryWrite(ins)) {
                mem_ins++;
            }
        }
        UINT32 compute_ins = num_ins - mem_ins;

        BBL_InsertCall(bbl, IPOINT_BEFORE, (AFUNPTR)OnBblExecution,
                       IARG_THREAD_ID,
                       IARG_UINT32, num_ins,
                       IARG_UINT32, compute_ins,
                       IARG_UINT32, mem_ins,
                       IARG_END);
    }
}

VOID Instruction(INS ins, VOID *v) {
    // Intercept ROI magic instructions (xchg %rbx, %rbx)
    if (INS_IsXchg(ins) && INS_OperandReg(ins, 0) == REG_RBX && INS_OperandReg(ins, 1) == REG_RBX) {
        INS_InsertCall(ins, IPOINT_BEFORE, (AFUNPTR)OnMagicInstruction,
                       IARG_THREAD_ID, IARG_REG_VALUE, REG_RCX, IARG_END);
    }

    UINT32 memOperands = INS_MemoryOperandCount(ins);
    for (UINT32 memOp = 0; memOp < memOperands; memOp++) {
        BOOL is_write = INS_MemoryOperandIsWritten(ins, memOp);
        INS_InsertPredicatedCall(
            ins, IPOINT_BEFORE, (AFUNPTR)OnMemoryAccess,
            IARG_THREAD_ID,
            IARG_MEMORYOP_EA, memOp,
            IARG_MEMORYOP_SIZE, memOp,
            IARG_BOOL, is_write,
            IARG_END);
    }
}

// =====================================================================
// Lifecycle Callbacks
// =====================================================================

VOID ThreadStart(THREADID tid, CONTEXT *ctxt, INT32 flags, VOID *v) {
    ThreadData* tdata = new ThreadData();

    // Initialize per-thread private L1 and L2 caches
    UINT64 l1_bytes = (UINT64)(KnobL1SizeMB.Value() * 1024.0 * 1024.0);
    UINT64 l2_bytes = (UINT64)(KnobL2SizeMB.Value() * 1024.0 * 1024.0);
    tdata->private_l1.Init("L1", l1_bytes, KnobL1Assoc.Value());
    tdata->private_l2.Init("L2", l2_bytes, KnobL2Assoc.Value());

    PIN_GetLock(&g_threads_list_lock, tid + 1);
    g_active_threads.push_back(tdata);
    PIN_ReleaseLock(&g_threads_list_lock);

    PIN_SetThreadData(tls_key, tdata, tid);
}

VOID ThreadFini(THREADID tid, const CONTEXT *ctxt, INT32 code, VOID *v) {
    ThreadData* tdata = static_cast<ThreadData*>(PIN_GetThreadData(tls_key, tid));
    if (tdata) {
        PIN_GetLock(&g_threads_list_lock, tid + 1);
        auto it = std::find(g_active_threads.begin(), g_active_threads.end(), tdata);
        if (it != g_active_threads.end()) {
            g_active_threads.erase(it);
        }
        PIN_ReleaseLock(&g_threads_list_lock);

        PIN_GetLock(&g_stats_lock, tid + 1);
        g_total_instructions += tdata->thread_instructions;
        g_total_compute_insts += tdata->thread_compute_insts;
        g_total_mem_insts += tdata->thread_mem_insts;
        g_total_mem_ops += tdata->thread_mem_ops;
        g_global_l1_hits += tdata->thread_l1_hits;
        g_global_l1_misses += tdata->thread_l1_misses;
        g_global_l2_hits += tdata->thread_l2_hits;
        g_global_l2_misses += tdata->thread_l2_misses;
        PIN_ReleaseLock(&g_stats_lock);

        delete tdata;
        PIN_SetThreadData(tls_key, NULL, tid);
    }
}

// =====================================================================
// CSV Export Helpers
// =====================================================================

static bool MakeDirRecursive(const std::string& path) {
    if (path.empty()) return false;
    struct stat st;
    if (stat(path.c_str(), &st) == 0) {
        return S_ISDIR(st.st_mode);
    }
    size_t pos = path.find_last_of("/\\");
    if (pos != std::string::npos && pos > 0) {
        MakeDirRecursive(path.substr(0, pos));
    }
    return (mkdir(path.c_str(), 0755) == 0) || (stat(path.c_str(), &st) == 0 && S_ISDIR(st.st_mode));
}

static void EnsureParentDir(const std::string& filepath) {
    size_t pos = filepath.find_last_of("/\\");
    if (pos != std::string::npos && pos > 0) {
        MakeDirRecursive(filepath.substr(0, pos));
    }
}

static std::string SanitizeFileName(const std::string& name) {
    std::string clean = name;
    for (size_t i = 0; i < clean.length(); ++i) {
        char c = clean[i];
        if (!((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') || c == '_')) {
            clean[i] = '_';
        }
    }
    return clean.empty() ? "unnamed_func" : clean;
}

static void WriteCsvHeader(std::ostream& csv) {
    csv << "func_name,call_hash,invocations,"
        << "arg0,arg1,arg2,arg3,arg4,arg5,"
        << "avg_instructions,total_instructions,"
        << "avg_cycles,avg_ndp_cycles,total_host_cycles,"
        << "cycle_fraction_of_total_app,cycle_pct_of_total_app,"
        << "avg_mem_ops,avg_l1_misses,l1_miss_pct,avg_l2_misses,l2_miss_pct,avg_llc_misses,llc_miss_pct,"
        << "avg_dram_reads,avg_dram_writes,avg_total_dram,dram_pki,"
        << "avg_mem_instructions,avg_compute_instructions,mem_compute_ratio,"
        << "host_ipc,ndp_ipc,kernel_speedup,app_speedup\n";
}

static void WriteCsvRow(std::ostream& csv, const FunctionDramStats& entry, double total_app_host_cycles) {
    std::string demangled = PIN_UndecorateSymbolName(entry.sig.func_name, UNDECORATION_NAME_ONLY);
    std::string fn = demangled.empty() ? entry.sig.func_name : demangled;

    double invocations = entry.invocations > 0 ? (double)entry.invocations : 1.0;

    double avg_insts = (double)entry.total_instructions / invocations;
    double avg_mem_ops = (double)entry.total_mem_ops / invocations;
    double avg_l1_misses = (double)entry.excl_l1_misses / invocations;
    double avg_l2_misses = (double)entry.excl_l2_misses / invocations;
    double avg_llc_misses = (double)entry.excl_llc_misses / invocations;
    double avg_dram_reads = (double)entry.excl_dram_reads / invocations;
    double avg_dram_writes = (double)entry.excl_dram_writes / invocations;
    double avg_total_dram = (double)entry.excl_total_dram / invocations;
    double avg_mem_insts = (double)entry.total_mem_insts / invocations;
    double avg_compute_insts = (double)entry.total_compute_insts / invocations;

    double l1_miss_pct = entry.total_mem_ops ? (double)entry.excl_l1_misses * 100.0 / entry.total_mem_ops : 0.0;
    double l2_miss_pct = entry.total_mem_ops ? (double)entry.excl_l2_misses * 100.0 / entry.total_mem_ops : 0.0;
    double llc_miss_pct = entry.total_mem_ops ? (double)entry.excl_llc_misses * 100.0 / entry.total_mem_ops : 0.0;

    double func_host_cycles = (double)entry.total_instructions * KnobHostBaseCpi.Value()
                            + (double)entry.excl_l1_misses * KnobL2Latency.Value()
                            + (double)entry.excl_l2_misses * KnobLlcLatency.Value()
                            + (double)entry.excl_dram_reads * (KnobDramLocalLatency.Value() + KnobCxlLinkLatency.Value())
                            + (double)entry.excl_dram_writes * KnobDramLocalLatency.Value();

    double func_ndp_cycles  = (double)entry.total_instructions * KnobNdpBaseCpi.Value()
                            + (double)entry.excl_l1_misses * KnobL2Latency.Value()
                            + (double)entry.excl_l2_misses * KnobLlcLatency.Value()
                            + (double)entry.excl_dram_reads * KnobDramLocalLatency.Value()
                            + (double)entry.excl_dram_writes * KnobDramLocalLatency.Value();

    double avg_cycles = func_host_cycles / invocations;
    double avg_ndp_cycles = func_ndp_cycles / invocations;

    double dram_pki = entry.total_instructions > 0 ? ((double)entry.excl_total_dram * 1000.0) / (double)entry.total_instructions : 0.0;
    double mem_compute_ratio = entry.total_compute_insts > 0 ? (double)entry.total_mem_insts / (double)entry.total_compute_insts : (double)entry.total_mem_insts;
    double host_ipc = func_host_cycles > 0.0 ? (double)entry.total_instructions / func_host_cycles : 0.0;
    double ndp_ipc  = func_ndp_cycles > 0.0 ? (double)entry.total_instructions / func_ndp_cycles : 0.0;

    double cycle_fraction = total_app_host_cycles > 0.0 ? func_host_cycles / total_app_host_cycles : 0.0;
    double cycle_pct = total_app_host_cycles > 0.0 ? (func_host_cycles * 100.0) / total_app_host_cycles : 0.0;
    double kernel_speedup = func_ndp_cycles > 0.0 ? func_host_cycles / func_ndp_cycles : 1.0;

    double f = cycle_fraction;
    if (f > 1.0) f = 1.0;
    if (f < 0.0) f = 0.0;

    double app_speedup = 1.0;
    if (kernel_speedup > 0.0) {
        double denom = (1.0 - f) + (f / kernel_speedup);
        if (denom > 0.0) {
            app_speedup = 1.0 / denom;
        }
    }

    csv << "\"" << fn << "\","
        << "0x" << std::hex << entry.sig.hash_val << std::dec << ","
        << entry.invocations << ",";

    for (size_t i = 0; i < 6; ++i) {
        if (i < entry.sig.args.size()) {
            csv << "0x" << std::hex << entry.sig.args[i] << std::dec << ",";
        } else {
            csv << "0x0,";
        }
    }

    csv << std::fixed << std::setprecision(1) << avg_insts << ","
        << entry.total_instructions << ","
        << std::fixed << std::setprecision(1) << avg_cycles << ","
        << avg_ndp_cycles << ","
        << std::setprecision(0) << func_host_cycles << ","
        << std::setprecision(6) << cycle_fraction << ","
        << std::setprecision(2) << cycle_pct << ","
        << std::setprecision(1) << avg_mem_ops << ","
        << avg_l1_misses << ","
        << std::setprecision(2) << l1_miss_pct << ","
        << std::setprecision(1) << avg_l2_misses << ","
        << std::setprecision(2) << l2_miss_pct << ","
        << std::setprecision(1) << avg_llc_misses << ","
        << std::setprecision(2) << llc_miss_pct << ","
        << std::setprecision(1) << avg_dram_reads << ","
        << avg_dram_writes << ","
        << avg_total_dram << ","
        << std::setprecision(3) << dram_pki << ","
        << std::setprecision(1) << avg_mem_insts << ","
        << avg_compute_insts << ","
        << std::setprecision(4) << mem_compute_ratio << ","
        << host_ipc << ","
        << ndp_ipc << ","
        << std::setprecision(3) << kernel_speedup << ","
        << app_speedup << "\n";
}

VOID Fini(INT32 code, VOID *v) {
    EnsureParentDir(KnobOutputFile.Value());
    std::ofstream out(KnobOutputFile.Value().c_str());
    if (!out.is_open()) {
        std::cerr << "Warning: Could not open output file: " << KnobOutputFile.Value() << "\n";
    }

    std::string inc_str = KnobFuncFilter.Value();
    if (inc_str.empty()) inc_str = KnobFilterAlias.Value();
    std::vector<std::string> includes = SplitCommaList(inc_str);
    std::vector<std::string> excludes = SplitCommaList(KnobExcludeFuncs.Value());
    bool hide_boilerplate = KnobHideBoilerplate.Value();
    bool exact_match = KnobExactMatch.Value();

    // Accumulate any remaining active threads that did not trigger ThreadFini
    PIN_GetLock(&g_threads_list_lock, 1);
    for (ThreadData* tdata : g_active_threads) {
        g_total_instructions += tdata->thread_instructions;
        g_total_compute_insts += tdata->thread_compute_insts;
        g_total_mem_insts += tdata->thread_mem_insts;
        g_total_mem_ops += tdata->thread_mem_ops;
        g_global_l1_hits += tdata->thread_l1_hits;
        g_global_l1_misses += tdata->thread_l1_misses;
        g_global_l2_hits += tdata->thread_l2_hits;
        g_global_l2_misses += tdata->thread_l2_misses;
    }
    g_active_threads.clear();
    PIN_ReleaseLock(&g_threads_list_lock);

    // Analytical Model Whole Application Totals
    double total_app_host_cycles = (double)g_total_instructions * KnobHostBaseCpi.Value()
                                 + (double)g_global_l1_misses * KnobL2Latency.Value()
                                 + (double)g_global_l2_misses * KnobLlcLatency.Value()
                                 + (double)g_global_dram_reads * (KnobDramLocalLatency.Value() + KnobCxlLinkLatency.Value())
                                 + (double)g_global_dram_writes * KnobDramLocalLatency.Value();

    double total_app_ndp_cycles  = (double)g_total_instructions * KnobNdpBaseCpi.Value()
                                 + (double)g_global_l1_misses * KnobL2Latency.Value()
                                 + (double)g_global_l2_misses * KnobLlcLatency.Value()
                                 + (double)g_global_dram_reads * KnobDramLocalLatency.Value()
                                 + (double)g_global_dram_writes * KnobDramLocalLatency.Value();

    if (total_app_host_cycles < 1.0) total_app_host_cycles = 1.0;
    if (total_app_ndp_cycles < 1.0) total_app_ndp_cycles = 1.0;
    double app_host_ipc = (double)g_total_instructions / total_app_host_cycles;
    double app_ndp_ipc  = (double)g_total_instructions / total_app_ndp_cycles;
    double app_dram_pki = g_total_instructions > 0 ? ((double)(g_global_dram_reads + g_global_dram_writes) * 1000.0) / (double)g_total_instructions : 0.0;

    out << "===================================================================================================================================================\n";
    out << "                                        CACHE HIERARCHY SIMULATION & DRAM ACCESS REPORT                                                            \n";
    out << "===================================================================================================================================================\n";
    out << "CACHE ARCHITECTURE MODEL:\n";
    out << "  L1 Data Cache: " << std::fixed << std::setprecision(3) << KnobL1SizeMB.Value() << " MB per thread ("
        << (UINT64)(KnobL1SizeMB.Value() * 1024.0) << " KB), " << KnobL1Assoc.Value() << "-way associative (64B line)\n";
    out << "  L2 Cache:      " << KnobL2SizeMB.Value() << " MB per thread ("
        << (UINT64)(KnobL2SizeMB.Value() * 1024.0) << " KB), " << KnobL2Assoc.Value() << "-way associative (64B line)\n";
    out << "  Shared LLC:    " << KnobLlcSizeMB.Value() << " MB, " << KnobLlcAssoc.Value() << "-way associative ("
        << g_shared_llc.num_sets << " sets, 64B line)\n";
    out << "  Target Memory: " << (KnobHeapOnly.Value() ? "Heap Allocations Only" : "All Data Memory Accesses") << "\n";
    out << "  Start Mode:    " << (KnobStartFF.Value() ? "Fast-Forward (ROI only via xchg %rbx, %rbx)" : "Whole Program") << "\n";
    out << "ANALYTICAL HARDWARE EXECUTION MODEL:\n";
    out << "  Host Architecture: 4-wide OOO Core (Base CPI=" << KnobHostBaseCpi.Value() << "), CXL Link Latency=" << KnobCxlLinkLatency.Value() << " cyc\n";
    out << "  NDP Architecture:  2-wide Core (Base CPI=" << KnobNdpBaseCpi.Value() << "), Local CXL Memory Latency=" << KnobDramLocalLatency.Value() << " cyc (Link=0 cyc)\n";

    if (!includes.empty() || !excludes.empty() || hide_boilerplate) {
        out << "FUNCTION FILTER SETTINGS:\n";
        if (!includes.empty()) {
            out << "  Include Filter:  ";
            for (size_t i = 0; i < includes.size(); ++i) out << (i ? ", " : "") << "\"" << includes[i] << "\"";
            out << "\n";
        }
        if (!excludes.empty()) {
            out << "  Exclude Filter:  ";
            for (size_t i = 0; i < excludes.size(); ++i) out << (i ? ", " : "") << "\"" << excludes[i] << "\"";
            out << "\n";
        }
        out << "  Match Mode:      " << (exact_match ? "Exact Match" : "Substring Match") << "\n";
        out << "  Boilerplate:     " << (hide_boilerplate ? "Hidden" : "Included") << "\n";
    }

    out << "---------------------------------------------------------------------------------------------------------------------------------------------------\n";
    out << std::left
        << std::setw(20) << "Call Hash"
        << std::setw(26) << "Function"
        << std::setw(24) << "Arguments"
        << std::setw(6)  << "Calls"
        << std::setw(12) << "Mem Ops"
        << std::setw(10) << "L1 Miss%"
        << std::setw(10) << "L2 Miss%"
        << std::setw(10) << "LLC Miss%"
        << std::setw(11) << "DRAM Reads"
        << std::setw(10) << "DRAM WBs"
        << std::setw(12) << "Total DRAM"
        << std::setw(12) << "Avg DRAM/C"
        << "Notes\n";
    out << "---------------------------------------------------------------------------------------------------------------------------------------------------\n";

    std::vector<FunctionDramStats> entries;
    entries.reserve(g_call_hash_map.size());

    for (const auto& pair : g_call_hash_map) {
        std::string raw_name = pair.second.sig.func_name;
        std::string demangled = PIN_UndecorateSymbolName(raw_name, UNDECORATION_NAME_ONLY);
        std::string check_name = demangled.empty() ? raw_name : demangled;

        if (RoutinePassesFilter(check_name, includes, excludes, hide_boilerplate, exact_match)) {
            entries.push_back(pair.second);
        }
    }

    std::sort(entries.begin(), entries.end(), [](const FunctionDramStats& a, const FunctionDramStats& b) {
        if (a.excl_total_dram != b.excl_total_dram)
            return a.excl_total_dram > b.excl_total_dram;
        return a.total_mem_ops > b.total_mem_ops;
    });

    for (const auto& entry : entries) {
        std::stringstream args_ss;
        args_ss << "(";
        for (size_t i = 0; i < entry.sig.args.size(); ++i) {
            args_ss << "0x" << std::hex << entry.sig.args[i];
            if (i + 1 < entry.sig.args.size()) args_ss << ", ";
        }
        args_ss << ")";

        std::string func_display = entry.sig.func_name;
        std::string demangled = PIN_UndecorateSymbolName(func_display, UNDECORATION_NAME_ONLY);
        if (!demangled.empty()) func_display = demangled;

        if (func_display.length() > 24) func_display = func_display.substr(0, 22) + "..";

        std::string args_display = args_ss.str();
        if (args_display.length() > 22) args_display = args_display.substr(0, 20) + "..";

        std::stringstream hash_ss;
        hash_ss << "0x" << std::hex << entry.sig.hash_val;

        double l1_miss_pct = entry.total_mem_ops ? (double)entry.excl_l1_misses * 100.0 / entry.total_mem_ops : 0.0;
        double l2_miss_pct = entry.total_mem_ops ? (double)entry.excl_l2_misses * 100.0 / entry.total_mem_ops : 0.0;
        double llc_miss_pct = entry.total_mem_ops ? (double)entry.excl_llc_misses * 100.0 / entry.total_mem_ops : 0.0;
        double avg_dram = entry.invocations ? (double)entry.excl_total_dram / entry.invocations : 0.0;

        std::string notes = "";
        if (entry.recursive_invocations > 0) {
            std::stringstream nss;
            nss << "[RECURSIVE d=" << entry.max_recursion_depth << "]";
            notes = nss.str();
        }

        out << std::left
            << std::setw(20) << hash_ss.str()
            << std::setw(26) << func_display
            << std::setw(24) << args_display
            << std::setw(6)  << std::dec << entry.invocations
            << std::setw(12) << entry.total_mem_ops
            << std::setw(10) << std::fixed << std::setprecision(1) << l1_miss_pct
            << std::setw(10) << l2_miss_pct
            << std::setw(10) << llc_miss_pct
            << std::setw(11) << entry.excl_dram_reads
            << std::setw(10) << entry.excl_dram_writes
            << std::setw(12) << entry.excl_total_dram
            << std::setw(12) << std::fixed << std::setprecision(1) << avg_dram
            << notes
            << "\n";
    }

    if (entries.empty()) {
        out << "  [No functions matched the specified filter criteria]\n";
    }

    // Secondary Table: Performance, Memory Intensity & NDP Offload Evaluation
    out << "\n====================================================================================================================================================================================\n";
    out << "                                        PERFORMANCE, MEMORY INTENSITY & NDP OFFLOAD EVALUATION                                                                      \n";
    out << "====================================================================================================================================================================================\n";
    out << std::left
        << std::setw(20) << "Call Hash"
        << std::setw(26) << "Function"
        << std::setw(12) << "Insts"
        << std::setw(10) << "Mem/Comp"
        << std::setw(10) << "DRAM-PKI"
        << std::setw(14) << "Host Cycles"
        << std::setw(10) << "Host IPC"
        << std::setw(10) << "% Cycles"
        << std::setw(14) << "NDP Cycles"
        << std::setw(10) << "NDP IPC"
        << std::setw(12) << "K-Speedup"
        << std::setw(12) << "App-Speedup"
        << "Recommendation\n";
    out << "------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------\n";

    for (const auto& entry : entries) {
        std::string func_display = entry.sig.func_name;
        std::string demangled = PIN_UndecorateSymbolName(func_display, UNDECORATION_NAME_ONLY);
        if (!demangled.empty()) func_display = demangled;
        if (func_display.length() > 24) func_display = func_display.substr(0, 22) + "..";

        std::stringstream hash_ss;
        hash_ss << "0x" << std::hex << entry.sig.hash_val;

        double func_host_cycles = (double)entry.total_instructions * KnobHostBaseCpi.Value()
                                + (double)entry.excl_l1_misses * KnobL2Latency.Value()
                                + (double)entry.excl_l2_misses * KnobLlcLatency.Value()
                                + (double)entry.excl_dram_reads * (KnobDramLocalLatency.Value() + KnobCxlLinkLatency.Value())
                                + (double)entry.excl_dram_writes * KnobDramLocalLatency.Value();

        double func_ndp_cycles  = (double)entry.total_instructions * KnobNdpBaseCpi.Value()
                                + (double)entry.excl_l1_misses * KnobL2Latency.Value()
                                + (double)entry.excl_l2_misses * KnobLlcLatency.Value()
                                + (double)entry.excl_dram_reads * KnobDramLocalLatency.Value()
                                + (double)entry.excl_dram_writes * KnobDramLocalLatency.Value();

        double dram_pki = entry.total_instructions > 0 ? ((double)entry.excl_total_dram * 1000.0) / (double)entry.total_instructions : 0.0;
        double mem_compute_ratio = entry.total_compute_insts > 0 ? (double)entry.total_mem_insts / (double)entry.total_compute_insts : (double)entry.total_mem_insts;
        double host_ipc = func_host_cycles > 0.0 ? (double)entry.total_instructions / func_host_cycles : 0.0;
        double ndp_ipc  = func_ndp_cycles > 0.0 ? (double)entry.total_instructions / func_ndp_cycles : 0.0;
        double cycle_pct = (func_host_cycles * 100.0) / total_app_host_cycles;
        double kernel_speedup = func_ndp_cycles > 0.0 ? func_host_cycles / func_ndp_cycles : 1.0;

        double f = func_host_cycles / total_app_host_cycles;
        if (f > 1.0) f = 1.0;
        if (f < 0.0) f = 0.0;

        double app_speedup = 1.0;
        if (kernel_speedup > 0.0) {
            double denom = (1.0 - f) + (f / kernel_speedup);
            if (denom > 0.0) {
                app_speedup = 1.0 / denom;
            }
        }

        std::string rec = "HOST_CPU";
        double avg_dram = entry.invocations ? (double)entry.excl_total_dram / entry.invocations : 0.0;
        if (kernel_speedup >= 1.05 && dram_pki >= 10.0 && avg_dram >= 50.0) {
            std::stringstream ss;
            ss << "NDP_OFFLOAD (" << std::fixed << std::setprecision(2) << kernel_speedup << "x)";
            rec = ss.str();
        } else if (kernel_speedup < 0.95) {
            rec = "HOST_CPU (Compute bound)";
        } else {
            rec = "HOST_CPU (Cache/Low DRAM)";
        }

        out << std::left
            << std::setw(20) << hash_ss.str()
            << std::setw(26) << func_display
            << std::setw(12) << entry.total_instructions
            << std::setw(10) << std::fixed << std::setprecision(2) << mem_compute_ratio
            << std::setw(10) << std::fixed << std::setprecision(2) << dram_pki
            << std::setw(14) << (UINT64)func_host_cycles
            << std::setw(10) << std::fixed << std::setprecision(3) << host_ipc
            << std::setw(10) << std::fixed << std::setprecision(2) << cycle_pct
            << std::setw(14) << (UINT64)func_ndp_cycles
            << std::setw(10) << std::fixed << std::setprecision(3) << ndp_ipc
            << std::setw(12) << std::fixed << std::setprecision(3) << kernel_speedup
            << std::setw(12) << std::fixed << std::setprecision(3) << app_speedup
            << rec
            << "\n";
    }

    out << "------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------\n";
    out << "GLOBAL EXECUTION & CYCLE MODEL TOTALS (Active Period):\n";
    out << "  Total Application Instructions:     " << g_total_instructions << "\n";
    out << "    - Compute Instructions:           " << g_total_compute_insts
        << " (" << std::fixed << std::setprecision(1) << (g_total_instructions ? (double)g_total_compute_insts * 100.0 / g_total_instructions : 0.0) << "%)\n";
    out << "    - Memory Instructions:            " << g_total_mem_insts
        << " (" << std::fixed << std::setprecision(1) << (g_total_instructions ? (double)g_total_mem_insts * 100.0 / g_total_instructions : 0.0) << "%)\n";
    out << "    - Memory / Compute Ratio:         " << std::fixed << std::setprecision(3)
        << (g_total_compute_insts ? (double)g_total_mem_insts / (double)g_total_compute_insts : (double)g_total_mem_insts) << "\n";
    out << "  Total Simulated Memory Operations:  " << g_total_mem_ops << "\n";
    out << "  L1 Hits:                            " << g_global_l1_hits << " ("
        << std::fixed << std::setprecision(2)
        << (g_total_mem_ops ? (double)g_global_l1_hits * 100.0 / g_total_mem_ops : 0.0) << "%)\n";
    out << "  L1 Misses:                          " << g_global_l1_misses << "\n";
    out << "  L2 Hits:                            " << g_global_l2_hits << "\n";
    out << "  L2 Misses:                          " << g_global_l2_misses << "\n";
    out << "  Shared LLC Hits:                    " << g_global_llc_hits << "\n";
    out << "  LLC Misses (DRAM Reads):            " << g_global_dram_reads << "\n";
    out << "  LLC Dirty Evictions (WBs):          " << g_global_dram_writes << "\n";
    out << "  TOTAL DRAM ACCESSES:                " << (g_global_dram_reads + g_global_dram_writes) << "\n";
    out << "  GLOBAL MEMORY INTENSITY (DRAM-PKI): " << std::fixed << std::setprecision(2) << app_dram_pki << " DRAM accesses / 1000 instructions\n";
    out << "  ANALYTICAL EXECUTION CYCLE TOTALS:\n";
    out << "    - Host Execution Cycles:          " << (UINT64)total_app_host_cycles << " (IPC: " << std::fixed << std::setprecision(3) << app_host_ipc << ")\n";
    out << "    - NDP Execution Cycles:           " << (UINT64)total_app_ndp_cycles  << " (IPC: " << std::fixed << std::setprecision(3) << app_ndp_ipc  << ")\n";
    out << "    - System-Wide Potential Speedup:  " << std::fixed << std::setprecision(3) << (total_app_host_cycles / total_app_ndp_cycles) << "x\n";
    out << "====================================================================================================================================================================================\n";
    out.close();

    // Export single consolidated CSV report if requested
    if (!KnobCsvFile.Value().empty()) {
        EnsureParentDir(KnobCsvFile.Value());
        std::ofstream csv(KnobCsvFile.Value().c_str());
        if (csv.is_open()) {
            WriteCsvHeader(csv);
            for (const auto& entry : entries) {
                WriteCsvRow(csv, entry, total_app_host_cycles);
            }
            csv.close();
            std::cerr << "CSV report written to: " << KnobCsvFile.Value() << "\n";
        }
    }

    // Export per-function CSV reports under output directory if requested
    if (!KnobCsvDir.Value().empty()) {
        std::string csv_dir = KnobCsvDir.Value();
        MakeDirRecursive(csv_dir);

        std::map<std::string, std::vector<const FunctionDramStats*>> func_groups;
        for (const auto& entry : entries) {
            std::string demangled = PIN_UndecorateSymbolName(entry.sig.func_name, UNDECORATION_NAME_ONLY);
            std::string fn = demangled.empty() ? entry.sig.func_name : demangled;
            std::string safe_name = SanitizeFileName(fn);
            func_groups[safe_name].push_back(&entry);
        }

        for (const auto& kv : func_groups) {
            std::string file_path = csv_dir + "/" + kv.first + ".csv";
            std::ofstream fcsv(file_path.c_str());
            if (fcsv.is_open()) {
                WriteCsvHeader(fcsv);
                for (const auto* p_entry : kv.second) {
                    WriteCsvRow(fcsv, *p_entry, total_app_host_cycles);
                }
                fcsv.close();
            } else {
                std::cerr << "Warning: Failed to open per-function CSV file: " << file_path << "\n";
            }
        }
        std::cerr << "Per-function CSV reports written to: " << csv_dir << "/ (" << func_groups.size() << " functions)\n";
    }

    std::cerr << "Simulation report written to: " << KnobOutputFile.Value() << "\n";
}

// =====================================================================
// Main Entry
// =====================================================================

int main(int argc, char *argv[]) {
    PIN_InitSymbols();

    if (PIN_Init(argc, argv)) {
        std::cerr << "PIN Initialization failed.\n";
        return 1;
    }

    // Set ROI initial state: if startFF requested, wait for magic instruction
    g_in_roi = !KnobStartFF.Value();

    // Initialize shared LLC
    UINT64 llc_bytes = (UINT64)(KnobLlcSizeMB.Value() * 1024.0 * 1024.0);
    g_shared_llc.Init("LLC", llc_bytes, KnobLlcAssoc.Value());

    ParseProcMaps();
    PIN_InitLock(&g_llc_lock);
    PIN_InitLock(&g_stats_lock);
    PIN_InitLock(&g_heap_lock);
    PIN_InitLock(&g_threads_list_lock);
    tls_key = PIN_CreateThreadDataKey(NULL);

    PIN_AddThreadStartFunction(ThreadStart, 0);
    PIN_AddThreadFiniFunction(ThreadFini, 0);
    IMG_AddInstrumentFunction(ImageLoad, 0);
    INS_AddInstrumentFunction(Instruction, 0);
    TRACE_AddInstrumentFunction(Trace, 0);
    PIN_AddFiniFunction(Fini, 0);

    PIN_StartProgram();

    return 0;
}
