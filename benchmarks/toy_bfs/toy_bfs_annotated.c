// ==========================================================================
// Annotated Source Code: toy_bfs.c
// Total Instructions Sampled: 100,000
// Columns: [ Execution Count | Share % | Loads | Stores ] Line: Source
// ==========================================================================

[                   |                      ]    1: /*
[                   |                      ]    2:  * toy_bfs.c  –  Stripped-down BFS microbenchmark modeled after GAPBS
[                   |                      ]    3:  *
[                   |                      ]    4:  * Captures the two dominant memory-access patterns of direction-optimizing BFS:
[                   |                      ]    5:  *
[                   |                      ]    6:  *   Top-Down  (TD): for each frontier node u, walk out_nbr[u] (sequential
[                   |                      ]    7:  *                   within the CSR block for u, scattered across nodes), then
[                   |                      ]    8:  *                   scatter-write to parent[v]  –  highly RANDOM load+store.
[                   |                      ]    9:  *
[                   |                      ]   10:  *   Bottom-Up (BU): sequential sweep of all nodes; for each unvisited node u
[                   |                      ]   11:  *                   walk in_nbr[u] (sequential CSR) and check bitmap[v] –
[                   |                      ]   12:  *                   random bitmap read, sequential parent[] read.
[                   |                      ]   13:  *
[                   |                      ]   14:  * Direction switching is identical to GAPBS (alpha=15, beta=18).
[                   |                      ]   15:  *
[                   |                      ]   16:  * ROI markers (xchg rbx,rbx) are compatible with the StarNUMA pin tracer.
[                   |                      ]   17:  * Run under PIN with -startFF so graph construction is excluded from traces.
[                   |                      ]   18:  *
[                   |                      ]   19:  * Build:
[                   |                      ]   20:  *   gcc -O2 -o toy_bfs toy_bfs.c          # single-threaded (for tracing)
[                   |                      ]   21:  *   gcc -O2 -DSCALE=18 -o toy_bfs toy_bfs.c   # custom graph size
[                   |                      ]   22:  *
[                   |                      ]   23:  * Run:
[                   |                      ]   24:  *   ./toy_bfs [scale] [avg_degree] [seed]
[                   |                      ]   25:  *   ./toy_bfs 16 16 42     # 64K nodes, avg-degree 16  (~4 MB working set)
[                   |                      ]   26:  *   ./toy_bfs 18 16 42     # 256K nodes               (~18 MB working set)
[                   |                      ]   27:  */
[                   |                      ]   28: 
[                   |                      ]   29: #include <stdio.h>
[                   |                      ]   30: #include <stdlib.h>
[                   |                      ]   31: #include <stdint.h>
[                   |                      ]   32: #include <string.h>
[                   |                      ]   33: 
[                   |                      ]   34: /* ============================================================
[                   |                      ]   35:  *  Direction-switching thresholds (identical to GAPBS defaults)
[                   |                      ]   36:  * ============================================================ */
[                   |                      ]   37: #define ALPHA 15
[                   |                      ]   38: #define BETA  18
[                   |                      ]   39: 
[                   |                      ]   40: /* ============================================================
[                   |                      ]   41:  *  ROI magic instructions
[                   |                      ]   42:  *  The pin tracer (combined_tracer.cpp) intercepts 'xchg rbx,rbx'
[                   |                      ]   43:  *  and dispatches on rcx:
[                   |                      ]   44:  *    rcx=0x5  → print instruction count (debug)
[                   |                      ]   45:  *    rcx=0x2  → ROI_START_MASTER (start tracing all threads)
[                   |                      ]   46:  *    rcx=0x4  → ROI_END per-thread
[                   |                      ]   47:  * ============================================================ */
[                   |                      ]   48: static inline void roi_start(void)
[                   |                      ]   49: {
[                   |                      ]   50:     volatile uint64_t d = 0;
[                   |                      ]   51:     /* debug: print current instruction count */
[                   |                      ]   52:     __asm__ __volatile__("xchg %%rbx, %%rbx;"
[                   |                      ]   53:         ::"b"((uint64_t)(&d)), "c"((uint64_t)0x5));
[                   |                      ]   54:     /* signal master ROI start */
[         1 ( 0.00%) | L:       0 S:       0]   55:     __asm__ __volatile__("xchg %%rbx, %%rbx;"
[                   |                      ]   56:         ::"b"((uint64_t)(&d)), "c"((uint64_t)0x2));
[                   |                      ]   57: }
[                   |                      ]   58: 
[                   |                      ]   59: static inline void roi_end(void)
[                   |                      ]   60: {
[                   |                      ]   61:     volatile uint64_t d = 0;
[                   |                      ]   62:     __asm__ __volatile__("xchg %%rbx, %%rbx;"
[         1 ( 0.00%) | L:       0 S:       1]   63:         ::"b"((uint64_t)(&d)), "c"((uint64_t)0x4));
[                   |                      ]   64: }
[                   |                      ]   65: 
[                   |                      ]   66: /* ============================================================
[                   |                      ]   67:  *  Graph storage (CSR, both forward and backward adjacency)
[                   |                      ]   68:  *
[                   |                      ]   69:  *  out_idx[u] .. out_idx[u+1]-1  →  out-neighbors of u  (for TD step)
[                   |                      ]   70:  *  in_idx[u]  .. in_idx[u+1]-1   →  in-neighbors  of u  (for BU step)
[                   |                      ]   71:  * ============================================================ */
[                   |                      ]   72: static uint32_t  g_n;           /* number of nodes */
[                   |                      ]   73: static uint64_t  g_e;           /* total directed edges in each CSR */
[                   |                      ]   74: 
[                   |                      ]   75: static uint32_t *out_idx;       /* [g_n + 1]  CSR row offsets  (out) */
[                   |                      ]   76: static uint32_t *out_nbr;       /* [g_e]      CSR column values (out) */
[                   |                      ]   77: static uint32_t *in_idx;        /* [g_n + 1]  CSR row offsets  (in)  */
[                   |                      ]   78: static uint32_t *in_nbr;        /* [g_e]      CSR column values (in)  */
[                   |                      ]   79: 
[                   |                      ]   80: /* ============================================================
[                   |                      ]   81:  *  BFS state
[                   |                      ]   82:  * ============================================================ */
[                   |                      ]   83: static int32_t  *g_parent;      /* [g_n]  -out_degree(v) if unvisited, else parent id */
[                   |                      ]   84: static uint32_t *g_qbuf;        /* [2*g_n]  two halves: current and next frontier queue */
[                   |                      ]   85: static uint64_t *g_bmap[2];     /* two bitmaps for BU direction (front / next) */
[                   |                      ]   86: 
[                   |                      ]   87: /* ============================================================
[                   |                      ]   88:  *  Bitmap helpers
[                   |                      ]   89:  * ============================================================ */
[                   |                      ]   90: static inline int bm_get(const uint64_t *bm, uint32_t i)
[                   |                      ]   91: {
[                   |                      ]   92:     return (int)((bm[i >> 6] >> (i & 63)) & 1u);
[                   |                      ]   93: }
[                   |                      ]   94: static inline void bm_set(uint64_t *bm, uint32_t i)
[                   |                      ]   95: {
[                   |                      ]   96:     bm[i >> 6] |= (uint64_t)1u << (i & 63);
[                   |                      ]   97: }
[                   |                      ]   98: static inline void bm_clear_all(uint64_t *bm)
[                   |                      ]   99: {
[                   |                      ]  100:     memset(bm, 0, (((size_t)g_n + 63) >> 6) * sizeof(uint64_t));
[                   |                      ]  101: }
[                   |                      ]  102: 
[                   |                      ]  103: /* ============================================================
[                   |                      ]  104:  *  Edge type used during graph construction only
[                   |                      ]  105:  * ============================================================ */
[                   |                      ]  106: typedef struct { uint32_t u, v; } Edge;
[                   |                      ]  107: 
[                   |                      ]  108: static int cmp_by_u(const void *a, const void *b)
[                   |                      ]  109: {
[                   |                      ]  110:     const Edge *ea = (const Edge *)a, *eb = (const Edge *)b;
[                   |                      ]  111:     if (ea->u != eb->u) return (ea->u < eb->u) ? -1 : 1;
[                   |                      ]  112:     return (ea->v < eb->v) ? -1 : (ea->v > eb->v) ? 1 : 0;
[                   |                      ]  113: }
[                   |                      ]  114: static int cmp_by_v(const void *a, const void *b)
[                   |                      ]  115: {
[                   |                      ]  116:     const Edge *ea = (const Edge *)a, *eb = (const Edge *)b;
[                   |                      ]  117:     if (ea->v != eb->v) return (ea->v < eb->v) ? -1 : 1;
[                   |                      ]  118:     return (ea->u < eb->u) ? -1 : (ea->u > eb->u) ? 1 : 0;
[                   |                      ]  119: }
[                   |                      ]  120: 
[                   |                      ]  121: /* ============================================================
[                   |                      ]  122:  *  Graph generation
[                   |                      ]  123:  *
[                   |                      ]  124:  *  Generates a random undirected graph:
[                   |                      ]  125:  *    - n_raw = n * avg_deg / 2 undirected edges  (random (u,v) pairs)
[                   |                      ]  126:  *    - Symmetrized: each undirected edge becomes two directed edges
[                   |                      ]  127:  *    - Self-loops are skipped
[                   |                      ]  128:  *  Builds both out-CSR and in-CSR for direction-optimized BFS.
[                   |                      ]  129:  * ============================================================ */
[                   |                      ]  130: static void build_graph(uint32_t n, int avg_deg, unsigned seed)
[                   |                      ]  131: {
[                   |                      ]  132:     g_n = n;
[                   |                      ]  133:     uint64_t n_raw = (uint64_t)n * (uint64_t)avg_deg / 2; /* undirected edges */
[                   |                      ]  134:     uint64_t n_sym = n_raw * 2;                            /* directed (symmetrized) */
[                   |                      ]  135:     g_e = n_sym;
[                   |                      ]  136: 
[                   |                      ]  137:     printf("[graph] Generating %llu undirected edges "
[                   |                      ]  138:            "(%u nodes, avg_deg~%d)...\n",
[                   |                      ]  139:            (unsigned long long)n_raw, n, avg_deg);
[                   |                      ]  140: 
[                   |                      ]  141:     /* Temporary edge list (freed after CSR is built) */
[                   |                      ]  142:     Edge *edges = (Edge *)malloc(n_sym * sizeof(Edge));
[                   |                      ]  143:     if (!edges) { perror("malloc edges"); exit(1); }
[                   |                      ]  144: 
[                   |                      ]  145:     srand(seed);
[                   |                      ]  146:     for (uint64_t i = 0; i < n_raw; i++) {
[                   |                      ]  147:         uint32_t u = (uint32_t)((unsigned)rand() % n);
[                   |                      ]  148:         uint32_t v = (uint32_t)((unsigned)rand() % n);
[                   |                      ]  149:         if (u == v) v = (v + 1) % n; /* no self-loops */
[                   |                      ]  150:         edges[2*i]   = (Edge){u, v};
[                   |                      ]  151:         edges[2*i+1] = (Edge){v, u}; /* reverse for bidirectional graph */
[                   |                      ]  152:     }
[                   |                      ]  153: 
[                   |                      ]  154:     /* ---- Build out-CSR (sorted by src) ---- */
[                   |                      ]  155:     out_idx = (uint32_t *)calloc(n + 1, sizeof(uint32_t));
[                   |                      ]  156:     out_nbr = (uint32_t *)malloc(n_sym  * sizeof(uint32_t));
[                   |                      ]  157:     if (!out_idx || !out_nbr) { perror("malloc out-CSR"); exit(1); }
[                   |                      ]  158: 
[                   |                      ]  159:     qsort(edges, n_sym, sizeof(Edge), cmp_by_u);
[                   |                      ]  160:     for (uint64_t i = 0; i < n_sym; i++) out_idx[edges[i].u + 1]++;
[                   |                      ]  161:     for (uint32_t i = 1; i <= n; i++)    out_idx[i] += out_idx[i-1];
[                   |                      ]  162:     for (uint64_t i = 0; i < n_sym; i++) out_nbr[i] = edges[i].v;
[                   |                      ]  163: 
[                   |                      ]  164:     /* ---- Build in-CSR (sorted by dst) ---- */
[                   |                      ]  165:     in_idx = (uint32_t *)calloc(n + 1, sizeof(uint32_t));
[                   |                      ]  166:     in_nbr = (uint32_t *)malloc(n_sym  * sizeof(uint32_t));
[                   |                      ]  167:     if (!in_idx || !in_nbr) { perror("malloc in-CSR"); exit(1); }
[                   |                      ]  168: 
[                   |                      ]  169:     qsort(edges, n_sym, sizeof(Edge), cmp_by_v);
[                   |                      ]  170:     for (uint64_t i = 0; i < n_sym; i++) in_idx[edges[i].v + 1]++;
[                   |                      ]  171:     for (uint32_t i = 1; i <= n; i++)    in_idx[i] += in_idx[i-1];
[                   |                      ]  172:     for (uint64_t i = 0; i < n_sym; i++) in_nbr[i] = edges[i].u;
[                   |                      ]  173: 
[                   |                      ]  174:     free(edges);
[                   |                      ]  175: 
[                   |                      ]  176:     /* ---- BFS state buffers ---- */
[                   |                      ]  177:     g_parent  = (int32_t  *)malloc(n * sizeof(int32_t));
[                   |                      ]  178:     g_qbuf    = (uint32_t *)malloc(2 * (size_t)n * sizeof(uint32_t));
[                   |                      ]  179:     size_t bm_words = ((size_t)n + 63) >> 6;
[                   |                      ]  180:     g_bmap[0] = (uint64_t *)calloc(bm_words, sizeof(uint64_t));
[                   |                      ]  181:     g_bmap[1] = (uint64_t *)calloc(bm_words, sizeof(uint64_t));
[                   |                      ]  182:     if (!g_parent || !g_qbuf || !g_bmap[0] || !g_bmap[1]) {
[                   |                      ]  183:         perror("malloc BFS state"); exit(1);
[                   |                      ]  184:     }
[                   |                      ]  185: 
[                   |                      ]  186:     /* Report working-set sizes */
[                   |                      ]  187:     uint64_t wset_bytes = (uint64_t)(n+1)*4*2       /* out_idx + in_idx */
[                   |                      ]  188:                         + n_sym*4*2                  /* out_nbr + in_nbr */
[                   |                      ]  189:                         + (uint64_t)n*4              /* parent            */
[                   |                      ]  190:                         + bm_words*8*2;              /* bitmaps           */
[                   |                      ]  191:     printf("[graph] Ready: %u nodes, %llu directed edges, "
[                   |                      ]  192:            "working-set ~%llu KB\n",
[                   |                      ]  193:            n, (unsigned long long)n_sym,
[                   |                      ]  194:            (unsigned long long)(wset_bytes / 1024));
[                   |                      ]  195: }
[                   |                      ]  196: 
[                   |                      ]  197: /* ============================================================
[                   |                      ]  198:  *  BFS phases
[                   |                      ]  199:  * ============================================================ */
[                   |                      ]  200: 
[                   |                      ]  201: /*
[                   |                      ]  202:  * Top-Down step:
[                   |                      ]  203:  *   For each node u in the frontier queue, walk its out-neighbors.
[                   |                      ]  204:  *   Key memory patterns:
[                   |                      ]  205:  *     out_idx[u]  – sequential-ish (queue order often not sorted by u)
[                   |                      ]  206:  *     out_nbr[ei] – sequential within u's block, scattered across edges
[                   |                      ]  207:  *     g_parent[v] – RANDOM load: v is a neighbor ID, unpredictable address
[                   |                      ]  208:  *     g_parent[v] – RANDOM store: write discovered parent
[                   |                      ]  209:  *
[                   |                      ]  210:  *   Returns scout_count = sum of out-degrees of newly discovered nodes
[                   |                      ]  211:  *   (used for direction switching heuristic).
[                   |                      ]  212:  */
[                   |                      ]  213: static int64_t td_step(const uint32_t *cur, int64_t cur_sz,
[                   |                      ]  214:                         uint32_t *nxt, int64_t *nxt_sz_out)
[                   |                      ]  215: {
[                   |                      ]  216:     int64_t scout = 0;
[                   |                      ]  217:     int64_t nq    = 0;
[                   |                      ]  218: 
[                   |                      ]  219:     for (int64_t qi = 0; qi < cur_sz; qi++) {
[                   |                      ]  220:         uint32_t u  = cur[qi];                         /* frontier node   */
[                   |                      ]  221:         uint32_t eo = out_idx[u];                      /* CSR row start   */
[                   |                      ]  222:         uint32_t ee = out_idx[u + 1];                  /* CSR row end     */
[                   |                      ]  223:         for (uint32_t ei = eo; ei < ee; ei++) {
[                   |                      ]  224:             uint32_t v  = out_nbr[ei];                 /* neighbor id (random) */
[                   |                      ]  225:             int32_t  pv = g_parent[v];                 /* RANDOM read: parent[v] */
[                   |                      ]  226:             if (pv < 0) {
[                   |                      ]  227:                 /* Single-threaded: no CAS needed, just set directly */
[                   |                      ]  228:                 g_parent[v] = (int32_t)u;              /* RANDOM write: parent[v] */
[                   |                      ]  229:                 nxt[nq++]   = v;
[                   |                      ]  230:                 scout      += (int64_t)(-pv);          /* -pv = out_degree[v] */
[                   |                      ]  231:             }
[                   |                      ]  232:         }
[                   |                      ]  233:     }
[                   |                      ]  234:     *nxt_sz_out = nq;
[                   |                      ]  235:     return scout;
[                   |                      ]  236: }
[                   |                      ]  237: 
[                   |                      ]  238: /*
[                   |                      ]  239:  * Bottom-Up step:
[                   |                      ]  240:  *   Sweep ALL nodes sequentially. For each unvisited node u,
[                   |                      ]  241:  *   check if any in-neighbor v is on the frontier bitmap.
[                   |                      ]  242:  *   Key memory patterns:
[                   |                      ]  243:  *     g_parent[u]  – SEQUENTIAL read (walking all nodes 0..N)
[                   |                      ]  244:  *     in_nbr[ei]   – SEQUENTIAL CSR read per node
[                   |                      ]  245:  *     bm_get(front, v) – RANDOM bitmap read: v is in-neighbor ID
[                   |                      ]  246:  *     bm_set(next, u)  – SEQUENTIAL bitmap write
[                   |                      ]  247:  *
[                   |                      ]  248:  *   Returns number of newly awoken (discovered) nodes.
[                   |                      ]  249:  */
[                   |                      ]  250: static int64_t bu_step(const uint64_t *front, uint64_t *next)
[                   |                      ]  251: {
[                   |                      ]  252:     int64_t awake = 0;
[                   |                      ]  253:     bm_clear_all(next);
[                   |                      ]  254: 
[                   |                      ]  255:     for (uint32_t u = 0; u < g_n; u++) {              /* sequential node sweep */
[                   |                      ]  256:         if (g_parent[u] < 0) {                         /* sequential read: unvisited? */
[                   |                      ]  257:             uint32_t ib = in_idx[u];                   /* CSR row start (in-edges) */
[                   |                      ]  258:             uint32_t ie = in_idx[u + 1];               /* CSR row end              */
[                   |                      ]  259:             for (uint32_t ei = ib; ei < ie; ei++) {
[                   |                      ]  260:                 uint32_t v = in_nbr[ei];               /* in-neighbor (random id)  */
[                   |                      ]  261:                 if (bm_get(front, v)) {                /* RANDOM bitmap read        */
[                   |                      ]  262:                     g_parent[u] = (int32_t)v;          /* sequential write: parent  */
[                   |                      ]  263:                     bm_set(next, u);                   /* sequential bitmap write   */
[                   |                      ]  264:                     awake++;
[                   |                      ]  265:                     break;
[                   |                      ]  266:                 }
[                   |                      ]  267:             }
[                   |                      ]  268:         }
[                   |                      ]  269:     }
[                   |                      ]  270:     return awake;
[                   |                      ]  271: }
[                   |                      ]  272: 
[                   |                      ]  273: /* Convert queue to bitmap (before entering BU direction) */
[                   |                      ]  274: static void queue_to_bitmap(const uint32_t *q, int64_t sz, uint64_t *bm)
[                   |                      ]  275: {
[                   |                      ]  276:     bm_clear_all(bm);
[                   |                      ]  277:     for (int64_t i = 0; i < sz; i++) bm_set(bm, q[i]);
[                   |                      ]  278: }
[                   |                      ]  279: 
[                   |                      ]  280: /* Convert bitmap back to queue (after leaving BU direction) */
[                   |                      ]  281: static int64_t bitmap_to_queue(const uint64_t *bm, uint32_t *q)
[                   |                      ]  282: {
[                   |                      ]  283:     int64_t n = 0;
[                   |                      ]  284:     for (uint32_t u = 0; u < g_n; u++)
[                   |                      ]  285:         if (bm_get(bm, u)) q[n++] = u;
[                   |                      ]  286:     return n;
[                   |                      ]  287: }
[                   |                      ]  288: 
[                   |                      ]  289: /* ============================================================
[                   |                      ]  290:  *  Top-level BFS  (direction-optimizing, identical logic to GAPBS)
[                   |                      ]  291:  * ============================================================ */
[                   |                      ]  292: static void bfs(uint32_t source)
[         8 ( 0.01%) | L:       0 S:       6]  293: {
[                   |                      ]  294:     /*
[                   |                      ]  295:      * Init parent[]:
[                   |                      ]  296:      *   parent[v] = -out_degree(v)  →  unvisited  (negative = degree stored)
[                   |                      ]  297:      *   parent[v] >= 0              →  visited, holds parent id
[                   |                      ]  298:      *   parent[source] = source     →  BFS root
[                   |                      ]  299:      */
[    29,710 (29.71%) | L:       1 S:       0]  300:     for (uint32_t n = 0; n < g_n; n++) {
[    19,805 (19.80%) | L:  19,805 S:       0]  301:         uint32_t deg = out_idx[n+1] - out_idx[n];
[    49,512 (49.51%) | L:       0 S:   9,902]  302:         g_parent[n]  = (deg > 0) ? -(int32_t)deg : -1;
[                   |                      ]  303:     }
[         1 ( 0.00%) | L:       1 S:       0]  304:     g_parent[source] = (int32_t)source;
[                   |                      ]  305: 
[                   |                      ]  306:     /* Use two halves of g_qbuf as current/next frontier queues */
[                   |                      ]  307:     uint32_t *cur = g_qbuf;
[                   |                      ]  308:     uint32_t *nxt = g_qbuf + g_n;
[                   |                      ]  309:     cur[0] = source;
[                   |                      ]  310:     int64_t cur_sz = 1;
[                   |                      ]  311: 
[                   |                      ]  312:     int64_t edges_to_check = (int64_t)g_e;
[         1 ( 0.00%) | L:       1 S:       0]  313:     int64_t scout_count    = (int64_t)(out_idx[source+1] - out_idx[source]);
[                   |                      ]  314: 
[                   |                      ]  315:     int level   = 0;
[                   |                      ]  316:     int bm_cur  = 0;   /* which g_bmap[] holds the current BU frontier */
[                   |                      ]  317:     int in_bu   = 0;   /* currently running bottom-up? */
[                   |                      ]  318: 
[                   |                      ]  319:     while (cur_sz > 0) {
[                   |                      ]  320: 
[                   |                      ]  321:         if (scout_count > edges_to_check / ALPHA) {
[                   |                      ]  322:             /* ---- Switch to or continue Bottom-Up ---- */
[                   |                      ]  323: 
[                   |                      ]  324:             if (!in_bu) {
[                   |                      ]  325:                 /* First BU iteration: convert queue to bitmap */
[                   |                      ]  326:                 queue_to_bitmap(cur, cur_sz, g_bmap[bm_cur]);
[                   |                      ]  327:                 in_bu = 1;
[                   |                      ]  328:             }
[                   |                      ]  329: 
[                   |                      ]  330:             int64_t awake = cur_sz, old_awake;
[                   |                      ]  331:             int bm_nxt = 1 - bm_cur;
[                   |                      ]  332:             do {
[                   |                      ]  333:                 old_awake = awake;
[                   |                      ]  334:                 awake = bu_step(g_bmap[bm_cur], g_bmap[bm_nxt]);
[                   |                      ]  335:                 /* Swap front/next bitmaps */
[                   |                      ]  336:                 int tmp = bm_cur; bm_cur = bm_nxt; bm_nxt = tmp;
[                   |                      ]  337:                 printf("  BU  lvl %-3d : awake = %lld\n",
[                   |                      ]  338:                        level++, (long long)awake);
[                   |                      ]  339:             } while (awake >= old_awake || awake > (int64_t)(g_n / BETA));
[                   |                      ]  340: 
[                   |                      ]  341:             /* Convert final BU frontier bitmap back to queue */
[                   |                      ]  342:             cur_sz = bitmap_to_queue(g_bmap[bm_cur], cur);
[                   |                      ]  343:             scout_count = 1;
[                   |                      ]  344:             in_bu = 0;
[                   |                      ]  345: 
[                   |                      ]  346:         } else {
[                   |                      ]  347:             /* ---- Top-Down step ---- */
[                   |                      ]  348:             int64_t nxt_sz;
[                   |                      ]  349:             scout_count    = td_step(cur, cur_sz, nxt, &nxt_sz);
[                   |                      ]  350:             edges_to_check -= scout_count;
[                   |                      ]  351: 
[                   |                      ]  352:             /* Swap cur/nxt queues */
[                   |                      ]  353:             uint32_t *tmp = cur; cur = nxt; nxt = tmp;
[                   |                      ]  354:             cur_sz = nxt_sz;
[                   |                      ]  355: 
[                   |                      ]  356:             printf("  TD  lvl %-3d : frontier = %-8lld  scout = %lld\n",
[                   |                      ]  357:                    level++, (long long)cur_sz, (long long)scout_count);
[                   |                      ]  358:         }
[                   |                      ]  359:     }
[                   |                      ]  360: }
[                   |                      ]  361: 
[                   |                      ]  362: /* ============================================================
[                   |                      ]  363:  *  Main
[                   |                      ]  364:  * ============================================================ */
[                   |                      ]  365: int main(int argc, char *argv[])
[                   |                      ]  366: {
[                   |                      ]  367:     int      scale    = (argc > 1) ? atoi(argv[1]) : 16;
[                   |                      ]  368:     int      avg_deg  = (argc > 2) ? atoi(argv[2]) : 16;
[                   |                      ]  369:     unsigned seed     = (argc > 3) ? (unsigned)atoi(argv[3]) : 42;
[                   |                      ]  370: 
[                   |                      ]  371:     if (scale < 1 || scale > 26) {
[                   |                      ]  372:         fprintf(stderr, "scale must be 1-26 (got %d)\n", scale);
[                   |                      ]  373:         return 1;
[                   |                      ]  374:     }
[                   |                      ]  375: 
[                   |                      ]  376:     uint32_t n = (uint32_t)1u << scale;
[                   |                      ]  377: 
[                   |                      ]  378:     printf("=== Toy BFS Benchmark ===\n");
[                   |                      ]  379:     printf("  Scale      : %d  (%u nodes)\n", scale, n);
[                   |                      ]  380:     printf("  Avg degree : %d\n", avg_deg);
[                   |                      ]  381:     printf("  Seed       : %u\n", seed);
[                   |                      ]  382:     printf("  ALPHA/BETA : %d / %d\n\n", ALPHA, BETA);
[                   |                      ]  383: 
[                   |                      ]  384:     /* ---- Setup (outside ROI: not traced) ---- */
[                   |                      ]  385:     build_graph(n, avg_deg, seed);
[                   |                      ]  386: 
[                   |                      ]  387:     /* Find a non-isolated source vertex */
[                   |                      ]  388:     uint32_t source = 0;
[                   |                      ]  389:     while (source < n && (out_idx[source+1] - out_idx[source]) == 0)
[                   |                      ]  390:         source++;
[                   |                      ]  391:     if (source == n) { fprintf(stderr, "All nodes isolated!\n"); return 1; }
[                   |                      ]  392:     printf("\n[setup] Source node: %u (out-degree = %u)\n\n",
[                   |                      ]  393:            source, out_idx[source+1] - out_idx[source]);
[                   |                      ]  394: 
[                   |                      ]  395:     /*
[                   |                      ]  396:      * Warm-up run (outside ROI).
[                   |                      ]  397:      * Primes instruction caches and branch predictors without being traced.
[                   |                      ]  398:      * This matches the GAPBS pattern where one trial runs before ROI.
[                   |                      ]  399:      */
[                   |                      ]  400:     printf("[warmup] Running BFS (not traced)...\n");
[                   |                      ]  401:     bfs(source);
[                   |                      ]  402: 
[                   |                      ]  403:     /* Verify reachability after warmup run */
[                   |                      ]  404:     int64_t reachable = 0;
[                   |                      ]  405:     for (uint32_t i = 0; i < n; i++)
[                   |                      ]  406:         if (g_parent[i] >= 0) reachable++;
[                   |                      ]  407:     printf("[warmup] Reachable: %lld / %u nodes\n\n",
[                   |                      ]  408:            (long long)reachable, n);
[                   |                      ]  409: 
[                   |                      ]  410:     /* ---- ROI START: Pin tracer begins capturing instructions here ---- */
[         1 ( 0.00%) | L:       0 S:       0]  411:     roi_start();
[                   |                      ]  412: 
[         2 ( 0.00%) | L:       0 S:       0]  413:     printf("[bfs] Running BFS from node %u (TRACED)...\n", source);
[                   |                      ]  414:     bfs(source);
[                   |                      ]  415: 
[                   |                      ]  416:     /* ---- ROI END ---- */
[         1 ( 0.00%) | L:       0 S:       0]  417:     roi_end();
[                   |                      ]  418: 
[                   |                      ]  419:     /* Final verification */
[                   |                      ]  420:     reachable = 0;
[                   |                      ]  421:     for (uint32_t i = 0; i < n; i++)
[                   |                      ]  422:         if (g_parent[i] >= 0) reachable++;
[                   |                      ]  423:     printf("\n[result] Reachable: %lld / %u nodes\n",
[                   |                      ]  424:            (long long)reachable, n);
[                   |                      ]  425: 
[                   |                      ]  426:     return 0;
[                   |                      ]  427: }
[                   |                      ]  428: 
