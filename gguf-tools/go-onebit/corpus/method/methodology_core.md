# Computing Methodology Core (方法论核心)

## First principles (第一性原理)

Break a problem down to invariant facts — data sizes, access frequencies, physical
bandwidths, hardware ceilings — and rebuild the solution from those numbers instead of
by analogy. Before optimizing anything, write the account: how many bytes move, how many
times per second, what the physical ceiling is. If the account says the design exceeds a
physical ceiling, no amount of tuning saves it; change the design. If the account says
there is headroom, the gap is an engineering problem and code can close it.

## Cache penetration (缓存穿透)

Queries for keys that exist in neither cache nor database bypass the cache every time and
hammer the database. Two standard fixes:

1. Cache empty results: store a short-TTL "not found" marker so repeat misses hit cache.
2. Bloom filter in front: membership test rejects keys that cannot exist before any I/O.

```go
if !bloom.MayContain(key) {
	return nil, ErrNotFound // cannot exist: reject before cache and DB
}
v, err := cache.Get(key)
if err == ErrCacheMiss {
	v, err = db.Get(key)
	if err == ErrNotFound {
		cache.SetWithTTL(key, tombstone, 30*time.Second) // cache the miss
	}
}
```

## Cache breakdown (缓存击穿)

One hot key expires and thousands of concurrent requests hit the database at the same
instant. Fixes: singleflight (exactly one loader per key, everyone else waits for its
result), or logical expiry (serve the stale value while one goroutine refreshes).

```go
import "golang.org/x/sync/singleflight"

var g singleflight.Group

func load(key string) (any, error) {
	v, err, _ := g.Do(key, func() (any, error) {
		return db.Get(key) // only one concurrent caller runs this per key
	})
	return v, err
}
```

## Cache avalanche (缓存雪崩)

Many keys expire simultaneously (same TTL set in one batch) and the database takes the
full load at once. Fixes: jitter the TTL (base + random spread) so expiry is smeared over
time; layer caches (local LRU in front of Redis); circuit-break and serve degraded
responses when the backing store saturates.

```go
ttl := baseTTL + time.Duration(rand.Int63n(int64(baseTTL/5))) // ±20% jitter
```

## Rate limiting (限流)

Token bucket allows bursts up to bucket size while enforcing an average rate; leaky
bucket enforces a strictly smooth outflow; sliding window counts requests in the trailing
interval. Default to token bucket — it matches real traffic.

```go
import "golang.org/x/time/rate"

var limiter = rate.NewLimiter(rate.Limit(100), 200) // 100 req/s average, burst 200

func handle(w http.ResponseWriter, r *http.Request) {
	if !limiter.Allow() {
		http.Error(w, "rate limited", http.StatusTooManyRequests)
		return
	}
	serve(w, r)
}
```

## Idempotency (幂等)

At-least-once delivery means every consumer will eventually run twice on the same
message. Make the operation idempotent: carry an idempotency key, record it in a dedup
table in the same transaction as the effect, and let a repeat key make the operation a
no-op. For updates, use conditional writes (version number / compare-and-swap) so a
replayed stale write cannot clobber a newer state.

## Retry with backoff (重试退避)

Retry only idempotent operations. Use exponential backoff with full jitter — synchronized
retries are how one failure becomes an outage (retry storm). Cap total attempts with a
retry budget so a down dependency sheds load instead of amplifying it.

```go
for attempt := 0; attempt < maxAttempts; attempt++ {
	if err = op(); err == nil {
		return nil
	}
	sleep := time.Duration(rand.Int63n(int64(base << attempt))) // full jitter
	time.Sleep(sleep)
}
```

## Consistent hashing (一致性哈希)

Hash both nodes and keys onto a ring; a key belongs to the first node clockwise from it.
Adding or removing one node remaps only ~1/N of the keys, versus nearly all keys for
`hash(key) % N`. Use virtual nodes (each physical node appears many times on the ring) to
smooth the load distribution.

## CAP and eventual consistency

Under a network partition a distributed system chooses: refuse writes (consistency) or
accept divergence (availability). Most business systems choose availability plus eventual
consistency — reconcile with versions/CRDTs/read-repair — and reserve strong consistency
(consensus: Raft/Paxos) for the small core that needs it: locks, leadership, configuration.

## Go concurrency patterns

Worker pool bounds concurrency; context propagates cancellation; errgroup joins parallel
work and returns the first error.

```go
g, ctx := errgroup.WithContext(ctx)
for _, u := range urls {
	u := u
	g.Go(func() error { return fetch(ctx, u) })
}
if err := g.Wait(); err != nil { // first error cancels ctx for the rest
	return err
}
```

## Algorithmic method one-liners (算法方法论)

- Binary search on the answer: when "is X feasible?" is monotone in X, search X itself.
- Two pointers: sorted input + pair/window condition → O(n) instead of O(n²).
- Hash map: trade O(n) memory for O(1) membership — the twoSum move.
- Dynamic programming: optimal substructure + overlapping subproblems → memoize; define
  the state first, the transition falls out.
- BFS gives fewest-steps on unweighted graphs; DFS gives reachability and structure;
  Dijkstra when edges carry weights.
- Sort first: sorting unlocks two pointers, binary search, dedup, and greedy exchanges.
- Bloom filter: probabilistic membership, no false negatives — the front door of caches.
- Amortize: batch small I/O into large sequential I/O; the disk and the network both
  price per operation, not per byte.
