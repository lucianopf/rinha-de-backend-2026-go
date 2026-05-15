#!/usr/bin/env python3
"""
Local load test that faithfully reproduces the Rinha engine test conditions.
Usage: python3 load_test.py [concurrency] [total_requests] [timeout_seconds]
"""
import json
import http.client
import time
import sys
import random
import threading
import os

# Load example payloads
EXAMPLES_PATH = "/tmp/example-payloads.json"
with open(EXAMPLES_PATH) as f:
    EXAMPLES = json.load(f)

# MCC codes for variation
MCCS = ["5411", "5812", "5814", "5912", "5944", "5999", "4814", "4900", "5310", "5541",
        "5813", "5921", "5942", "5970", "6010", "7011", "7994", "7999", "8062", "8099"]

def random_payload():
    """Generate a varied payload based on the examples."""
    ex = random.choice(EXAMPLES)
    p = json.loads(json.dumps(ex))  # deep copy
    
    # Vary amount
    p["transaction"]["amount"] = round(random.uniform(1, 5000), 2)
    p["transaction"]["installments"] = random.randint(1, 12)
    
    # Vary timestamps
    hour = random.randint(0, 23)
    p["transaction"]["requested_at"] = f"2026-01-{random.randint(1,28):02d}T{hour:02d}:{random.randint(0,59):02d}:{random.randint(0,59):02d}Z"
    
    # Vary customer
    p["customer"]["avg_amount"] = round(random.uniform(10, 2000), 2)
    p["customer"]["tx_count_24h"] = random.randint(0, 30)
    
    # Vary merchant
    p["merchant"]["mcc"] = random.choice(MCCS)
    p["merchant"]["avg_amount"] = round(random.uniform(5, 3000), 2)
    
    # Vary terminal
    p["terminal"]["is_online"] = random.choice([True, False])
    p["terminal"]["card_present"] = random.choice([True, False])
    p["terminal"]["km_from_home"] = round(random.uniform(0, 500), 1)
    
    # Vary last transaction
    if p.get("last_transaction"):
        if random.random() < 0.3:  # 30% chance to remove last_tx
            del p["last_transaction"]
        else:
            p["last_transaction"]["km_from_current"] = round(random.uniform(0, 500), 1)
            p["last_transaction"]["timestamp"] = f"2026-01-{random.randint(1,28):02d}T{random.randint(0,23):02d}:{random.randint(0,59):02d}:{random.randint(0,59):02d}Z"
    
    return p


def run_load_test(host="localhost:9999", concurrency=50, total=5000, timeout=10):
    """Run concurrent load test against the service."""
    times = []
    errors = []
    status_counts = {}
    lock = threading.Lock()
    start_event = threading.Event()
    counter = [0]
    
    def worker():
        while True:
            with lock:
                if counter[0] >= total:
                    break
                counter[0] += 1
                idx = counter[0]
            
            payload = json.dumps(random_payload())
            
            try:
                t0 = time.time()
                conn = http.client.HTTPConnection(host, timeout=timeout)
                conn.request('POST', '/fraud-score', body=payload,
                           headers={'Content-Type': 'application/json'})
                resp = conn.getresponse()
                body = resp.read()
                conn.close()
                elapsed = time.time() - t0
                
                with lock:
                    times.append(elapsed)
                    sc = resp.status
                    status_counts[sc] = status_counts.get(sc, 0) + 1
                    if sc != 200:
                        errors.append(f"HTTP {sc}: {body[:100]}")
            except Exception as e:
                with lock:
                    errors.append(str(e))
                    status_counts[0] = status_counts.get(0, 0) + 1
    
    threads = []
    print(f"Starting load test: {total} requests, {concurrency} concurrent, timeout={timeout}s")
    t_start = time.time()
    
    for i in range(concurrency):
        t = threading.Thread(target=worker)
        t.start()
        threads.append(t)
    
    # Progress reporting
    while any(t.is_alive() for t in threads):
        time.sleep(1)
        with lock:
            done = len(times) + len(errors)
            elapsed = time.time() - t_start
            if elapsed > 0 and done > 0:
                rate = done / elapsed
            else:
                rate = 0
            print(f"\r  Progress: {done}/{total} ({done*100//total}%) — {rate:.0f} req/s — errors: {len(errors)}", end="")
    
    for t in threads:
        t.join()
    
    t_total = time.time() - t_start
    print()  # newline
    
    if not times:
        print("NO SUCCESSFUL REQUESTS!")
        print(f"Errors: {len(errors)}")
        for e in errors[:10]:
            print(f"  {e}")
        return
    
    times.sort()
    n = len(times)
    p50 = times[n//2]
    p90 = times[int(n*0.9)]
    p95 = times[int(n*0.95)]
    p99 = times[int(n*0.99)]
    p995 = times[int(n*0.995)]
    avg = sum(times) / n
    min_t = times[0]
    max_t = times[-1]
    
    error_rate = len(errors) / (len(times) + len(errors)) * 100
    
    print(f"\n{'='*60}")
    print(f"LOAD TEST RESULTS — {host}")
    print(f"{'='*60}")
    print(f"Total requests:    {total}")
    print(f"Successful:        {len(times)} ({len(times)/total*100:.1f}%)")
    print(f"HTTP errors:       {len(errors)} ({error_rate:.1f}%)")
    print(f"Duration:          {t_total:.2f}s")
    print(f"Throughput:        {len(times)/t_total:.1f} req/s")
    print(f"")
    print(f"Latency (ms):")
    print(f"  min:  {min_t*1000:8.1f}")
    print(f"  avg:  {avg*1000:8.1f}")
    print(f"  p50:  {p50*1000:8.1f}")
    print(f"  p90:  {p90*1000:8.1f}")
    print(f"  p95:  {p95*1000:8.1f}")
    print(f"  p99:  {p99*1000:8.1f}")
    print(f"  p99.5:{p995*1000:8.1f}")
    print(f"  max:  {max_t*1000:8.1f}")
    print(f"")
    print(f"Status codes: {status_counts}")
    print(f"{'='*60}")
    
    # Scoring simulation (simplified)
    if p99 > 2.0:
        p99_score = -3000
    else:
        p99_score = "> -3000 (needs exact formula)"
    
    if error_rate > 15:
        det_score = -3000
    else:
        det_score = "> -3000 (needs exact formula)"
    
    print(f"Simulated scores: p99_score={p99_score}, det_score={det_score}")
    
    return {
        'p99': p99,
        'p50': p50,
        'avg': avg,
        'error_rate': error_rate,
        'throughput': len(times)/t_total,
        'success_count': len(times),
        'error_count': len(errors),
        'total_time': t_total
    }


if __name__ == '__main__':
    conc = int(sys.argv[1]) if len(sys.argv) > 1 else 50
    total = int(sys.argv[2]) if len(sys.argv) > 2 else 5000
    to = int(sys.argv[3]) if len(sys.argv) > 3 else 10
    
    run_load_test(concurrency=conc, total=total, timeout=to)
