package main

import (
	"flag"
	"io"
	"log"
	"net"
	"sync"
	"sync/atomic"
	"time"
)

// Minimal least-connection TCP load balancer.
// Replaces HAProxy — zero HTTP parsing, pure io.Copy forwarding.
// Memory footprint: ~5MB. Designed for 0.10 CPU / 50MB limit.

var (
	backends   = []string{"api1:8080", "api2:8080"}
	connCounts = make([]int64, len(backends))
	healthy    = make([]int32, len(backends))
	mu         sync.Mutex
)

func init() {
	for i := range healthy {
		atomic.StoreInt32(&healthy[i], 1)
	}
}

func healthCheck(addr string, idx int) {
	for {
		conn, err := net.DialTimeout("tcp", addr, 1*time.Second)
		if err != nil {
			atomic.StoreInt32(&healthy[idx], 0)
		} else {
			conn.Close()
			atomic.StoreInt32(&healthy[idx], 1)
		}
		time.Sleep(1 * time.Second)
	}
}

func pickBackend() (string, int) {
	mu.Lock()
	defer mu.Unlock()

	bestIdx := -1
	bestCount := int64(1<<63 - 1)

	for i := range backends {
		if atomic.LoadInt32(&healthy[i]) == 0 {
			continue
		}
		c := atomic.LoadInt64(&connCounts[i])
		if c < bestCount {
			bestCount = c
			bestIdx = i
		}
	}

	if bestIdx == -1 {
		bestIdx = 0
	}

	atomic.AddInt64(&connCounts[bestIdx], 1)
	return backends[bestIdx], bestIdx
}

func releaseBackend(idx int) {
	atomic.AddInt64(&connCounts[idx], -1)
}

func proxy(client net.Conn, backend string, idx int) {
	defer releaseBackend(idx)
	defer client.Close()

	server, err := net.DialTimeout("tcp", backend, 2*time.Second)
	if err != nil {
		return
	}
	defer server.Close()

	// No deadlines — let the TCP stack handle timeouts natively.
	// Deadlines cause RST when they fire mid-transfer with Java clients.

	var wg sync.WaitGroup
	wg.Add(2)

	go func() {
		defer wg.Done()
		io.Copy(server, client)
	}()

	go func() {
		defer wg.Done()
		io.Copy(client, server)
	}()

	wg.Wait()
}

func main() {
	listenAddr := flag.String("listen", ":9999", "listen address")
	flag.Parse()

	for i, addr := range backends {
		go healthCheck(addr, i)
	}

	ln, err := net.Listen("tcp", *listenAddr)
	if err != nil {
		log.Fatalf("Failed to listen on %s: %v", *listenAddr, err)
	}
	log.Printf("[lb] Listening on %s → %v (least-conn)", *listenAddr, backends)

	for {
		conn, err := ln.Accept()
		if err != nil {
			log.Printf("[lb] Accept error: %v", err)
			continue
		}
		backend, idx := pickBackend()
		go proxy(conn, backend, idx)
	}
}
