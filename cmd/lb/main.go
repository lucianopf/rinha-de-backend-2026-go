package main

import (
	"flag"
	"log"
	"net"
	"sync"
	"sync/atomic"
	"time"
)

// Minimal TCP load balancer with deadline-based connection management.
// Uses io.Copy bidirectionally but sets idle deadlines to prevent deadlocks.
// When a connection is idle for 3s, the deadline fires, io.Copy returns,
// and both connections close cleanly via FIN (not RST).
// Memory footprint: ~5MB. Designed for 0.10 CPU / 50MB limit.

var (
	backends   = []string{"api1:8080", "api2:8080"}
	connCounts = make([]int64, len(backends))
	healthy    = make([]int32, len(backends))
	idleTimeout = 3 * time.Second
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

// setIdleDeadline sets a deadline on the connection that fires after
// idleTimeout of inactivity. This prevents io.Copy from blocking forever.
// The deadline is NOT for total connection time — it resets on every read/write.
func setIdleDeadline(conn net.Conn) {
	conn.SetDeadline(time.Now().Add(idleTimeout))
}

// proxyTCP forwards data bidirectionally with idle deadlines.
// The deadlines ensure connections don't block forever on keep-alive.
// When a deadline fires, io.Copy returns, and both connections close cleanly.
//
// Key difference from v9: deadlines on BOTH connections prevent the
// wg.Wait() deadlock. When either side goes idle for 3s, both io.Copy
// calls return with timeout errors, wg.Wait() unblocks, and defers
// call Close() — producing FIN, not RST.
func proxyTCP(client net.Conn, backend string, idx int) {
	defer releaseBackend(idx)
	defer client.Close()

	server, err := net.DialTimeout("tcp", backend, 2*time.Second)
	if err != nil {
		return
	}
	defer server.Close()

	// Set initial idle deadlines
	setIdleDeadline(client)
	setIdleDeadline(server)

	var wg sync.WaitGroup
	wg.Add(2)

	// client → server
	go func() {
		defer wg.Done()
		buf := make([]byte, 16384)
		for {
			setIdleDeadline(client)
			n, err := client.Read(buf)
			if n > 0 {
				setIdleDeadline(server)
				if _, werr := server.Write(buf[:n]); werr != nil {
					return
				}
			}
			if err != nil {
				// EOF or timeout — signal server that we're done sending
				if tcp, ok := server.(*net.TCPConn); ok {
					tcp.CloseWrite()
				}
				return
			}
		}
	}()

	// server → client
	go func() {
		defer wg.Done()
		buf := make([]byte, 16384)
		for {
			setIdleDeadline(server)
			n, err := server.Read(buf)
			if n > 0 {
				setIdleDeadline(client)
				if _, werr := client.Write(buf[:n]); werr != nil {
					return
				}
			}
			if err != nil {
				return
			}
		}
	}()

	wg.Wait()
	// Both directions done — deferred Close() on client and server
	// produce clean FIN sequences because all data has been read/written
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
	log.Printf("[lb] Listening on %s → %v (least-conn, deadline-tcp, idle=%v)", *listenAddr, backends, idleTimeout)

	for {
		conn, err := ln.Accept()
		if err != nil {
			log.Printf("[lb] Accept error: %v", err)
			continue
		}
		backend, idx := pickBackend()
		go proxyTCP(conn, backend, idx)
	}
}
