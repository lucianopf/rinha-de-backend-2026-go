package main

import (
	"bufio"
	"bytes"
	"flag"
	"io"
	"log"
	"net"
	"strconv"
	"sync/atomic"
	"time"
)

// Minimal HTTP half-proxy load balancer.
// Only parses Content-Length from the request to know when client is done
// sending. Then raw io.Copy for the response (zero response parsing).
// Uses CloseWrite() to signal EOF to the backend, forcing clean teardown.
// Memory footprint: ~5MB. Designed for 0.10 CPU / 50MB limit.

var (
	backends   = []string{"api1:8080", "api2:8080"}
	connCounts = make([]int64, len(backends))
	healthy    = make([]int32, len(backends))
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

// parseContentLength scans HTTP headers for Content-Length.
// Returns the value, or -1 if not found (chunked / no body).
func parseContentLength(headers []byte) int {
	// Look for "\nContent-Length: " or "\ncontent-length: "
	// HTTP headers end with \r\n\r\n
	idx := bytes.Index(headers, []byte("\nContent-Length:"))
	if idx < 0 {
		idx = bytes.Index(headers, []byte("\ncontent-length:"))
	}
	if idx < 0 {
		return -1
	}
	// Skip to after the colon + optional space
	start := idx + len("\nContent-Length:")
	for start < len(headers) && headers[start] == ' ' || headers[start] == '\t' {
		start++
	}
	// Read digits
	end := start
	for end < len(headers) && headers[end] >= '0' && headers[end] <= '9' {
		end++
	}
	n, err := strconv.Atoi(string(headers[start:end]))
	if err != nil {
		return -1
	}
	return n
}

// readHeaders reads from reader until \r\n\r\n, returning the header bytes
// (including the trailing \r\n\r\n). Max 64KB headers.
func readHeaders(r *bufio.Reader) ([]byte, error) {
	var buf []byte
	for {
		line, err := r.ReadBytes('\n')
		if err != nil {
			return buf, err
		}
		buf = append(buf, line...)
		if bytes.HasSuffix(buf, []byte("\r\n\r\n")) || bytes.HasSuffix(buf, []byte("\n\n")) {
			return buf, nil
		}
		if len(buf) > 65536 {
			return buf, io.ErrUnexpectedEOF
		}
	}
}

// proxyHalf handles one HTTP request-response cycle.
// Minimal parse: only Content-Length to know request boundaries.
// Raw io.Copy for response forwarding (zero response parsing).
func proxyHalf(client net.Conn, backend string, idx int) {
	defer releaseBackend(idx)
	defer client.Close()

	clientReader := bufio.NewReaderSize(client, 16384)

	// 1. Read request headers from client (up to \r\n\r\n)
	headers, err := readHeaders(clientReader)
	if err != nil {
		return
	}

	// 2. Determine body size
	bodyLen := parseContentLength(headers)
	contentLengthSet := bodyLen >= 0

	// 3. If no Content-Length, add Connection: close to force simple teardown
	if !contentLengthSet {
		// Replace or add Connection: close
		headers = addConnectionClose(headers)
	}

	// 4. Dial backend
	server, err := net.DialTimeout("tcp", backend, 2*time.Second)
	if err != nil {
		return
	}
	defer server.Close()

	// 5. Write headers + body to backend
	if _, err := server.Write(headers); err != nil {
		return
	}

	// 6. Copy exactly bodyLen bytes from client to server (if POST/PUT with body)
	if contentLengthSet && bodyLen > 0 {
		if _, err := io.CopyN(server, clientReader, int64(bodyLen)); err != nil {
			return
		}
	}
	// For GET/HEAD/DELETE/POST-without-body: nothing to copy, just signal EOF below.

	// 7. CRITICAL: signal EOF to backend so it knows request is complete.
	// This triggers the server to process and send the response.
	if tcpServer, ok := server.(*net.TCPConn); ok {
		tcpServer.CloseWrite()
	}

	// 8. Raw io.Copy response from backend to client (zero parsing)
	io.Copy(client, server)

	// 9. Clean close — server.Close() and client.Close() via defer.
	// Because we did CloseWrite() on the server, the TCP stack sends FIN,
	// server responds, and then both sides close cleanly.
	// No RST packets → no java.net.SocketException: Connection reset.
}

// addConnectionClose inserts "Connection: close\r\n" before the final \r\n.
func addConnectionClose(headers []byte) []byte {
	// Find the position of the last \r\n\r\n
	pos := bytes.LastIndex(headers, []byte("\r\n\r\n"))
	if pos < 0 {
		return headers
	}
	// Insert "Connection: close\r\n" before the final \r\n
	result := make([]byte, 0, len(headers)+21)
	result = append(result, headers[:pos]...)
	result = append(result, []byte("\r\nConnection: close")...)
	result = append(result, headers[pos:]...)
	return result
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
	log.Printf("[lb] Listening on %s → %v (least-conn, half-proxy)", *listenAddr, backends)

	for {
		conn, err := ln.Accept()
		if err != nil {
			log.Printf("[lb] Accept error: %v", err)
			continue
		}
		backend, idx := pickBackend()
		go proxyHalf(conn, backend, idx)
	}
}
