package main

import (
	"encoding/binary"
	"flag"
	"log"
	"math"
	"os"
	"strconv"
	"sync/atomic"
	"syscall"
	"time"
	"unsafe"

	"github.com/valyala/fasthttp"
	"github.com/valyala/fastjson"
)

// ====================================================================
// Data Structures
// ====================================================================

type IVFIndex struct {
	Count     int
	Dim       int
	NClusters int

	Centroids   []float32 // nClusters * dim, row-major
	Clusters    [][]int32 // per-cluster vector indices
	Vectors     []float32 // count * dim, row-major
	Labels      []uint8   // count bytes: 0=legit, 1=fraud

	Data []byte // mmap'd raw data (for lifetime)
}

type QueryParams struct {
	Amount       float64
	Installments int
	RequestedAt  string

	AvgAmount     float64
	TxCount24h    int
	KnownMerchants []string

	MerchantID  string
	MCC         string
	MerchantAvg float64

	IsOnline    bool
	CardPresent bool
	KmFromHome  float64

	LastTxTimestamp    string
	LastTxKmFromCurrent float64
	HasLastTx          bool
}

// Normalization constants
type NormConstants struct {
	MaxAmount             float64 `json:"max_amount"`
	MaxInstallments       float64 `json:"max_installments"`
	AmountVsAvgRatio      float64 `json:"amount_vs_avg_ratio"`
	MaxMinutes            float64 `json:"max_minutes"`
	MaxKm                 float64 `json:"max_km"`
	MaxTxCount24h         float64 `json:"max_tx_count_24h"`
	MaxMerchantAvgAmount  float64 `json:"max_merchant_avg_amount"`
}

// ====================================================================
// KNN Search Implementation
// ====================================================================

// Min-heap for tracking K nearest neighbors
type neighbor struct {
	dist  float32
	label uint8
}

type neighborHeap struct {
	items []neighbor
	k     int
}

func newNeighborHeap(k int) *neighborHeap {
	return &neighborHeap{items: make([]neighbor, 0, k), k: k}
}

func (h *neighborHeap) push(dist float32, label uint8) {
	if len(h.items) < h.k {
		h.items = append(h.items, neighbor{dist, label})
		// sift up
		i := len(h.items) - 1
		for i > 0 {
			parent := (i - 1) / 2
			if h.items[parent].dist >= h.items[i].dist {
				break
			}
			h.items[parent], h.items[i] = h.items[i], h.items[parent]
			i = parent
		}
	} else if dist < h.items[0].dist {
		h.items[0] = neighbor{dist, label}
		// sift down
		i := 0
		for {
			largest := i
			left := 2*i + 1
			right := 2*i + 2
			if left < h.k && h.items[left].dist > h.items[largest].dist {
				largest = left
			}
			if right < h.k && h.items[right].dist > h.items[largest].dist {
				largest = right
			}
			if largest == i {
				break
			}
			h.items[i], h.items[largest] = h.items[largest], h.items[i]
			i = largest
		}
	}
}

func (h *neighborHeap) countFrauds() int {
	n := 0
	for _, nb := range h.items {
		if nb.label == 1 {
			n++
		}
	}
	return n
}

// IVF search: find nearest centroids, probe top clusters
func (idx *IVFIndex) Search(query []float32, probes int) (approved bool, fraudScore float64) {
	dim := idx.Dim
	nClusters := idx.NClusters

	// Find distances to all centroids
	centroidDists := make([]float32, nClusters)
	for c := 0; c < nClusters; c++ {
		base := c * dim
		var sum float32
		// Unrolled for auto-vectorization
		for d := 0; d < dim; d++ {
			diff := query[d] - idx.Centroids[base+d]
			sum += diff * diff
		}
		centroidDists[c] = sum
	}

	// Find top 'probes' clusters (simple partial sort)
	topClusters := make([]int, probes)
	topDists := make([]float32, probes)
	for i := range topClusters {
		topClusters[i] = -1
		topDists[i] = float32(math.MaxFloat32)
	}

	for c := 0; c < nClusters; c++ {
		dist := centroidDists[c]
		// Find position in topClusters
		pos := probes
		for p := 0; p < probes; p++ {
			if dist < topDists[p] {
				pos = p
				break
			}
		}
		if pos < probes {
			// Shift right and insert
			copy(topClusters[pos+1:], topClusters[pos:probes-1])
			copy(topDists[pos+1:], topDists[pos:probes-1])
			topClusters[pos] = c
			topDists[pos] = dist
		}
	}

	// Search exact KNN in probed clusters
	heap := newNeighborHeap(5)
	for p := 0; p < probes && topClusters[p] >= 0; p++ {
		c := topClusters[p]
		for _, vi := range idx.Clusters[c] {
			vbase := int(vi) * dim
			var sum float32
			// Unrolled euclidean distance
			sum += (query[0] - idx.Vectors[vbase+0]) * (query[0] - idx.Vectors[vbase+0])
			sum += (query[1] - idx.Vectors[vbase+1]) * (query[1] - idx.Vectors[vbase+1])
			sum += (query[2] - idx.Vectors[vbase+2]) * (query[2] - idx.Vectors[vbase+2])
			sum += (query[3] - idx.Vectors[vbase+3]) * (query[3] - idx.Vectors[vbase+3])
			sum += (query[4] - idx.Vectors[vbase+4]) * (query[4] - idx.Vectors[vbase+4])
			sum += (query[5] - idx.Vectors[vbase+5]) * (query[5] - idx.Vectors[vbase+5])
			sum += (query[6] - idx.Vectors[vbase+6]) * (query[6] - idx.Vectors[vbase+6])
			sum += (query[7] - idx.Vectors[vbase+7]) * (query[7] - idx.Vectors[vbase+7])
			sum += (query[8] - idx.Vectors[vbase+8]) * (query[8] - idx.Vectors[vbase+8])
			sum += (query[9] - idx.Vectors[vbase+9]) * (query[9] - idx.Vectors[vbase+9])
			sum += (query[10] - idx.Vectors[vbase+10]) * (query[10] - idx.Vectors[vbase+10])
			sum += (query[11] - idx.Vectors[vbase+11]) * (query[11] - idx.Vectors[vbase+11])
			sum += (query[12] - idx.Vectors[vbase+12]) * (query[12] - idx.Vectors[vbase+12])
			sum += (query[13] - idx.Vectors[vbase+13]) * (query[13] - idx.Vectors[vbase+13])
			heap.push(sum, idx.Labels[vi])
		}
	}

	frauds := heap.countFrauds()
	fraudScore = float64(frauds) / 5.0
	approved = fraudScore < 0.6
	return
}

// ====================================================================
// IVF Index Loading (mmap)
// ====================================================================

func LoadIVFIndex(path string) (*IVFIndex, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	fi, err := f.Stat()
	if err != nil {
		f.Close()
		return nil, err
	}
	size := int(fi.Size())

	data, err := syscall.Mmap(int(f.Fd()), 0, size, syscall.PROT_READ, syscall.MAP_SHARED)
	if err != nil {
		f.Close()
		return nil, err
	}
	f.Close() // fd can be closed after mmap

	idx := &IVFIndex{Data: data}

	// Header
	idx.Count = int(binary.LittleEndian.Uint32(data[0:4]))
	idx.Dim = int(binary.LittleEndian.Uint32(data[4:8]))
	idx.NClusters = int(binary.LittleEndian.Uint32(data[8:12]))

	dim := idx.Dim
	nClusters := idx.NClusters

	// Centroids
	centroidBytes := nClusters * dim * 4
	idx.Centroids = unsafe.Slice((*float32)(unsafe.Pointer(&data[12])), nClusters*dim)

	pos := 12 + centroidBytes

	// Cluster assignments
	idx.Clusters = make([][]int32, nClusters)
	for c := 0; c < nClusters; c++ {
		n := int(binary.LittleEndian.Uint32(data[pos : pos+4]))
		pos += 4
		clusterSize := n * 4
		idx.Clusters[c] = unsafe.Slice((*int32)(unsafe.Pointer(&data[pos])), n)
		pos += clusterSize
	}

	// Vectors
	vectorBytes := idx.Count * dim * 4
	idx.Vectors = unsafe.Slice((*float32)(unsafe.Pointer(&data[pos])), idx.Count*dim)
	pos += vectorBytes

	// Labels
	idx.Labels = data[pos : pos+idx.Count]

	return idx, nil
}

// ====================================================================
// Vectorization (14 dimensions)
// ====================================================================

var (
	norm   NormConstants
	mccMap map[string]float64
	ready  int32
	index  *IVFIndex
)

func loadMCCRisk(path string) (map[string]float64, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	var p fastjson.Parser
	v, err := p.Parse(string(data))
	if err != nil {
		return nil, err
	}
	m := make(map[string]float64)
	v.GetObject().Visit(func(key []byte, val *fastjson.Value) {
		m[string(key)] = val.GetFloat64()
	})
	return m, nil
}

func floatClamp(v, lo, hi float64) float64 {
	if v < lo {
		return lo
	}
	if v > hi {
		return hi
	}
	return v
}

func parseTimestamp(ts string) time.Time {
	t, err := time.Parse(time.RFC3339, ts)
	if err != nil {
		// Attempt with simpler format
		t, _ = time.Parse("2006-01-02T15:04:05Z", ts)
	}
	return t
}

// Day of week: Monday=0, Sunday=6 (C convention, Tomohiko Sakamoto)
// Go's Weekday: Sunday=0, Saturday=6
// Conversion: cDow = (goDow + 6) % 7
func dayOfWeekC(t time.Time) int {
	return (int(t.Weekday()) + 6) % 7
}

func buildVector(body []byte) (vec [14]float32) {
	var p fastjson.Parser
	v, err := p.ParseBytes(body)
	if err != nil {
		return vec
	}

	// Transaction fields
	amount := v.GetFloat64("transaction", "amount")
	installments := v.GetFloat64("transaction", "installments")
	requestedAt := string(v.GetStringBytes("transaction", "requested_at"))

	// Customer fields
	avgAmount := v.GetFloat64("customer", "avg_amount")
	txCount24h := v.GetFloat64("customer", "tx_count_24h")
	knownMerchants := v.GetArray("customer", "known_merchants")

	// Merchant fields
	merchantID := string(v.GetStringBytes("merchant", "id"))
	mcc := string(v.GetStringBytes("merchant", "mcc"))
	merchantAvg := v.GetFloat64("merchant", "avg_amount")

	// Terminal fields
	isOnline := v.GetBool("terminal", "is_online")
	cardPresent := v.GetBool("terminal", "card_present")
	kmFromHome := v.GetFloat64("terminal", "km_from_home")

	// Last transaction
	lastTx := v.Get("last_transaction")
	hasLastTx := lastTx != nil && lastTx.Type() != fastjson.TypeNull

	// ---- Build vector ----

	// 0: amount
	vec[0] = float32(floatClamp(amount/norm.MaxAmount, 0, 1))

	// 1: installments
	vec[1] = float32(floatClamp(installments/norm.MaxInstallments, 0, 1))

	// 2: amount_vs_avg
	vec[2] = float32(floatClamp((amount/avgAmount)/norm.AmountVsAvgRatio, 0, 1))

	// 3: hour_of_day
	t := parseTimestamp(requestedAt)
	vec[3] = float32(floatClamp(float64(t.Hour())/23.0, 0, 1))

	// 4: day_of_week (Monday=0, Sunday=6)
	cDow := dayOfWeekC(t)
	vec[4] = float32(floatClamp(float64(cDow)/6.0, 0, 1))

	// 5: minutes_since_last_tx
	if hasLastTx {
		lastTs := string(v.GetStringBytes("last_transaction", "timestamp"))
		lt := parseTimestamp(lastTs)
		mins := t.Sub(lt).Minutes()
		vec[5] = float32(floatClamp(mins/norm.MaxMinutes, 0, 1))
	} else {
		vec[5] = -1
	}

	// 6: km_from_last_tx
	if hasLastTx {
		kmLast := v.GetFloat64("last_transaction", "km_from_current")
		vec[6] = float32(floatClamp(kmLast/norm.MaxKm, 0, 1))
	} else {
		vec[6] = -1
	}

	// 7: km_from_home
	vec[7] = float32(floatClamp(kmFromHome/norm.MaxKm, 0, 1))

	// 8: tx_count_24h
	vec[8] = float32(floatClamp(txCount24h/norm.MaxTxCount24h, 0, 1))

	// 9: is_online
	if isOnline {
		vec[9] = 1
	}

	// 10: card_present
	if cardPresent {
		vec[10] = 1
	}

	// 11: unknown_merchant (1 = desconhecido)
	isKnown := false
	if knownMerchants != nil {
		for _, km := range knownMerchants {
			if string(km.GetStringBytes()) == merchantID {
				isKnown = true
				break
			}
		}
	}
	if !isKnown {
		vec[11] = 1
	}

	// 12: mcc_risk
	risk, ok := mccMap[mcc]
	if !ok {
		risk = 0.5
	}
	vec[12] = float32(risk)

	// 13: merchant_avg_amount
	vec[13] = float32(floatClamp(merchantAvg/norm.MaxMerchantAvgAmount, 0, 1))

	return vec
}

// ====================================================================
// HTTP Handlers
// ====================================================================

// Fraud score response templates
const (
	respApproved0 = `{"approved":true,"fraud_score":0.0}`
	respApproved2 = `{"approved":true,"fraud_score":0.2}`
	respApproved4 = `{"approved":true,"fraud_score":0.4}`
	respDenied6   = `{"approved":false,"fraud_score":0.6}`
	respDenied8   = `{"approved":false,"fraud_score":0.8}`
	respDenied10  = `{"approved":false,"fraud_score":1.0}`
)

func fraudScoreResponse(approved bool, fraudScore float64) []byte {
	switch {
	case fraudScore < 0.2:
		return []byte(respApproved0)
	case fraudScore < 0.4:
		return []byte(respApproved2)
	case fraudScore < 0.6:
		return []byte(respApproved4)
	case fraudScore < 0.8:
		return []byte(respDenied6)
	case fraudScore < 1.0:
		return []byte(respDenied8)
	default:
		return []byte(respDenied10)
	}
}

func handleReady(ctx *fasthttp.RequestCtx) {
	ctx.SetStatusCode(fasthttp.StatusOK)
	ctx.SetBodyString("OK")
}

var probesFlag = flag.Int("probes", 16, "number of IVF clusters to probe per query")
var dataPath = flag.String("data", "resources/references.bin", "path to preprocessed binary index")
var normPath = flag.String("norm", "resources/normalization.json", "path to normalization constants")
var mccPath = flag.String("mcc", "resources/mcc_risk.json", "path to MCC risk map")
var port = flag.Int("port", 8080, "HTTP listen port")

func handleFraudScore(ctx *fasthttp.RequestCtx) {
	vect := buildVector(ctx.PostBody())
	approved, fraudScore := index.Search(vect[:], *probesFlag)

	resp := fraudScoreResponse(approved, fraudScore)
	ctx.SetStatusCode(fasthttp.StatusOK)
	ctx.SetContentType("application/json")
	ctx.SetBody(resp)
}

func main() {
	flag.Parse()

	var err error

	// Load normalization constants
	normData, err := os.ReadFile(*normPath)
	if err != nil {
		log.Fatalf("Failed to read normalization.json: %v", err)
	}
	var p fastjson.Parser
	normV, err := p.Parse(string(normData))
	if err != nil {
		log.Fatalf("Failed to parse normalization.json: %v", err)
	}
	norm = NormConstants{
		MaxAmount:            normV.GetFloat64("max_amount"),
		MaxInstallments:      normV.GetFloat64("max_installments"),
		AmountVsAvgRatio:     normV.GetFloat64("amount_vs_avg_ratio"),
		MaxMinutes:           normV.GetFloat64("max_minutes"),
		MaxKm:                normV.GetFloat64("max_km"),
		MaxTxCount24h:        normV.GetFloat64("max_tx_count_24h"),
		MaxMerchantAvgAmount: normV.GetFloat64("max_merchant_avg_amount"),
	}

	// Load MCC risk map
	mccMap, err = loadMCCRisk(*mccPath)
	if err != nil {
		log.Fatalf("Failed to load mcc_risk.json: %v", err)
	}

	// Load IVF index
	log.Printf("Loading IVF index from %s...", *dataPath)
	index, err = LoadIVFIndex(*dataPath)
	if err != nil {
		log.Fatalf("Failed to load IVF index: %v", err)
	}
	log.Printf("Loaded: %d vectors, %d dims, %d clusters, %d probes",
		index.Count, index.Dim, index.NClusters, *probesFlag)

	atomic.StoreInt32(&ready, 1)

	// Start fasthttp server
	addr := ":" + strconv.Itoa(*port)
	log.Printf("Listening on %s", addr)

	server := &fasthttp.Server{
		Handler: func(ctx *fasthttp.RequestCtx) {
			switch string(ctx.Path()) {
			case "/ready":
				handleReady(ctx)
			case "/fraud-score":
				handleFraudScore(ctx)
			default:
				ctx.SetStatusCode(fasthttp.StatusNotFound)
			}
		},
		Name:               "rinha-fraud",
		MaxRequestBodySize: 4096,
		ReadTimeout:        2 * time.Second,
		WriteTimeout:       2 * time.Second,
		MaxConnsPerIP:      1000,
	}

	if err := server.ListenAndServe(addr); err != nil {
		log.Fatalf("Server error: %v", err)
	}
}
