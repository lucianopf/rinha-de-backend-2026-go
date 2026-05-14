package main

import (
	"encoding/binary"
	"flag"
	"log"
	"math"
	"os"
	"strconv"
	"sync"
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

	Centroids []float32
	Clusters  [][]int32
	Vectors   []float32
	Labels    []uint8
	Data      []byte
}

type NormConstants struct {
	MaxAmount            float64
	MaxInstallments      float64
	AmountVsAvgRatio     float64
	MaxMinutes           float64
	MaxKm                float64
	MaxTxCount24h        float64
	MaxMerchantAvgAmount float64
}

// ====================================================================
// Pre-allocated search buffers (pooled per request)
// ====================================================================

type searchBufs struct {
	centroidDists []float32
	topClusters   []int32
	topDists      []float32
	heapDists     [5]float32
	heapLabels    [5]uint8
	heapSize      int
}

var bufsPool = sync.Pool{
	New: func() interface{} {
		return &searchBufs{
			centroidDists: make([]float32, 2048),
			topClusters:   make([]int32, 32),
			topDists:      make([]float32, 32),
		}
	},
}

var parserPool = sync.Pool{
	New: func() interface{} {
		return &fastjson.Parser{}
	},
}

// ====================================================================
// Fixed-size max-heap (tracks K smallest distances, inlined)
// ====================================================================

func heapPush(dists *[5]float32, labels *[5]uint8, size *int, k int, dist float32, label uint8) {
	if *size < k {
		i := *size
		dists[i] = dist
		labels[i] = label
		*size++
		for i > 0 {
			parent := (i - 1) / 2
			if dists[parent] >= dists[i] {
				break
			}
			dists[parent], dists[i] = dists[i], dists[parent]
			labels[parent], labels[i] = labels[i], labels[parent]
			i = parent
		}
	} else if dist < dists[0] {
		dists[0] = dist
		labels[0] = label
		i := 0
		for {
			largest := i
			left := 2*i + 1
			right := 2*i + 2
			if left < k && dists[left] > dists[largest] {
				largest = left
			}
			if right < k && dists[right] > dists[largest] {
				largest = right
			}
			if largest == i {
				break
			}
			dists[i], dists[largest] = dists[largest], dists[i]
			labels[i], labels[largest] = labels[largest], labels[i]
			i = largest
		}
	}
}

// ====================================================================
// IVF Search (zero-allocation via pool)
// ====================================================================

func (idx *IVFIndex) Search(query []float32, probes int) (approved bool, fraudScore float64) {
	b := bufsPool.Get().(*searchBufs)
	defer bufsPool.Put(b)

	dim := idx.Dim
	nClusters := idx.NClusters
	centroidDists := b.centroidDists[:nClusters]

	// Compute distances to all centroids
	centroids := idx.Centroids
	for c := 0; c < nClusters; c++ {
		base := c * dim
		var sum float32
		for d := 0; d < dim; d++ {
			diff := query[d] - centroids[base+d]
			sum += diff * diff
		}
		centroidDists[c] = sum
	}

	// Find top probes clusters
	topClusters := b.topClusters[:probes]
	topDists := b.topDists[:probes]
	for i := range topClusters {
		topClusters[i] = -1
		topDists[i] = float32(math.MaxFloat32)
	}

	for c := 0; c < nClusters; c++ {
		dist := centroidDists[c]
		pos := probes
		for p := 0; p < probes; p++ {
			if dist < topDists[p] {
				pos = p
				break
			}
		}
		if pos < probes {
			copy(topClusters[pos+1:], topClusters[pos:probes-1])
			copy(topDists[pos+1:], topDists[pos:probes-1])
			topClusters[pos] = int32(c)
			topDists[pos] = dist
		}
	}

	// Exact KNN in probed clusters
	heapDists := &b.heapDists
	heapLabels := &b.heapLabels
	heapSize := 0
	vectors := idx.Vectors
	labels := idx.Labels

	for p := 0; p < probes && topClusters[p] >= 0; p++ {
		c := topClusters[p]
		for _, vi := range idx.Clusters[c] {
			vbase := int(vi) * dim
			q := query

			d0 := q[0] - vectors[vbase+0]
			d1 := q[1] - vectors[vbase+1]
			d2 := q[2] - vectors[vbase+2]
			d3 := q[3] - vectors[vbase+3]
			d4 := q[4] - vectors[vbase+4]
			d5 := q[5] - vectors[vbase+5]
			d6 := q[6] - vectors[vbase+6]
			d7 := q[7] - vectors[vbase+7]
			d8 := q[8] - vectors[vbase+8]
			d9 := q[9] - vectors[vbase+9]
			d10 := q[10] - vectors[vbase+10]
			d11 := q[11] - vectors[vbase+11]
			d12 := q[12] - vectors[vbase+12]
			d13 := q[13] - vectors[vbase+13]

			sum := d0*d0 + d1*d1 + d2*d2 + d3*d3 + d4*d4 + d5*d5 + d6*d6 +
				d7*d7 + d8*d8 + d9*d9 + d10*d10 + d11*d11 + d12*d12 + d13*d13

			heapPush(heapDists, heapLabels, &heapSize, 5, sum, labels[vi])
		}
	}

	frauds := 0
	for i := 0; i < heapSize; i++ {
		if heapLabels[i] == 1 {
			frauds++
		}
	}
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
	f.Close()

	idx := &IVFIndex{Data: data}

	idx.Count = int(binary.LittleEndian.Uint32(data[0:4]))
	idx.Dim = int(binary.LittleEndian.Uint32(data[4:8]))
	idx.NClusters = int(binary.LittleEndian.Uint32(data[8:12]))

	dim := idx.Dim
	nClusters := idx.NClusters

	centroidBytes := nClusters * dim * 4
	idx.Centroids = unsafe.Slice((*float32)(unsafe.Pointer(&data[12])), nClusters*dim)

	pos := 12 + centroidBytes

	idx.Clusters = make([][]int32, nClusters)
	for c := 0; c < nClusters; c++ {
		n := int(binary.LittleEndian.Uint32(data[pos : pos+4]))
		pos += 4
		idx.Clusters[c] = unsafe.Slice((*int32)(unsafe.Pointer(&data[pos])), n)
		pos += n * 4
	}

	vectorBytes := idx.Count * dim * 4
	idx.Vectors = unsafe.Slice((*float32)(unsafe.Pointer(&data[pos])), idx.Count*dim)
	pos += vectorBytes

	idx.Labels = data[pos : pos+idx.Count]

	return idx, nil
}

// ====================================================================
// Vectorization (14 dimensions)
// ====================================================================

var (
	norm      NormConstants
	mccMap    map[string]float64
	readyFlag int32
	index     *IVFIndex
)

func clamp(v float64) float64 {
	if v < 0 {
		return 0
	}
	if v > 1 {
		return 1
	}
	return v
}

func parseTimestamp(ts string) time.Time {
	t, err := time.Parse(time.RFC3339, ts)
	if err != nil {
		t, _ = time.Parse("2006-01-02T15:04:05Z", ts)
	}
	return t
}

func dayOfWeekC(t time.Time) int {
	return (int(t.Weekday()) + 6) % 7
}

func buildVector(body []byte) (vec [14]float32) {
	p := parserPool.Get().(*fastjson.Parser)
	v, err := p.ParseBytes(body)
	if err != nil {
		parserPool.Put(p)
		return vec
	}

	amount := v.GetFloat64("transaction", "amount")
	installments := v.GetFloat64("transaction", "installments")
	requestedAt := string(v.GetStringBytes("transaction", "requested_at"))

	avgAmount := v.GetFloat64("customer", "avg_amount")
	txCount24h := v.GetFloat64("customer", "tx_count_24h")
	knownMerchants := v.GetArray("customer", "known_merchants")

	merchantID := string(v.GetStringBytes("merchant", "id"))
	mcc := string(v.GetStringBytes("merchant", "mcc"))
	merchantAvg := v.GetFloat64("merchant", "avg_amount")

	isOnline := v.GetBool("terminal", "is_online")
	cardPresent := v.GetBool("terminal", "card_present")
	kmFromHome := v.GetFloat64("terminal", "km_from_home")

	lastTx := v.Get("last_transaction")
	hasLastTx := lastTx != nil && lastTx.Type() != fastjson.TypeNull

	// Cache values we need after parser return
	var lastTs string
	var kmLast float64
	if hasLastTx {
		lastTs = string(v.GetStringBytes("last_transaction", "timestamp"))
		kmLast = v.GetFloat64("last_transaction", "km_from_current")
	}

	parserPool.Put(p)

	// 0: amount
	vec[0] = float32(clamp(amount / norm.MaxAmount))

	// 1: installments
	vec[1] = float32(clamp(installments / norm.MaxInstallments))

	// 2: amount_vs_avg
	vec[2] = float32(clamp((amount / avgAmount) / norm.AmountVsAvgRatio))

	// 3: hour_of_day
	t := parseTimestamp(requestedAt)
	vec[3] = float32(clamp(float64(t.Hour()) / 23.0))

	// 4: day_of_week
	cDow := dayOfWeekC(t)
	vec[4] = float32(clamp(float64(cDow) / 6.0))

	// 5: minutes_since_last_tx
	if hasLastTx {
		lt := parseTimestamp(lastTs)
		mins := t.Sub(lt).Minutes()
		vec[5] = float32(clamp(mins / norm.MaxMinutes))
	} else {
		vec[5] = -1
	}

	// 6: km_from_last_tx
	if hasLastTx {
		vec[6] = float32(clamp(kmLast / norm.MaxKm))
	} else {
		vec[6] = -1
	}

	// 7: km_from_home
	vec[7] = float32(clamp(kmFromHome / norm.MaxKm))

	// 8: tx_count_24h
	vec[8] = float32(clamp(txCount24h / norm.MaxTxCount24h))

	// 9: is_online
	if isOnline {
		vec[9] = 1
	}

	// 10: card_present
	if cardPresent {
		vec[10] = 1
	}

	// 11: unknown_merchant
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
	vec[13] = float32(clamp(merchantAvg / norm.MaxMerchantAvgAmount))

	return vec
}

// ====================================================================
// HTTP Handlers
// ====================================================================

var (
	respApproved0 = []byte(`{"approved":true,"fraud_score":0.0}`)
	respApproved2 = []byte(`{"approved":true,"fraud_score":0.2}`)
	respApproved4 = []byte(`{"approved":true,"fraud_score":0.4}`)
	respDenied6   = []byte(`{"approved":false,"fraud_score":0.6}`)
	respDenied8   = []byte(`{"approved":false,"fraud_score":0.8}`)
	respDenied10  = []byte(`{"approved":false,"fraud_score":1.0}`)
)

func fraudScoreResponse(approved bool, fraudScore float64) []byte {
	switch {
	case fraudScore < 0.2:
		return respApproved0
	case fraudScore < 0.4:
		return respApproved2
	case fraudScore < 0.6:
		return respApproved4
	case fraudScore < 0.8:
		return respDenied6
	case fraudScore < 1.0:
		return respDenied8
	default:
		return respDenied10
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
	mccData, err := os.ReadFile(*mccPath)
	if err != nil {
		log.Fatalf("Failed to load mcc_risk.json: %v", err)
	}
	mccV, err := p.Parse(string(mccData))
	if err != nil {
		log.Fatalf("Failed to parse mcc_risk.json: %v", err)
	}
	mccMap = make(map[string]float64)
	mccV.GetObject().Visit(func(key []byte, val *fastjson.Value) {
		mccMap[string(key)] = val.GetFloat64()
	})

	// Load IVF index
	log.Printf("Loading IVF index from %s...", *dataPath)
	index, err = LoadIVFIndex(*dataPath)
	if err != nil {
		log.Fatalf("Failed to load IVF index: %v", err)
	}
	log.Printf("Loaded: %d vectors, %d dims, %d clusters, %d probes",
		index.Count, index.Dim, index.NClusters, *probesFlag)

	atomic.StoreInt32(&readyFlag, 1)

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
		ReadTimeout:        5 * time.Second,
		WriteTimeout:       5 * time.Second,
		MaxConnsPerIP:      2000,
	}

	if err := server.ListenAndServe(addr); err != nil {
		log.Fatalf("Server error: %v", err)
	}
}
