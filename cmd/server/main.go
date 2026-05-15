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
	"time"

	"github.com/valyala/fasthttp"
	"github.com/valyala/fastjson"
)

// ====================================================================
// Logistic Regression Model
// ====================================================================

type Model struct {
	Bias    float64
	Weights [14]float64
}

func LoadModel(path string) (*Model, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	if len(data) < 120 {
		log.Fatalf("model.bin too small: %d bytes, expected 120", len(data))
	}

	m := &Model{}
	m.Bias = math.Float64frombits(binary.LittleEndian.Uint64(data[0:8]))
	for i := 0; i < 14; i++ {
		off := 8 + i*8
		m.Weights[i] = math.Float64frombits(binary.LittleEndian.Uint64(data[off : off+8]))
	}
	return m, nil
}

// Predict returns fraud probability [0, 1] via sigmoid(dot(weights, features) + bias).
func (m *Model) Predict(feats [14]float32) float64 {
	z := m.Bias
	for i := 0; i < 14; i++ {
		z += m.Weights[i] * float64(feats[i])
	}
	// Sigmoid with clipping to avoid overflow
	if z < -20 {
		return 0.0
	}
	if z > 20 {
		return 1.0
	}
	return 1.0 / (1.0 + math.Exp(-z))
}

// ====================================================================
// Normalization & MCC constants
// ====================================================================

type NormConstants struct {
	MaxAmount            float64
	MaxInstallments      float64
	AmountVsAvgRatio     float64
	MaxMinutes           float64
	MaxKm                float64
	MaxTxCount24h        float64
	MaxMerchantAvgAmount float64
}

var (
	norm      NormConstants
	mccMap    map[string]float64
	readyFlag int32
	model     *Model
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

// ====================================================================
// Vectorization (14 dimensions)
// ====================================================================

var parserPool = sync.Pool{
	New: func() interface{} {
		return &fastjson.Parser{}
	},
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

func scoreToResponse(prob float64) (approved bool, fraudScore float64) {
	// Map continuous probability to discrete fraud_score buckets
	switch {
	case prob < 0.1:
		fraudScore = 0.0
	case prob < 0.3:
		fraudScore = 0.2
	case prob < 0.5:
		fraudScore = 0.4
	case prob < 0.7:
		fraudScore = 0.6
	case prob < 0.9:
		fraudScore = 0.8
	default:
		fraudScore = 1.0
	}
	approved = fraudScore < 0.6
	return
}

func handleReady(ctx *fasthttp.RequestCtx) {
	ctx.SetStatusCode(fasthttp.StatusOK)
	ctx.SetBodyString("OK")
}

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

func handleFraudScore(ctx *fasthttp.RequestCtx) {
	feats := buildVector(ctx.PostBody())
	prob := model.Predict(feats)
	approved, fraudScore := scoreToResponse(prob)
	resp := fraudScoreResponse(approved, fraudScore)
	ctx.SetStatusCode(fasthttp.StatusOK)
	ctx.SetContentType("application/json")
	ctx.SetBody(resp)
}

// ====================================================================
// Flags
// ====================================================================

var modelPath = flag.String("model", "resources/model.bin", "path to logistic regression model")
var normPath = flag.String("norm", "resources/normalization.json", "path to normalization constants")
var mccPath = flag.String("mcc", "resources/mcc_risk.json", "path to MCC risk map")
var port = flag.Int("port", 8080, "HTTP listen port")

// ====================================================================
// Main
// ====================================================================

func main() {
	flag.Parse()

	var err error

	// Load model
	log.Printf("[classifier] Loading model from %s...", *modelPath)
	model, err = LoadModel(*modelPath)
	if err != nil {
		log.Fatalf("Failed to load model: %v", err)
	}
	log.Printf("[classifier] Bias=%.4f, Weights[0]=%.4f ... Weights[13]=%.4f",
		model.Bias, model.Weights[0], model.Weights[13])

	// Load normalization
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

	// Load MCC risk
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

	atomic.StoreInt32(&readyFlag, 1)

	addr := ":" + strconv.Itoa(*port)
	log.Printf("[classifier] Listening on %s", addr)

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
		Name:               "rinha-classifier",
		MaxRequestBodySize: 4096,
		ReadTimeout:        2 * time.Second,
		WriteTimeout:       2 * time.Second,
		MaxConnsPerIP:      1000,
	}

	if err := server.ListenAndServe(addr); err != nil {
		log.Fatalf("Server error: %v", err)
	}
}
