# Stage 1: Build Go binary
FROM golang:1.22-alpine AS builder

RUN apk add --no-cache git ca-certificates

WORKDIR /app

# Download dependencies
COPY go.mod go.sum ./
RUN go mod download

# Build
COPY cmd/ ./cmd/
RUN CGO_ENABLED=0 GOOS=linux GOARCH=amd64 go build \
    -ldflags="-s -w" \
    -trimpath \
    -o /app/server \
    ./cmd/server/

# Stage 2: Minimal runtime
FROM alpine:3.20

RUN apk add --no-cache ca-certificates tzdata

ENV GOMEMLIMIT=140MiB
ENV GOGC=50

WORKDIR /app

COPY --from=builder /app/server .
COPY resources/references.bin ./resources/references.bin
COPY resources/normalization.json ./resources/normalization.json
COPY resources/mcc_risk.json ./resources/mcc_risk.json

EXPOSE 8080

ENTRYPOINT ["./server"]
CMD ["--port", "8080", "--data", "resources/references.bin"]
