FROM golang:1.24-alpine AS builder

WORKDIR /src
ENV GOTOOLCHAIN=auto
COPY go.mod go.sum ./
RUN go mod download
COPY . .
RUN CGO_ENABLED=0 go build -o /astar-tui .

FROM alpine:3.21
RUN apk add --no-cache ca-certificates python3

# Mirror the expected directory structure:
#   /app/tui/       <- workdir (so filepath.Base == "tui")
#   /app/.env       <- loaded from ../.env
#   /app/data/      <- data dir at ../data
WORKDIR /app/tui
COPY --from=builder /astar-tui /app/tui/astar-tui

EXPOSE 7091
ENTRYPOINT ["/app/tui/astar-tui", "--web", "--web-addr=:7091"]
