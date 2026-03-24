# ── Stage 1: Build Svelte frontend ────────────────────────────────────
FROM node:22-alpine AS frontend-build

WORKDIR /app/frontend
COPY frontend/package.json frontend/package-lock.json* ./
RUN npm ci
COPY frontend/ ./
RUN npm run build

# ── Stage 2: Build Rust backend ──────────────────────────────────────
FROM rust:latest AS backend-build

WORKDIR /app
COPY Cargo.toml Cargo.lock* ./
COPY src/ src/
COPY migrations/ migrations/

# Build release binary
RUN cargo build --release

# ── Stage 3: Final runtime image ─────────────────────────────────────
FROM debian:bookworm-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates libpq5 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy Rust binary
COPY --from=backend-build /app/target/release/astar-island-server ./

# Copy built Svelte SPA
COPY --from=frontend-build /app/frontend/build ./frontend/build/

# Copy migrations and static files
COPY migrations/ migrations/
COPY static/ static/

ENV DATABASE_URL="postgres://astar:astar@postgres:5432/astar_island"
ENV LISTEN_ADDR="0.0.0.0:8080"
ENV JWT_SECRET="change-me-in-production"
ENV RUST_LOG="info"

EXPOSE 8080

CMD ["./astar-island-server"]
