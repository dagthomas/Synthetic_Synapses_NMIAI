# NM i AI Pre-Competition Challenge

## What is this?

The Grocery Bot challenge is a pre-competition warm-up for NM i AI 2026. Build a bot that controls agents via WebSocket to navigate a grocery store, pick up items, and deliver orders.

- **Purpose**: Pre-competition warm-up to get familiar with the platform
- **Period**: February 20 - March 16, 2026
- **Platform**: [app.ainm.no](https://app.ainm.no)

## How It Works

1. **Pick a difficulty** from the available maps on the Challenge page
2. **Get a WebSocket URL** — click Play to get a game token
3. **Connect your bot** to the WebSocket URL
4. **Receive game state** each round as JSON
5. **Respond with actions** — one per bot (move, pickup, dropoff, or wait)
6. **300 rounds** maximum per game (500 for Nightmare), 120 seconds wall-clock limit (300s Nightmare)
7. **Best score per map** is saved automatically. Leaderboard = sum of best scores.

## 5 Difficulty Levels

| Level | Bots | Grid | Item Types | Drop Zones | Rounds | Description |
|-------|------|------|------------|------------|--------|-------------|
| Easy | 1 | 12x10 | 4 | 1 | 300 | Solo pathfinding |
| Medium | 3 | 16x12 | 8 | 1 | 300 | Team coordination |
| Hard | 5 | 22x14 | 12 | 1 | 300 | Multi-agent planning |
| Expert | 10 | 28x18 | 16 | 1 | 300 | Massive coordination |
| Nightmare | 20 | 30x18 | 21 | 3 | 500 | Total chaos |

## Quick Start

1. Sign in at [dev.ainm.no](https://dev.ainm.no) with Google
2. Create or join a team
3. Go to the Challenge page, pick a difficulty, click Play
4. Copy the WebSocket URL and connect your bot
5. Play all maps to maximize your leaderboard score

## Key Features

- **WebSocket** — you connect to the game server, not the other way around
- **No fog of war** — full map visible from round 1
- **Bot collision** — bots block each other (no two on same tile, except spawn)
- **Infinite orders** — orders keep generating, rounds are the only limit
- **Daily rotation** — item placement and orders change daily to prevent hardcoding
- **Deterministic within a day** — same map + same day = same game every time
- **Rate limits** — 60s cooldown, max 40/hour, 300/day per team
