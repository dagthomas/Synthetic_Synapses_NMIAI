# Grocery Bot — NM i AI 2026
# ===========================

# Grab the second word from the command line as URL argument
URL := $(word 2,$(MAKECMDGOALS))

# Only run the first target (not the URL pseudo-target)
.DEFAULT_GOAL := help
FIRST_GOAL := $(firstword $(MAKECMDGOALS))

.PHONY: help install easy medium hard expert scout-easy scout-medium scout-hard scout-expert plan-easy plan-medium plan-hard plan-expert play-easy play-medium play-hard play-expert sim-easy sim-medium sim-hard sim-expert local-easy local-medium local-hard local-expert optimize-easy optimize-medium optimize-hard optimize-expert replay-easy replay-medium replay-hard replay-expert recordings clean

help:
	@echo ""
	@echo "  Grocery Bot — NM i AI 2026"
	@echo "  =========================="
	@echo ""
	@echo "  Setup:"
	@echo "    make install                        Install dependencies"
	@echo ""
	@echo "  Quick play (smart mode):"
	@echo "    make easy   <url>                   Play Easy   (1 bot)"
	@echo "    make medium <url>                   Play Medium (3 bots)"
	@echo "    make hard   <url>                   Play Hard   (5 bots)"
	@echo "    make expert <url>                   Play Expert (10 bots)"
	@echo ""
	@echo "  Two-phase optimized (scout → plan → play):"
	@echo "    make scout-easy <url>               1. Scout: discover orders"
	@echo "    make plan-easy                      2. Plan: compute optimal routes"
	@echo "    make play-easy  <url>               3. Play: execute with plan"
	@echo ""
	@echo "  Local simulation (no server needed):"
	@echo "    make sim-easy                       Simulate Easy   (1 bot)"
	@echo "    make sim-medium                     Simulate Medium (3 bots)"
	@echo "    make sim-hard                       Simulate Hard   (5 bots)"
	@echo "    make sim-expert                     Simulate Expert (10 bots)"
	@echo ""
	@echo "  Local replay:"
	@echo "    make local-easy                     Replay saved Easy game"
	@echo "    make local-medium                   Replay saved Medium game"
	@echo ""
	@echo "  Full-game optimizer (beam search):"
	@echo "    make optimize-hard                  Optimize Hard  (offline)"
	@echo "    make replay-hard  <url>             Replay optimized plan"
	@echo ""
	@echo "  Utilities:"
	@echo "    make recordings                     List all saved recordings"
	@echo "    make clean                          Delete all recordings + plans"
	@echo ""

# --- Setup ---

install:
	pip install -r requirements.txt

# --- Quick play (smart mode) ---

easy medium hard expert:
	@test -n "$(URL)" || (echo "Usage: make $@ <url>" && exit 1)
	python main.py $@ $(URL)

# --- Two-phase: Scout ---

scout-easy scout-medium scout-hard scout-expert:
	@test -n "$(URL)" || (echo "Usage: make $@ <url>" && exit 1)
	python main.py scout $(subst scout-,,$@) $(URL)

# --- Two-phase: Plan ---

plan-easy plan-medium plan-hard plan-expert:
	python main.py plan $(subst plan-,,$@)

# --- Two-phase: Play with plan ---

play-easy play-medium play-hard play-expert:
	@test -n "$(URL)" || (echo "Usage: make $@ <url>" && exit 1)
	python main.py play $(subst play-,,$@) $(URL)

# --- Local simulation (full game, real scores) ---

sim-easy sim-medium sim-hard sim-expert:
	python main.py sim $(subst sim-,,$@)

# --- Local replay (compare actions) ---

local-easy local-medium local-hard local-expert:
	python main.py local $(subst local-,,$@)

# --- Perturbation search (CPU parallel) ---

perturb-easy perturb-medium perturb-hard perturb-expert:
	python main.py perturb $(subst perturb-,,$@)

# --- Play with perturbation overrides ---

pp-easy pp-medium pp-hard pp-expert:
	@test -n "$(URL)" || (echo "Usage: make $@ <url>" && exit 1)
	python main.py perturb-play $(subst pp-,,$@) $(URL)

# --- Beam search (explores multiple paths) ---

beam-easy beam-medium beam-hard beam-expert:
	python beam_search.py $(subst beam-,,$@)

# --- Full-game optimizer ---

optimize-easy optimize-medium optimize-hard optimize-expert:
	python main.py optimize $(subst optimize-,,$@)

# --- Replay optimized plan ---

replay-easy replay-medium replay-hard replay-expert:
	@test -n "$(URL)" || (echo "Usage: make $@ <url>" && exit 1)
	python main.py replay $(subst replay-,,$@) $(URL)

# --- Utilities ---

recordings:
	@echo "Saved recordings:"
	@echo ""
	@for d in easy medium hard expert; do \
		count=$$(find simulation/$$d -name "*.json" -not -name "plan.json" 2>/dev/null | wc -l | tr -d ' '); \
		plan=""; \
		if [ -f simulation/$$d/plan.json ]; then plan=" [plan ready]"; fi; \
		echo "  $$d: $$count games$$plan"; \
		if [ "$$count" -gt 0 ]; then \
			ls -1t simulation/$$d/game_*.json 2>/dev/null | head -3 | sed 's/^/    /'; \
		fi; \
		echo ""; \
	done

clean:
	@echo "Deleting all saved recordings and plans..."
	rm -f simulation/easy/*.json
	rm -f simulation/medium/*.json
	rm -f simulation/hard/*.json
	rm -f simulation/expert/*.json
	@echo "Done."

# Swallow any extra arguments (e.g. the URL) so make doesn't error
%:
	@:

