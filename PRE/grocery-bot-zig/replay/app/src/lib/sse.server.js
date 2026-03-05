/**
 * Shared SSE (Server-Sent Events) helpers for pipeline API routes.
 *
 * Provides cleanup, heartbeat, and stream factory functions used across
 * all subprocess-spawning endpoints.
 */

/**
 * Create an SSE cleanup function that tears down process, timers, and controller.
 *
 * @param {object} ctx - Mutable context bag with: closed, safetyTimeout, heartbeatInterval, process
 * @param {ReadableStreamDefaultController} controller
 * @returns {function} cleanup - Call to shut everything down
 */
export function createCleanup(ctx, controller) {
	return function cleanup() {
		if (ctx.closed) return;
		ctx.closed = true;
		if (ctx.safetyTimeout) { clearTimeout(ctx.safetyTimeout); ctx.safetyTimeout = null; }
		if (ctx.heartbeatInterval) { clearInterval(ctx.heartbeatInterval); ctx.heartbeatInterval = null; }
		if (ctx.process && !ctx.process.killed) {
			try { ctx.process.kill(); } catch (_) { /* best effort */ }
		}
		try { controller?.close(); } catch (_) { /* best effort */ }
	};
}

/**
 * Create an SSE event sender.
 *
 * @param {object} ctx - Mutable context bag with: closed
 * @param {ReadableStreamDefaultController} controller
 * @param {TextEncoder} encoder
 * @returns {function(string, object): void} sendEvent
 */
export function createSendEvent(ctx, controller, encoder) {
	return function sendEvent(type, data) {
		if (ctx.closed) return;
		try {
			controller.enqueue(encoder.encode(`data: ${JSON.stringify({ type, ...data })}\n\n`));
		} catch (_) { /* stream closed */ }
	};
}
