import { fetchAPI } from '$lib/api';
import type { DaemonStatus, DaemonParams, DaemonAutoloop, DaemonRoundScore } from '$lib/types';

export async function load({ fetch }) {
    try {
        const [status, params, autoloop, scores] = await Promise.all([
            fetchAPI<string[]>('/api/daemon/status', fetch),
            fetchAPI<DaemonParams>('/api/daemon/params', fetch),
            fetchAPI<DaemonAutoloop>('/api/daemon/autoloop', fetch),
            fetchAPI<DaemonRoundScore[]>('/api/daemon/scores', fetch),
        ]);
        return { status, params, autoloop, scores };
    } catch {
        return { status: [], params: null, autoloop: null, scores: [] };
    }
}
