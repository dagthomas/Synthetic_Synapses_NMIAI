import useSWR from "swr"
import {
  fetchTasks,
  fetchLanguages,
  fetchRuns,
  fetchSandboxHealth,
  fetchPayloads,
  fetchCoverage,
  fetchLogs,
  fetchTasksLiveSummary,
  autoPollScores,
} from "@/lib/api"

export function useTasks() {
  return useSWR("tasks", fetchTasks, {
    revalidateOnFocus: false,
    dedupingInterval: 60_000,
  })
}

export function useLanguages() {
  return useSWR("languages", fetchLanguages, {
    revalidateOnFocus: false,
    dedupingInterval: 60_000,
  })
}

export function useRuns(filter: string = "all", source: string = "all", refreshInterval = 5000) {
  return useSWR(
    ["runs", filter, source],
    () => fetchRuns({ status: filter, source, limit: 200 }),
    { refreshInterval, dedupingInterval: 2000 }
  )
}

export function useSandboxHealth(enabled = true) {
  return useSWR(enabled ? "sandbox-health" : null, fetchSandboxHealth, {
    revalidateOnFocus: false,
    dedupingInterval: 10_000,
  })
}

export function usePayloads(enabled = true) {
  return useSWR(enabled ? "payloads" : null, () => fetchPayloads(100), {
    revalidateOnFocus: false,
  })
}

export function useCoverage() {
  return useSWR("coverage", fetchCoverage, {
    revalidateOnFocus: false,
    dedupingInterval: 60_000,
  })
}

export function useLogs(enabled = true) {
  return useSWR(enabled ? "logs" : null, () => fetchLogs(100), {
    refreshInterval: 10_000,
    dedupingInterval: 5_000,
  })
}

export function useTasksLiveSummary() {
  return useSWR("tasks-live-summary", fetchTasksLiveSummary, {
    revalidateOnFocus: false,
    dedupingInterval: 30_000,
  })
}

export function useScoreAutoPolling(enabled = true) {
  return useSWR(
    enabled ? "score-auto-poll" : null,
    autoPollScores,
    {
      refreshInterval: 30_000,
      revalidateOnFocus: false,
      dedupingInterval: 15_000,
      errorRetryCount: 2,
    }
  )
}
