import type { QueryClient } from '@tanstack/react-query'
import type { LlmConnection } from '@/lib/api'

export const AGENT_CONNECTIONS_KEY = ['agent-connections'] as const

/** Stop only the shared root list. Exact-item reads remain available for reconciliation. */
export async function cancelAgentConnectionsList(queryClient: QueryClient): Promise<void> {
  await queryClient.cancelQueries({ queryKey: AGENT_CONNECTIONS_KEY, exact: true })
}

export async function upsertAgentConnection(queryClient: QueryClient, connection: LlmConnection): Promise<void> {
  await cancelAgentConnectionsList(queryClient)
  if (connection.is_default) {
    for (const [queryKey, cached] of queryClient.getQueriesData<LlmConnection>({ queryKey: AGENT_CONNECTIONS_KEY })) {
      if (queryKey.length === 2 && queryKey[1] !== connection.id && cached) {
        queryClient.setQueryData(queryKey, { ...cached, is_default: false })
      }
    }
  }
  queryClient.setQueryData<LlmConnection[]>(AGENT_CONNECTIONS_KEY, (current = []) => {
    const index = current.findIndex((item) => item.id === connection.id)
    const next = connection.is_default
      ? current.map((item) => item.id === connection.id ? item : { ...item, is_default: false })
      : current.slice()
    if (index < 0) return [...next, connection]
    next[index] = connection
    return next
  })
  queryClient.setQueryData([...AGENT_CONNECTIONS_KEY, connection.id], connection)
}

export async function removeAgentConnection(queryClient: QueryClient, id: string): Promise<void> {
  await cancelAgentConnectionsList(queryClient)
  queryClient.setQueryData<LlmConnection[]>(AGENT_CONNECTIONS_KEY, (current = []) => current.filter((item) => item.id !== id))
  queryClient.removeQueries({ queryKey: [...AGENT_CONNECTIONS_KEY, id], exact: true })
}
