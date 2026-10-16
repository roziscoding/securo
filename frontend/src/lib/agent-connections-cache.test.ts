import { QueryClient } from '@tanstack/react-query'
import { describe, expect, it } from 'vitest'
import type { LlmConnection } from './api'
import { AGENT_CONNECTIONS_KEY, removeAgentConnection, upsertAgentConnection } from './agent-connections-cache'

const connection = (id: string, isDefault: boolean): LlmConnection => ({
  id, name: id, kind: 'openai', base_url: null, default_model: null, extra: {}, is_default: isDefault,
  has_api_key: false, auth_method: 'api_key', credential_status: null, credential_error_code: null,
  account_email: null, scopes: [], access_token_expires_at: null, needs_reauthorization: false,
  created_at: '', updated_at: '',
})

describe('agent connection cache default invariant', () => {
  it('promotes a created authoritative default without changing order or duplicating rows', async () => {
    const client = new QueryClient()
    client.setQueryData(AGENT_CONNECTIONS_KEY, [connection('old-default', true), connection('other', false)])
    client.setQueryData([...AGENT_CONNECTIONS_KEY, 'old-default'], connection('old-default', true))

    await upsertAgentConnection(client, connection('created-default', true))

    expect(client.getQueryData<LlmConnection[]>(AGENT_CONNECTIONS_KEY)?.map(({ id, is_default }) => [id, is_default])).toEqual([
      ['old-default', false], ['other', false], ['created-default', true],
    ])
    expect(client.getQueryData<LlmConnection>([...AGENT_CONNECTIONS_KEY, 'old-default'])?.is_default).toBe(false)
    expect(client.getQueryData<LlmConnection>([...AGENT_CONNECTIONS_KEY, 'created-default'])?.is_default).toBe(true)
  })

  it('promotes an updated authoritative default in place and updates every exact cache', async () => {
    const client = new QueryClient()
    client.setQueryData(AGENT_CONNECTIONS_KEY, [connection('old-default', true), connection('promoted', false)])
    client.setQueryData([...AGENT_CONNECTIONS_KEY, 'old-default'], connection('old-default', true))
    client.setQueryData([...AGENT_CONNECTIONS_KEY, 'promoted'], connection('promoted', false))
    client.setQueryData([...AGENT_CONNECTIONS_KEY, 'orphan-default'], connection('orphan-default', true))

    await upsertAgentConnection(client, connection('promoted', true))

    expect(client.getQueryData<LlmConnection[]>(AGENT_CONNECTIONS_KEY)?.map(({ id, is_default }) => [id, is_default])).toEqual([
      ['old-default', false], ['promoted', true],
    ])
    expect(client.getQueryData<LlmConnection>([...AGENT_CONNECTIONS_KEY, 'old-default'])?.is_default).toBe(false)
    expect(client.getQueryData<LlmConnection>([...AGENT_CONNECTIONS_KEY, 'promoted'])?.is_default).toBe(true)
    expect(client.getQueryData<LlmConnection>([...AGENT_CONNECTIONS_KEY, 'orphan-default'])?.is_default).toBe(false)
  })

  it('replaces a non-default connection without demoting the existing default', async () => {
    const client = new QueryClient()
    client.setQueryData(AGENT_CONNECTIONS_KEY, [connection('default', true), connection('updated', false)])

    await upsertAgentConnection(client, { ...connection('updated', false), name: 'renamed' })

    expect(client.getQueryData<LlmConnection[]>(AGENT_CONNECTIONS_KEY)?.map(({ id, name, is_default }) => [id, name, is_default])).toEqual([
      ['default', 'default', true], ['updated', 'renamed', false],
    ])
  })

  it.each([
    ['signal-aware', true],
    ['signal-ignoring', false],
  ] as const)('keeps an authoritative upsert after a delayed %s root fetch settles', async (_name, observesSignal) => {
    const client = new QueryClient({ defaultOptions: { queries: { retry: false } } })
    client.setQueryData(AGENT_CONNECTIONS_KEY, [connection('old-default', true)])
    let resolveStale!: (value: LlmConnection[]) => void
    let observedSignal: AbortSignal | undefined
    const fetch = client.fetchQuery({
      queryKey: AGENT_CONNECTIONS_KEY,
      queryFn: ({ signal }) => {
        observedSignal = signal
        return new Promise<LlmConnection[]>((resolve, reject) => {
          resolveStale = resolve
          if (observesSignal) signal.addEventListener('abort', () => reject(new DOMException('Aborted', 'AbortError')))
        })
      },
    }).catch(() => undefined)

    await upsertAgentConnection(client, connection('created-default', true))
    expect(observedSignal?.aborted).toBe(true)
    resolveStale([connection('old-default', true)])
    await fetch

    expect(client.getQueryData<LlmConnection[]>(AGENT_CONNECTIONS_KEY)?.map(({ id, is_default }) => [id, is_default])).toEqual([
      ['old-default', false], ['created-default', true],
    ])
  })

  it('keeps an authoritative removal after a delayed signal-ignoring root fetch settles and removes only the exact detail', async () => {
    const client = new QueryClient({ defaultOptions: { queries: { retry: false } } })
    client.setQueryData(AGENT_CONNECTIONS_KEY, [connection('removed', false), connection('survivor', true)])
    client.setQueryData([...AGENT_CONNECTIONS_KEY, 'removed'], connection('removed', false))
    client.setQueryData([...AGENT_CONNECTIONS_KEY, 'survivor'], connection('survivor', true))
    let resolveStale!: (value: LlmConnection[]) => void
    const fetch = client.fetchQuery({
      queryKey: AGENT_CONNECTIONS_KEY,
      queryFn: () => new Promise<LlmConnection[]>((resolve) => { resolveStale = resolve }),
    }).catch(() => undefined)

    await removeAgentConnection(client, 'removed')
    resolveStale([connection('removed', false), connection('survivor', true)])
    await fetch

    expect(client.getQueryData<LlmConnection[]>(AGENT_CONNECTIONS_KEY)?.map(({ id }) => id)).toEqual(['survivor'])
    expect(client.getQueryData([...AGENT_CONNECTIONS_KEY, 'removed'])).toBeUndefined()
    expect(client.getQueryData([...AGENT_CONNECTIONS_KEY, 'survivor'])).toEqual(connection('survivor', true))
  })
})
