// @vitest-environment-options { "url": "https://securo.example/agents/connections" }
import { act, screen, waitFor } from '@testing-library/react'
import { beforeEach, expect, it, vi } from 'vitest'
import { createTestQueryClient, renderWithProviders } from '@/test/utils'
import type { LlmConnection, PairingStart } from '@/lib/api'
import { AGENT_CONNECTIONS_KEY } from '@/lib/agent-connections-cache'
import AgentConnectionsPage from './agent-connections'

const api = vi.hoisted(() => ({
  list: vi.fn(), get: vi.fn(), create: vi.fn(), update: vi.fn(), remove: vi.fn(),
  test: vi.fn(), revoke: vi.fn(), startPairing: vi.fn(), pairingStatus: vi.fn(),
  cancelPairing: vi.fn(),
}))
const toast = vi.hoisted(() => ({ success: vi.fn(), error: vi.fn() }))
vi.mock('@/lib/api', async (original) => {
  const actual = await original<typeof import('@/lib/api')>()
  return { ...actual, agents: { connections: api } }
})
vi.mock('@/components/agents/mcp-external-panel', () => ({ McpExternalPanel: () => null }))
vi.mock('sonner', () => ({ toast }))

const encoder = new TextEncoder()

function disconnected(overrides: Partial<LlmConnection> = {}): LlmConnection {
  return {
    id: 'connection-1', name: 'Lifecycle ChatGPT', kind: 'openai_chatgpt', base_url: null,
    default_model: null, extra: {}, is_default: false, has_api_key: false,
    auth_method: 'chatgpt_oauth', credential_status: 'not_connected', credential_error_code: null,
    account_email: null, scopes: [], access_token_expires_at: null, needs_reauthorization: false,
    created_at: '', updated_at: '', ...overrides,
  }
}

beforeEach(() => {
  vi.clearAllMocks()
  localStorage.clear()
  sessionStorage.clear()
  window.history.replaceState({}, '', '/agents/connections')
})

it('traces the whole ChatGPT connection lifecycle through the real page and dialogs', async () => {
  let rows: LlmConnection[] = []
  let pairNumber = 0
  const pending = new Map<string, (state: 'completed') => void>()
  const secrets: string[] = []

  api.list.mockImplementation(async () => rows.map((row) => ({ ...row })))
  api.create.mockImplementation(async (payload) => {
    const created = disconnected({ name: payload.name, is_default: payload.is_default })
    rows = [created]
    return { ...created }
  })
  api.get.mockImplementation(async (id) => {
    const row = rows.find((candidate) => candidate.id === id)
    if (!row) throw { response: { status: 404 } }
    return { ...row }
  })
  api.startPairing.mockImplementation(async (): Promise<PairingStart> => {
    pairNumber += 1
    const secret = `terminal-secret-${pairNumber}`
    secrets.push(secret)
    return {
      pairing_id: `pair-${pairNumber}`,
      secretBytes: encoder.encode(secret),
      expires_at: new Date(Date.now() + 60_000).toISOString(),
    }
  })
  api.pairingStatus.mockImplementation(async (_id, pairingId) => {
    if (pairingId === 'pair-2') throw { response: { status: 404, data: { detail: 'Bearer raw-offline-secret' } } }
    await new Promise<'completed'>((resolve) => pending.set(pairingId, resolve))
    rows = [disconnected({
      credential_status: 'active', account_email: 'paired@example.test',
      access_token_expires_at: '2030-01-01T00:00:00Z', scopes: ['openid'],
      default_model: rows[0]?.default_model ?? null, is_default: rows[0]?.is_default ?? false,
    })]
    return { pairing_id: pairingId, state: 'completed', expires_at: new Date(Date.now() + 60_000).toISOString(), failure_code: null }
  })
  api.test.mockResolvedValue({
    ok: true, detail: 'raw provider payload',
    models: [{ slug: 'gpt-integration', display_name: 'GPT Integration' }],
  })
  api.update.mockImplementation(async (id, patch) => {
    rows = rows.map((row) => row.id === id ? { ...row, ...patch } : row)
    return { ...rows.find((row) => row.id === id)! }
  })
  api.revoke.mockImplementation(async (id) => {
    rows = rows.map((row) => row.id === id ? disconnected({
      ...row, credential_status: 'revoked', account_email: null,
      access_token_expires_at: null, scopes: [],
    }) : row)
  })
  api.cancelPairing.mockResolvedValue(undefined)
  api.remove.mockImplementation(async (id) => { rows = rows.filter((row) => row.id !== id) })

  const queryClient = createTestQueryClient()
  const view = renderWithProviders(
    <AgentConnectionsPage />,
    { queryClient },
  )

  await view.user.click(await screen.findByRole('button', { name: 'Add connection' }))
  await view.user.type(screen.getByLabelText('Name'), 'Lifecycle ChatGPT')
  await view.user.click(screen.getByRole('combobox', { name: 'Provider type' }))
  await view.user.click(await screen.findByRole('option', { name: 'Sign in with ChatGPT' }))
  await view.user.click(screen.getByRole('button', { name: 'Create and continue' }))

  expect(await screen.findByText('terminal-secret-1')).toBeVisible()
  expect(screen.getByText('Waiting for the helper…')).toBeVisible()
  await act(async () => pending.get('pair-1')?.('completed'))
  const model = await screen.findByRole('combobox', { name: 'Choose a ChatGPT model' })
  expect(screen.queryByText('terminal-secret-1')).not.toBeInTheDocument()
  await view.user.click(model)
  await view.user.click(await screen.findByRole('option', { name: 'GPT Integration' }))
  await waitFor(() => expect(api.update).toHaveBeenCalledWith('connection-1', { default_model: 'gpt-integration' }))
  await view.user.click(screen.getAllByRole('button', { name: 'Close' })[0])

  await view.user.click(await screen.findByRole('button', { name: 'Test' }))
  expect(await screen.findByText('ChatGPT connection is working.')).toBeVisible()
  await view.user.click(screen.getByRole('button', { name: 'Edit' }))
  await view.user.click(screen.getByRole('switch', { name: /default/i }))
  await view.user.click(screen.getByRole('button', { name: 'Save' }))
  await waitFor(() => expect(api.update).toHaveBeenCalledWith('connection-1', expect.objectContaining({ is_default: true })))

  await view.user.click(await screen.findByRole('button', { name: 'Reconnect' }))
  await view.user.click(screen.getByRole('button', { name: 'Revoke and reconnect' }))
  expect(await screen.findByText('Pairing expired. Start again.')).toBeVisible()
  expect(screen.queryByText('terminal-secret-2')).not.toBeInTheDocument()
  expect(document.body.textContent).not.toContain('raw-offline-secret')
  await view.user.click(screen.getByRole('button', { name: 'Cancel' }))
  await view.user.click(await screen.findByRole('button', { name: 'Sign in' }))
  expect(await screen.findByText('terminal-secret-3')).toBeVisible()
  await act(async () => pending.get('pair-3')?.('completed'))
  await screen.findByRole('combobox', { name: 'Choose a ChatGPT model' })
  await view.user.click(screen.getAllByRole('button', { name: 'Close' })[0])

  let resolveStale!: (value: LlmConnection[]) => void
  const staleSnapshot = rows.map((row) => ({ ...row }))
  const staleFetch = queryClient.fetchQuery({
    queryKey: AGENT_CONNECTIONS_KEY,
    queryFn: () => new Promise<LlmConnection[]>((resolve) => { resolveStale = resolve }),
  }).catch(() => undefined)
  await waitFor(() => expect(queryClient.getQueryState(AGENT_CONNECTIONS_KEY)?.fetchStatus).toBe('fetching'))

  await view.user.click(screen.getByRole('button', { name: 'Delete' }))
  await view.user.click(screen.getAllByRole('button', { name: 'Delete' }).at(-1)!)
  expect(await screen.findByText('No connections yet')).toBeVisible()
  resolveStale(staleSnapshot)
  await staleFetch
  expect(screen.getByText('No connections yet')).toBeVisible()

  const retained = JSON.stringify({
    queries: queryClient.getQueryCache().getAll().map((query) => query.state.data),
    mutations: queryClient.getMutationCache().getAll().map((mutation) => mutation.state.data),
  })
  for (const secret of secrets) {
    expect(document.body.textContent).not.toContain(secret)
    expect(window.location.href).not.toContain(secret)
    expect(localStorage.getItem(secret)).toBeNull()
    expect(sessionStorage.getItem(secret)).toBeNull()
    expect(retained).not.toContain(secret)
  }
  expect(api.revoke).toHaveBeenCalledTimes(2)
  expect(api.remove).toHaveBeenCalledWith('connection-1')
}, 60_000)
