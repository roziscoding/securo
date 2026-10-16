import { act, fireEvent, screen, waitFor } from '@testing-library/react'
import { beforeEach, expect, it, vi } from 'vitest'
import { createTestQueryClient, renderWithProviders } from '@/test/utils'
import AgentConnectionsPage from './agent-connections'
import type { LlmConnection } from '@/lib/api'
import { AGENT_CONNECTIONS_KEY } from '@/lib/agent-connections-cache'

const api = vi.hoisted(() => ({ list: vi.fn(), get: vi.fn(), revoke: vi.fn(), remove: vi.fn(), test: vi.fn() }))
const toast = vi.hoisted(() => ({ success: vi.fn(), error: vi.fn() }))
let canWrite = true
vi.mock('@/lib/api', async (original) => {
  const actual = await original<typeof import('@/lib/api')>()
  return { ...actual, agents: { connections: api } }
})
vi.mock('@/components/agents/mcp-external-panel', () => ({ McpExternalPanel: () => null }))
vi.mock('@/components/agents/connection-form-dialog', () => ({ ConnectionFormDialog: () => null }))
vi.mock('@/components/agents/chatgpt-pairing-dialog', () => ({
  ChatGptPairingDialog: ({ open }: { open: boolean }) => open ? <div data-testid="pairing-open" /> : null,
}))
vi.mock('sonner', () => ({ toast }))
vi.mock('@/contexts/workspace-context', () => ({ useWorkspace: () => ({ canWrite }) }))

const active: LlmConnection = {
  id: 'c1', name: 'ChatGPT', kind: 'openai_chatgpt', base_url: null, default_model: 'gpt-test', extra: {},
  is_default: true, has_api_key: false, auth_method: 'chatgpt_oauth', credential_status: 'active',
  credential_error_code: null, account_email: 'person@example.test', scopes: ['openid'],
  access_token_expires_at: '2030-01-01T00:00:00Z', needs_reauthorization: false, created_at: '', updated_at: '',
}

beforeEach(() => {
  vi.clearAllMocks()
  canWrite = true
  api.list.mockResolvedValue([active])
  api.revoke.mockResolvedValue(undefined)
  api.get.mockResolvedValue({ ...active, credential_status: 'revoked' })
  api.remove.mockResolvedValue(undefined)
  api.test.mockResolvedValue({ ok: true, detail: 'provider detail' })
})

it('disables Add until the initial connection list request settles', async () => {
  let resolveList!: (value: LlmConnection[]) => void
  api.list.mockReturnValue(new Promise((resolve) => { resolveList = resolve }))
  renderWithProviders(<AgentConnectionsPage />)

  const add = screen.getByRole('button', { name: 'Add connection' })
  expect(add).toBeDisabled()
  resolveList([active])
  await waitFor(() => expect(add).not.toBeDisabled())
})

it('lets read-only workspace members manage their user-owned credentials while showing only sanitized metadata', async () => {
  canWrite = false
  api.list.mockResolvedValue([
    active,
    { ...active, id: 'c2', name: 'Disconnected ChatGPT', is_default: false, credential_status: 'not_connected' },
  ])
  renderWithProviders(<AgentConnectionsPage />)

  expect(await screen.findAllByText('person@example.test')).toHaveLength(2)
  expect(screen.getAllByText('gpt-test')).toHaveLength(2)
  for (const name of ['Add connection', 'Test', 'Edit', 'Delete', 'Reconnect', 'Disconnect', 'Sign in']) {
    expect(screen.getAllByRole('button', { name }).length).toBeGreaterThan(0)
  }
  expect(document.body.textContent).not.toContain('access_token')
  expect(document.body.textContent).not.toContain('refresh_token')
})

it('deduplicates repeated Test clicks while a request is pending', async () => {
  let resolveTest!: (value: { ok: boolean; detail: string }) => void
  api.test.mockReturnValue(new Promise((resolve) => { resolveTest = resolve }))
  const view = renderWithProviders(<AgentConnectionsPage />)
  const button = await screen.findByRole('button', { name: 'Test' })
  await view.user.click(button)
  expect(button).toBeDisabled()
  fireEvent.click(button)
  expect(api.test).toHaveBeenCalledTimes(1)

  resolveTest({ ok: true, detail: 'provider detail' })
  expect(await screen.findByText('ChatGPT connection is working.')).toBeVisible()
  await waitFor(() => expect(button).not.toBeDisabled())
})

it('ignores an old Test result after an authoritative list refresh starts a newer request', async () => {
  let resolveOld!: (value: { ok: boolean; detail: string }) => void
  let resolveNew!: (value: { ok: boolean; detail: string }) => void
  api.test
    .mockReturnValueOnce(new Promise((resolve) => { resolveOld = resolve }))
    .mockReturnValueOnce(new Promise((resolve) => { resolveNew = resolve }))
  const queryClient = createTestQueryClient()
  const view = renderWithProviders(<AgentConnectionsPage />, { queryClient })
  const firstButton = await screen.findByRole('button', { name: 'Test' })
  await view.user.click(firstButton)

  act(() => {
    queryClient.setQueryData(['agent-connections'], [{ ...active, name: 'Refreshed ChatGPT' }])
  })
  const nextButton = await screen.findByRole('button', { name: 'Test' })
  await waitFor(() => expect(nextButton).not.toBeDisabled())
  await view.user.click(nextButton)
  expect(api.test).toHaveBeenCalledTimes(2)

  resolveNew({ ok: true, detail: 'new provider detail' })
  expect(await screen.findByText('ChatGPT connection is working.')).toBeVisible()
  resolveOld({ ok: false, detail: 'old provider detail' })
  await act(async () => { await Promise.resolve() })
  expect(screen.getByText('ChatGPT connection is working.')).toBeVisible()
  expect(document.body.textContent).not.toContain('old provider detail')
})

it('synchronously guards confirmation actions against duplicate destructive requests', async () => {
  let resolveRevoke!: () => void
  api.revoke.mockReturnValue(new Promise<void>((resolve) => { resolveRevoke = resolve }))
  const view = renderWithProviders(<AgentConnectionsPage />)
  await view.user.click(await screen.findByRole('button', { name: 'Delete' }))
  const confirm = screen.getAllByRole('button', { name: 'Delete' }).at(-1)!

  act(() => {
    confirm.click()
    confirm.click()
  })
  await waitFor(() => expect(api.revoke).toHaveBeenCalledTimes(1))
  expect(confirm).toBeDisabled()
  resolveRevoke()
  await waitFor(() => expect(api.remove).toHaveBeenCalledTimes(1))
})

it('reconnects only after revoke and authoritative refetch show a pairable state', async () => {
  const view = renderWithProviders(<AgentConnectionsPage />)
  await view.user.click(await screen.findByRole('button', { name: 'Reconnect' }))
  await view.user.click(screen.getByRole('button', { name: 'Revoke and reconnect' }))

  await waitFor(() => expect(screen.getByTestId('pairing-open')).toBeInTheDocument())
  expect(api.revoke).toHaveBeenCalledWith('c1')
  expect(api.get).toHaveBeenCalledWith('c1')
  expect(api.revoke.mock.invocationCallOrder[0]).toBeLessThan(api.get.mock.invocationCallOrder[0])
})

it('does not start pairing when revoke outcome remains active', async () => {
  api.revoke.mockRejectedValue(new Error('raw provider detail'))
  api.get.mockResolvedValue(active)
  const view = renderWithProviders(<AgentConnectionsPage />)
  await view.user.click(await screen.findByRole('button', { name: 'Reconnect' }))
  await view.user.click(screen.getByRole('button', { name: 'Revoke and reconnect' }))

  await waitFor(() => expect(api.get).toHaveBeenCalled())
  expect(screen.queryByTestId('pairing-open')).not.toBeInTheDocument()
  expect(document.body.textContent).not.toContain('raw provider detail')
})

it('always refetches after a failed revoke and reconnects only when the refetch is pairable', async () => {
  api.revoke.mockRejectedValue(new Error('provider secret'))
  api.get.mockResolvedValue({ ...active, credential_status: 'revoked' })
  const view = renderWithProviders(<AgentConnectionsPage />)
  await view.user.click(await screen.findByRole('button', { name: 'Reconnect' }))
  await view.user.click(screen.getByRole('button', { name: 'Revoke and reconnect' }))

  await waitFor(() => expect(screen.getByTestId('pairing-open')).toBeInTheDocument())
  expect(api.get).toHaveBeenCalledWith('c1')
  expect(document.body.textContent).not.toContain('provider secret')
})

it('disconnects an active credential without starting a new pairing', async () => {
  const view = renderWithProviders(<AgentConnectionsPage />)
  await view.user.click(await screen.findByRole('button', { name: 'Disconnect' }))
  await view.user.click(screen.getAllByRole('button', { name: 'Disconnect' }).at(-1)!)

  await waitFor(() => expect(api.get).toHaveBeenCalledWith('c1'))
  expect(api.revoke).toHaveBeenCalledWith('c1')
  expect(screen.queryByTestId('pairing-open')).not.toBeInTheDocument()
  expect(toast.success).toHaveBeenCalledWith('ChatGPT disconnected; connection settings were kept.')
})

it('stops revoke-then-delete when refetch is cleanup-blocked', async () => {
  api.get.mockResolvedValue({ ...active, credential_status: 'revoked', credential_error_code: 'resource_cleanup_pending' })
  const view = renderWithProviders(<AgentConnectionsPage />)
  await view.user.click(await screen.findByRole('button', { name: 'Delete' }))
  await view.user.click(screen.getAllByRole('button', { name: 'Delete' }).at(-1)!)

  await waitFor(() => expect(api.get).toHaveBeenCalledWith('c1'))
  expect(api.remove).not.toHaveBeenCalled()
  expect(toast.error).toHaveBeenCalledWith('The provider session outcome is not confirmed. No further action was taken.')
})

it('deletes only after revoke and authoritative pairable refetch', async () => {
  const view = renderWithProviders(<AgentConnectionsPage />)
  await view.user.click(await screen.findByRole('button', { name: 'Delete' }))
  await view.user.click(screen.getAllByRole('button', { name: 'Delete' }).at(-1)!)
  await waitFor(() => expect(api.remove).toHaveBeenCalledWith('c1'))
  expect(api.revoke.mock.invocationCallOrder[0]).toBeLessThan(api.get.mock.invocationCallOrder[0])
  expect(api.get.mock.invocationCallOrder[0]).toBeLessThan(api.remove.mock.invocationCallOrder[0])
})

it('patches the rendered list from exact GET when the broad disconnect refetch fails', async () => {
  api.list.mockResolvedValueOnce([active]).mockRejectedValueOnce(new Error('list unavailable'))
  api.get.mockResolvedValue({ ...active, credential_status: 'revoked', account_email: null })
  const queryClient = createTestQueryClient()
  const view = renderWithProviders(<AgentConnectionsPage />, { queryClient })
  await view.user.click(await screen.findByRole('button', { name: 'Disconnect' }))
  await view.user.click(screen.getAllByRole('button', { name: 'Disconnect' }).at(-1)!)

  expect(await screen.findByText('Disconnected')).toBeVisible()
  await queryClient.refetchQueries({ queryKey: ['agent-connections'] })
  expect(screen.getByRole('button', { name: 'Sign in' })).toBeVisible()
  expect(api.get).toHaveBeenCalledWith('c1')
})

it('keeps revoke reconciliation after a delayed root list fetch resolves stale', async () => {
  const revoked = { ...active, credential_status: 'revoked' as const, account_email: null }
  api.get.mockResolvedValue(revoked)
  const queryClient = createTestQueryClient()
  const view = renderWithProviders(<AgentConnectionsPage />, { queryClient })
  await screen.findByRole('button', { name: 'Disconnect' })
  let resolveStale!: (value: LlmConnection[]) => void
  let staleSignal: AbortSignal | undefined
  const staleFetch = queryClient.fetchQuery({
    queryKey: AGENT_CONNECTIONS_KEY,
    queryFn: ({ signal }) => {
      staleSignal = signal
      return new Promise<LlmConnection[]>((resolve) => { resolveStale = resolve })
    },
  }).catch(() => undefined)

  await view.user.click(screen.getByRole('button', { name: 'Disconnect' }))
  await view.user.click(screen.getAllByRole('button', { name: 'Disconnect' }).at(-1)!)
  await waitFor(() => expect(api.get).toHaveBeenCalledWith('c1'))
  expect(staleSignal?.aborted).toBe(true)
  resolveStale([active])
  await staleFetch

  expect(await screen.findByText('Disconnected')).toBeVisible()
  expect(screen.getByRole('button', { name: 'Sign in' })).toBeVisible()
})

it('removes a deleted connection from the rendered list even when list refetch fails', async () => {
  api.list.mockResolvedValueOnce([active]).mockRejectedValueOnce(new Error('list unavailable'))
  const queryClient = createTestQueryClient()
  const view = renderWithProviders(<AgentConnectionsPage />, { queryClient })
  await view.user.click(await screen.findByRole('button', { name: 'Delete' }))
  await view.user.click(screen.getAllByRole('button', { name: 'Delete' }).at(-1)!)

  await waitFor(() => expect(api.remove).toHaveBeenCalledWith('c1'))
  await queryClient.refetchQueries({ queryKey: ['agent-connections'] })
  expect(await screen.findByText('No connections yet')).toBeVisible()
})

it('keeps a successful delete removed after a delayed signal-ignoring root list fetch settles', async () => {
  const plain = { ...active, kind: 'openai' as const, credential_status: null }
  api.list.mockResolvedValue([plain])
  const queryClient = createTestQueryClient()
  const view = renderWithProviders(<AgentConnectionsPage />, { queryClient })
  await screen.findByRole('button', { name: 'Delete' })
  let resolveStale!: (value: LlmConnection[]) => void
  const staleFetch = queryClient.fetchQuery({
    queryKey: AGENT_CONNECTIONS_KEY,
    queryFn: () => new Promise<LlmConnection[]>((resolve) => { resolveStale = resolve }),
  }).catch(() => undefined)

  await view.user.click(screen.getByRole('button', { name: 'Delete' }))
  await view.user.click(screen.getAllByRole('button', { name: 'Delete' }).at(-1)!)
  await waitFor(() => expect(api.remove).toHaveBeenCalledWith('c1'))
  resolveStale([plain])
  await staleFetch

  expect(await screen.findByText('No connections yet')).toBeVisible()
})

it.each([
  ['ordinary', { ...active, kind: 'openai', credential_status: null }],
  ['ChatGPT revoke-then-delete', active],
] as const)('treats a reconciled 404 after ambiguous %s DELETE as committed', async (_name, target) => {
  api.list.mockResolvedValue([target])
  api.remove.mockRejectedValue(new Error('ambiguous delete'))
  api.get.mockReset()
  if (target.kind === 'openai_chatgpt') api.get.mockResolvedValueOnce({ ...active, credential_status: 'revoked' })
  api.get.mockRejectedValueOnce({ response: { status: 404, data: { detail: 'raw detail' } } })
  const queryClient = createTestQueryClient()
  const view = renderWithProviders(<AgentConnectionsPage />, { queryClient })
  await view.user.click(await screen.findByRole('button', { name: 'Delete' }))
  await view.user.click(screen.getAllByRole('button', { name: 'Delete' }).at(-1)!)

  expect(await screen.findByText('No connections yet')).toBeVisible()
  expect(queryClient.getQueryData(['agent-connections', 'c1'])).toBeUndefined()
  expect(toast.success).toHaveBeenCalledWith('Connection removed')
  expect(document.body.textContent).not.toContain('raw detail')
  expect(api.remove).toHaveBeenCalledTimes(1)
})

it.each([
  ['ordinary', { ...active, kind: 'openai', credential_status: null }],
  ['ChatGPT revoke-then-delete', active],
] as const)('restores an authoritative survivor after ambiguous %s DELETE and shows fixed failure', async (_name, target) => {
  const survivor = { ...target, name: 'Authoritative survivor' }
  api.list.mockResolvedValue([target])
  api.remove.mockRejectedValue(new Error('raw ambiguous delete'))
  api.get.mockReset()
  if (target.kind === 'openai_chatgpt') api.get.mockResolvedValueOnce({ ...active, credential_status: 'revoked' })
  api.get.mockResolvedValueOnce(survivor)
  const view = renderWithProviders(<AgentConnectionsPage />)
  await view.user.click(await screen.findByRole('button', { name: 'Delete' }))
  await view.user.click(screen.getAllByRole('button', { name: 'Delete' }).at(-1)!)

  expect(await screen.findByText('Authoritative survivor')).toBeVisible()
  expect(toast.error).toHaveBeenCalledWith('Could not remove connection')
  expect(document.body.textContent).not.toContain('raw ambiguous delete')
  expect(api.remove).toHaveBeenCalledTimes(1)
})

it.each([
  ['ordinary', { ...active, kind: 'openai', credential_status: null }],
  ['ChatGPT revoke-then-delete', active],
] as const)('preserves the item when ambiguous %s DELETE reconciliation is inconclusive', async (_name, target) => {
  api.list.mockResolvedValue([target])
  api.remove.mockRejectedValue(new Error('raw ambiguous delete'))
  api.get.mockReset()
  if (target.kind === 'openai_chatgpt') api.get.mockResolvedValueOnce({ ...active, credential_status: 'revoked' })
  api.get.mockRejectedValueOnce(new Error('raw reconciliation failure'))
  const view = renderWithProviders(<AgentConnectionsPage />)
  await view.user.click(await screen.findByRole('button', { name: 'Delete' }))
  await view.user.click(screen.getAllByRole('button', { name: 'Delete' }).at(-1)!)

  expect((await screen.findAllByText(target.name)).length).toBeGreaterThan(0)
  await waitFor(() => expect(toast.error).toHaveBeenCalledWith('Could not confirm whether the connection was removed. Refresh before trying again.'))
  expect(document.body.textContent).not.toContain('raw reconciliation failure')
  expect(api.remove).toHaveBeenCalledTimes(1)
})

it('uses safe status/test copy, displays metadata, and leaves user-owned controls reachable', async () => {
  api.list.mockResolvedValue([{ ...active, credential_status: 'unexpected', credential_error_code: 'untranslated-secret' }])
  const view = renderWithProviders(<AgentConnectionsPage />)
  expect(await screen.findByText('Not connected')).toBeVisible()
  expect(screen.getByText('person@example.test')).toBeVisible()
  expect(screen.getByText('gpt-test')).toBeVisible()
  expect(screen.getByRole('button', { name: 'Test' })).toBeVisible()
  expect(screen.getByRole('button', { name: 'Edit' })).toBeVisible()
  expect(screen.getByRole('button', { name: 'Delete' })).toBeVisible()
  expect(document.body.textContent).not.toContain('untranslated-secret')
  await view.user.click(screen.getByRole('button', { name: 'Test' }))
  expect(await screen.findByText('ChatGPT connection is working.')).toBeVisible()
  expect(document.body.textContent).not.toContain('provider detail')
})
