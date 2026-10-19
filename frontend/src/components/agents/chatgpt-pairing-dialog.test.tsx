import { StrictMode } from 'react'
import { act, screen, waitFor } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { QueryClient } from '@tanstack/react-query'
import { createTestQueryClient, renderWithProviders } from '@/test/utils'
import { ChatGptPairingDialog } from './chatgpt-pairing-dialog'
import { normalizeOAuthModels } from './chatgpt-pairing-utils'
import type { LlmConnection, PairingStart } from '@/lib/api'
import { AGENT_CONNECTIONS_KEY } from '@/lib/agent-connections-cache'

const api = vi.hoisted(() => ({
  startPairing: vi.fn(), pairingStatus: vi.fn(), cancelPairing: vi.fn(), test: vi.fn(), update: vi.fn(), get: vi.fn(),
}))
const toast = vi.hoisted(() => ({ success: vi.fn(), error: vi.fn() }))
vi.mock('@/lib/api', async (original) => {
  const actual = await original<typeof import('@/lib/api')>()
  return { ...actual, agents: { connections: api } }
})
vi.mock('sonner', () => ({ toast }))

const secureLocation = { protocol: 'https:', origin: 'https://securo.example' } as Location
const connection: LlmConnection = {
  id: 'connection-1', name: 'ChatGPT', kind: 'openai_chatgpt', base_url: null, default_model: null,
  extra: {}, is_default: false, has_api_key: false, auth_method: 'chatgpt_oauth', credential_status: 'not_connected',
  credential_error_code: null, account_email: null, scopes: [], access_token_expires_at: null,
  needs_reauthorization: false, created_at: '', updated_at: '',
}
const encoder = new TextEncoder()
const decoder = new TextDecoder()
const helperCommand = "curl --disable --noproxy '*' --proto '=https' --tlsv1.2 --fail --silent --show-error https://securo.example/api/agents/chatgpt/securo-siwc.py | python3 - pair-1 --server https://securo.example"
const started = (overrides: { pairing_id?: string; secret?: string; expires_at?: string; helper_command?: string } = {}): PairingStart => ({
  pairing_id: overrides.pairing_id ?? 'pair-1',
  secretBytes: encoder.encode(overrides.secret ?? 'one-time-secret'),
  expires_at: overrides.expires_at ?? new Date(Date.now() + 60_000).toISOString(),
  helper_command: overrides.helper_command ?? helperCommand,
})

type Fiber = Record<string, unknown> & {
  alternate?: Fiber | null; child?: Fiber | null; sibling?: Fiber | null; return?: Fiber | null
}

function fiberFor(node: Element): Fiber {
  const key = Object.keys(node).find((candidate) => candidate.startsWith('__reactFiber$'))
  if (!key) throw new Error('React Fiber not found')
  return (node as unknown as Record<string, Fiber>)[key]
}

function fiberRetains(start: Fiber, sentinel: string): string[] {
  let root = start
  while (root.return) root = root.return
  const hits: string[] = []
  const seenFibers = new Set<object>()
  const seenValues = new Set<object>()
  const inspect = (value: unknown, path: string) => {
    if (typeof value === 'string') {
      if (value.includes(sentinel)) hits.push(path)
      return
    }
    if (value instanceof Uint8Array) {
      if (decoder.decode(value).includes(sentinel)) hits.push(path)
      return
    }
    if (!value || typeof value !== 'object' || seenValues.has(value)) return
    seenValues.add(value)
    for (const [key, child] of Object.entries(value)) {
      if (key === 'stateNode' || key === 'return' || key === 'child' || key === 'sibling' || key === 'alternate') continue
      inspect(child, `${path}.${key}`)
    }
  }
  const pending = [root]
  while (pending.length) {
    const fiber = pending.pop()!
    if (seenFibers.has(fiber)) continue
    seenFibers.add(fiber)
    for (const field of ['pendingProps', 'memoizedProps', 'memoizedState', 'baseState', 'updateQueue'] as const) {
      inspect(fiber[field], `fiber.${field}`)
    }
    for (const next of [fiber.alternate, fiber.child, fiber.sibling]) if (next) pending.push(next)
  }
  return hits
}

beforeEach(() => {
  vi.clearAllMocks()
  window.history.replaceState({}, '', '/')
  localStorage.clear()
  sessionStorage.clear()
  api.cancelPairing.mockResolvedValue(undefined)
  api.pairingStatus.mockReturnValue(new Promise(() => undefined))
  api.test.mockResolvedValue({ ok: true, detail: 'raw detail', models: [] })
  api.get.mockResolvedValue(connection)
})
afterEach(() => vi.useRealTimers())

async function settleStart(value = started()) {
  api.startPairing.mockResolvedValue(value)
  return value
}

describe('secure ChatGPT pairing', () => {
  it('normalizes both model contracts', () => {
    expect(normalizeOAuthModels(['gpt-a', { slug: 'gpt-b', display_name: 'GPT B' }])).toEqual([
      { slug: 'gpt-a', display_name: 'gpt-a' }, { slug: 'gpt-b', display_name: 'GPT B' },
    ])
  })

  it('starts exactly once in StrictMode and never puts the secret in caches, storage, URL, logs, toasts, or requests', async () => {
    const result = await settleStart()
    const queryClient = createTestQueryClient()
    const log = vi.spyOn(console, 'log').mockImplementation(() => undefined)
    const error = vi.spyOn(console, 'error').mockImplementation(() => undefined)
    renderWithProviders(
      <StrictMode><ChatGptPairingDialog browserLocation={secureLocation} open connection={connection} onOpenChange={vi.fn()} /></StrictMode>,
      { queryClient },
    )

    expect(await screen.findByText('one-time-secret')).toBeVisible()
    expect(api.startPairing).toHaveBeenCalledTimes(1)
    expect(decoder.decode(result.secretBytes)).toBe('one-time-secret')
    expect(JSON.stringify(queryClient.getQueryCache().getAll())).not.toContain('one-time-secret')
    expect(JSON.stringify(queryClient.getMutationCache().getAll())).not.toContain('one-time-secret')
    expect(localStorage.getItem('one-time-secret')).toBeNull()
    expect(sessionStorage.getItem('one-time-secret')).toBeNull()
    expect(window.location.href).not.toContain('one-time-secret')
    expect(JSON.stringify(api.startPairing.mock.calls)).not.toContain('one-time-secret')
    expect(JSON.stringify(api.pairingStatus.mock.calls)).not.toContain('one-time-secret')
    expect(JSON.stringify(toast)).not.toContain('one-time-secret')
    expect(JSON.stringify(log.mock.calls)).not.toContain('one-time-secret')
    expect(JSON.stringify(error.mock.calls)).not.toContain('one-time-secret')
    log.mockRestore()
    error.mockRestore()
  })

  it('does not start on an insecure origin', async () => {
    expect(window.location.protocol).toBe('http:')
    renderWithProviders(<ChatGptPairingDialog open connection={connection} onOpenChange={vi.fn()} />)
    expect(await screen.findByRole('alert')).toHaveTextContent('HTTPS')
    expect(api.startPairing).not.toHaveBeenCalled()
  })

  it('blocks every dismissal path while the one-shot start is pending', async () => {
    api.startPairing.mockReturnValue(new Promise(() => undefined))
    const onOpenChange = vi.fn()
    const view = renderWithProviders(<ChatGptPairingDialog browserLocation={secureLocation} open connection={connection} onOpenChange={onOpenChange} />)
    await waitFor(() => expect(api.startPairing).toHaveBeenCalledTimes(1))
    expect(screen.getByRole('button', { name: 'Cancel' })).toBeDisabled()
    expect(screen.queryByRole('button', { name: 'Close' })).not.toBeInTheDocument()
    await view.user.keyboard('{Escape}{Escape}')
    await view.user.click(screen.getByRole('button', { name: 'Cancel' }))
    expect(onOpenChange).not.toHaveBeenCalled()
    expect(api.startPairing).toHaveBeenCalledTimes(1)
  })

  it('scrubs synchronously, reconciles the exact connection, and sends at most one DELETE across cancel and unmount', async () => {
    const result = await settleStart()
    const onOpenChange = vi.fn()
    const onReconciled = vi.fn()
    const view = renderWithProviders(<ChatGptPairingDialog browserLocation={secureLocation} open connection={connection} onOpenChange={onOpenChange} onReconciled={onReconciled} />)
    expect(await screen.findByText('one-time-secret')).toBeVisible()
    await view.user.click(screen.getByRole('button', { name: 'Cancel' }))
    expect(screen.queryByText('one-time-secret')).not.toBeInTheDocument()
    expect(result.secretBytes.every((byte) => byte === 0)).toBe(true)
    await waitFor(() => expect(onOpenChange).toHaveBeenCalledWith(false))
    expect(api.get).toHaveBeenCalledWith('connection-1')
    expect(onReconciled).toHaveBeenCalledWith(connection)
    view.unmount()
    await waitFor(() => expect(api.cancelPairing).toHaveBeenCalledTimes(1))
    expect(api.cancelPairing).toHaveBeenCalledWith('connection-1', 'pair-1')
  })

  it.each([
    ['completed', null], ['failed', 'invalid_client'], ['cancelled', null],
  ] as const)('scrubs on the %s terminal state without cancelling a terminal pairing', async (state, failureCode) => {
    await settleStart()
    api.pairingStatus.mockResolvedValue({ pairing_id: 'pair-1', state, expires_at: started().expires_at, failure_code: failureCode })
    renderWithProviders(<ChatGptPairingDialog browserLocation={secureLocation} open connection={connection} onOpenChange={vi.fn()} />)
    await waitFor(() => expect(api.pairingStatus).toHaveBeenCalled())
    await waitFor(() => expect(screen.queryByText('one-time-secret')).not.toBeInTheDocument())
    expect(api.cancelPairing).not.toHaveBeenCalled()
    expect(document.body.textContent).not.toContain('raw detail')
  })

  it('scrubs on terminal 404', async () => {
    await settleStart()
    api.pairingStatus.mockRejectedValue({ response: { status: 404, data: { detail: 'secret server detail' } } })
    renderWithProviders(<ChatGptPairingDialog browserLocation={secureLocation} open connection={connection} onOpenChange={vi.fn()} />)
    expect(await screen.findByText('Pairing expired. Start again.')).toBeVisible()
    expect(screen.queryByText('one-time-secret')).not.toBeInTheDocument()
    expect(document.body.textContent).not.toContain('secret server detail')
    expect(api.cancelPairing).not.toHaveBeenCalled()
  })

  it('scrubs at client expiry even when a status request never settles', async () => {
    vi.useFakeTimers()
    await settleStart(started({ pairing_id: 'expires', expires_at: new Date(Date.now() + 1000).toISOString() }))
    const view = renderWithProviders(<ChatGptPairingDialog browserLocation={secureLocation} open connection={connection} onOpenChange={vi.fn()} />)
    await act(async () => { await vi.advanceTimersByTimeAsync(0) })
    expect(screen.getByText('one-time-secret')).toBeVisible()
    await act(async () => { await vi.advanceTimersByTimeAsync(1000) })
    expect(screen.queryByText('one-time-secret')).not.toBeInTheDocument()
    expect(api.cancelPairing).toHaveBeenCalledWith('connection-1', 'expires')
    view.unmount()
    expect(api.cancelPairing.mock.calls.filter((call) => call[1] === 'expires')).toHaveLength(1)
  })

  it('labels the dialog, focuses its heading, and copies command and secret separately', async () => {
    const clipboard = vi.spyOn(navigator.clipboard, 'writeText').mockResolvedValue(undefined)
    await settleStart()
    const view = renderWithProviders(<ChatGptPairingDialog browserLocation={secureLocation} open connection={connection} onOpenChange={vi.fn()} />)
    const dialog = await screen.findByRole('dialog', { name: 'Sign in with ChatGPT' })
    expect(dialog).toHaveAccessibleDescription()
    expect(screen.getByRole('heading', { name: 'Sign in with ChatGPT' })).toHaveFocus()
    await screen.findByText('one-time-secret')
    await view.user.click(screen.getByRole('button', { name: 'Copy helper command' }))
    await view.user.click(screen.getByRole('button', { name: 'Copy one-time secret' }))
    expect(clipboard.mock.calls).toEqual([
      [helperCommand],
      ['one-time-secret'],
    ])
    clipboard.mockRestore()
  })

  it('cancels a stale late start response and never exposes it after close/reopen', async () => {
    let resolveFirst!: (value: PairingStart) => void
    api.startPairing.mockReturnValueOnce(new Promise((resolve) => { resolveFirst = resolve })).mockResolvedValueOnce(started({ pairing_id: 'fresh', secret: 'fresh-secret' }))
    const view = renderWithProviders(<ChatGptPairingDialog browserLocation={secureLocation} open connection={connection} onOpenChange={vi.fn()} />)
    await waitFor(() => expect(api.startPairing).toHaveBeenCalledTimes(1))
    view.rerender(<ChatGptPairingDialog open={false} connection={connection} onOpenChange={vi.fn()} />)
    view.rerender(<ChatGptPairingDialog browserLocation={secureLocation} open connection={connection} onOpenChange={vi.fn()} />)
    await waitFor(() => expect(api.startPairing).toHaveBeenCalledTimes(2))
    resolveFirst(started({ pairing_id: 'late', secret: 'late-secret' }))
    await waitFor(() => expect(api.cancelPairing).toHaveBeenCalledWith('connection-1', 'late'))
    expect(document.body.textContent).not.toContain('late-secret')
    view.unmount()
  })

  it('does not restart or cancel when parent rerenders change callbacks before or after completion', async () => {
    await settleStart()
    let resolveStatus!: (value: { pairing_id: string; state: 'completed'; expires_at: string; failure_code: null }) => void
    api.pairingStatus.mockReturnValue(new Promise((resolve) => { resolveStatus = resolve }))
    api.test.mockResolvedValue({ ok: true, detail: 'safe', models: [{ slug: 'gpt-stable', display_name: 'GPT Stable' }] })
    const firstCompleted = vi.fn()
    const secondCompleted = vi.fn()
    const thirdCompleted = vi.fn()
    const view = renderWithProviders(
      <ChatGptPairingDialog browserLocation={secureLocation} open connection={connection} onOpenChange={vi.fn()} onCompleted={firstCompleted} />,
    )
    expect(await screen.findByText('one-time-secret')).toBeVisible()

    view.rerender(
      <ChatGptPairingDialog browserLocation={secureLocation} open connection={{ ...connection }} onOpenChange={vi.fn()} onCompleted={secondCompleted} />,
    )
    expect(screen.getByText('one-time-secret')).toBeVisible()
    expect(api.startPairing).toHaveBeenCalledTimes(1)
    expect(api.cancelPairing).not.toHaveBeenCalled()

    resolveStatus({ pairing_id: 'pair-1', state: 'completed', expires_at: started().expires_at, failure_code: null })
    expect(await screen.findByRole('combobox', { name: 'Choose a ChatGPT model' })).toBeVisible()
    expect(firstCompleted).not.toHaveBeenCalled()
    expect(secondCompleted).toHaveBeenCalledTimes(1)

    view.rerender(
      <ChatGptPairingDialog browserLocation={secureLocation} open connection={{ ...connection }} onOpenChange={vi.fn()} onCompleted={thirdCompleted} />,
    )
    expect(screen.getByRole('combobox', { name: 'Choose a ChatGPT model' })).toBeVisible()
    expect(api.startPairing).toHaveBeenCalledTimes(1)
    expect(api.cancelPairing).not.toHaveBeenCalled()
  })

  it.each(['completed', 'failed', 'cancelled'] as const)(
    'removes the secret from DOM and reachable React Fiber data after %s',
    async (state) => {
      const result = await settleStart()
      let resolveStatus!: (value: {
        pairing_id: string; state: typeof state; expires_at: string; failure_code: 'invalid_client' | null
      }) => void
      api.pairingStatus.mockReturnValue(new Promise((resolve) => { resolveStatus = resolve }))
      renderWithProviders(<ChatGptPairingDialog browserLocation={secureLocation} open connection={connection} onOpenChange={vi.fn()} />)
      const secretNode = await screen.findByTestId('pairing-secret')
      const fiber = fiberFor(secretNode)
      resolveStatus({
        pairing_id: 'pair-1', state, expires_at: started().expires_at,
        failure_code: state === 'failed' ? 'invalid_client' : null,
      })
      await waitFor(() => expect(screen.queryByTestId('pairing-secret')).not.toBeInTheDocument())
      expect(document.body.textContent).not.toContain('one-time-secret')
      expect(result.secretBytes.every((byte) => byte === 0)).toBe(true)
      expect(fiberRetains(fiber, 'one-time-secret')).toEqual([])
    },
  )

  it('removes the secret from DOM and reachable React Fiber data after close and unmount', async () => {
    const result = await settleStart()
    const view = renderWithProviders(<ChatGptPairingDialog browserLocation={secureLocation} open connection={connection} onOpenChange={vi.fn()} />)
    const secretNode = await screen.findByTestId('pairing-secret')
    const fiber = fiberFor(secretNode)
    await view.user.click(screen.getByRole('button', { name: 'Cancel' }))
    view.unmount()
    expect(secretNode.textContent).toBe('')
    expect(document.body.textContent).not.toContain('one-time-secret')
    expect(result.secretBytes.every((byte) => byte === 0)).toBe(true)
    expect(fiberRetains(fiber, 'one-time-secret')).toEqual([])
  })

  it('removes the secret from DOM and reachable React Fiber data when polling and cancellation fail', async () => {
    vi.useFakeTimers()
    const result = await settleStart()
    api.pairingStatus.mockRejectedValue(new Error('network unavailable'))
    api.cancelPairing.mockRejectedValue(new Error('cancellation unavailable'))
    const view = renderWithProviders(<ChatGptPairingDialog browserLocation={secureLocation} open connection={connection} onOpenChange={vi.fn()} />)
    await act(async () => { await vi.advanceTimersByTimeAsync(0) })
    const secretNode = screen.getByTestId('pairing-secret')
    const fiber = fiberFor(secretNode)
    await act(async () => { await vi.runAllTimersAsync() })
    expect(secretNode.textContent).toBe('')
    expect(document.body.textContent).not.toContain('one-time-secret')
    expect(result.secretBytes.every((byte) => byte === 0)).toBe(true)
    expect(fiberRetains(fiber, 'one-time-secret')).toEqual([])
    view.unmount()
  })

  it('keeps retrying status outages with bounded backoff until expiry and never renders raw errors', async () => {
    vi.useFakeTimers()
    await settleStart()
    api.pairingStatus.mockRejectedValue(new Error('raw network secret'))
    const view = renderWithProviders(<ChatGptPairingDialog browserLocation={secureLocation} open connection={connection} onOpenChange={vi.fn()} />)
    await act(async () => { await vi.advanceTimersByTimeAsync(30_000) })
    expect(api.pairingStatus.mock.calls.length).toBeGreaterThanOrEqual(4)
    expect(document.body.textContent).not.toContain('raw network secret')
    expect(screen.getByText('one-time-secret')).toBeVisible()
    expect(api.cancelPairing).not.toHaveBeenCalled()
    await act(async () => { await vi.advanceTimersByTimeAsync(30_000) })
    expect(screen.queryByText('one-time-secret')).not.toBeInTheDocument()
    expect(api.cancelPairing).toHaveBeenCalledTimes(1)
    view.unmount()
    await act(async () => { await vi.runOnlyPendingTimersAsync() })
    expect(vi.getTimerCount()).toBe(0)
  })

  it('serializes model PATCHes and converges on the authoritative refetch', async () => {
    await settleStart()
    api.pairingStatus.mockResolvedValue({ pairing_id: 'pair-1', state: 'completed', expires_at: started().expires_at, failure_code: null })
    api.test.mockResolvedValue({
      ok: true, detail: 'raw',
      models: [{ slug: 'gpt-a', display_name: 'GPT A' }, { slug: 'gpt-b', display_name: 'GPT B' }],
    })
    let resolveUpdate!: (value: LlmConnection) => void
    api.update.mockReturnValue(new Promise((resolve) => { resolveUpdate = resolve }))
    api.get.mockResolvedValue({ ...connection, default_model: 'gpt-a' })
    const view = renderWithProviders(<ChatGptPairingDialog browserLocation={secureLocation} open connection={connection} onOpenChange={vi.fn()} />)
    const select = await screen.findByRole('combobox', { name: 'Choose a ChatGPT model' })
    await view.user.click(select)
    await view.user.click(await screen.findByRole('option', { name: 'GPT B' }))

    expect(select).toBeDisabled()
    expect(api.update).toHaveBeenCalledTimes(1)
    expect(api.update).toHaveBeenCalledWith('connection-1', { default_model: 'gpt-b' })
    await view.user.click(select)
    expect(api.update).toHaveBeenCalledTimes(1)

    api.get.mockResolvedValue({ ...connection, default_model: 'gpt-b' })
    resolveUpdate({ ...connection, default_model: 'stale-patch-response' })
    await waitFor(() => expect(select).not.toBeDisabled())
    expect(select).toHaveTextContent('GPT B')
  })

  it('restores an authoritative model after PATCH rejection', async () => {
    await settleStart()
    api.pairingStatus.mockResolvedValue({ pairing_id: 'pair-1', state: 'completed', expires_at: started().expires_at, failure_code: null })
    api.test.mockResolvedValue({ ok: true, detail: 'raw', models: [{ slug: 'gpt-a', display_name: 'GPT A' }, { slug: 'gpt-b', display_name: 'GPT B' }] })
    api.update.mockRejectedValue(new Error('raw save error'))
    api.get.mockResolvedValue({ ...connection, default_model: 'gpt-a' })
    const onReconciled = vi.fn()
    const view = renderWithProviders(<ChatGptPairingDialog browserLocation={secureLocation} open connection={{ ...connection, default_model: 'gpt-a' }} onOpenChange={vi.fn()} onReconciled={onReconciled} />)
    const select = await screen.findByRole('combobox', { name: 'Choose a ChatGPT model' })
    await view.user.click(select)
    await view.user.click(await screen.findByRole('option', { name: 'GPT B' }))

    await waitFor(() => expect(api.get).toHaveBeenCalledWith('connection-1'))
    await waitFor(() => expect(select).not.toBeDisabled())
    expect(select).toHaveTextContent('GPT A')
    expect(onReconciled).toHaveBeenCalledWith(expect.objectContaining({ default_model: 'gpt-a' }))
    expect(toast.error).toHaveBeenCalledWith('Save failed')
  })

  it('does not let a late model save from an old opening disable or overwrite a reopened dialog', async () => {
    await settleStart()
    api.pairingStatus.mockResolvedValue({ pairing_id: 'pair-1', state: 'completed', expires_at: started().expires_at, failure_code: null })
    api.test.mockResolvedValue({ ok: true, detail: 'raw', models: [{ slug: 'gpt-a', display_name: 'GPT A' }, { slug: 'gpt-b', display_name: 'GPT B' }] })
    let rejectOld!: (reason: Error) => void
    api.update.mockReturnValueOnce(new Promise((_resolve, reject) => { rejectOld = reject }))
    const view = renderWithProviders(<ChatGptPairingDialog browserLocation={secureLocation} open connection={connection} onOpenChange={vi.fn()} />)
    const first = await screen.findByRole('combobox', { name: 'Choose a ChatGPT model' })
    await view.user.click(first)
    await view.user.click(await screen.findByRole('option', { name: 'GPT A' }))
    expect(first).toBeDisabled()

    view.rerender(<ChatGptPairingDialog browserLocation={secureLocation} open={false} connection={connection} onOpenChange={vi.fn()} />)
    view.rerender(<ChatGptPairingDialog browserLocation={secureLocation} open connection={connection} onOpenChange={vi.fn()} />)
    await waitFor(() => expect(api.startPairing).toHaveBeenCalledTimes(2))
    await waitFor(() => expect(screen.getByRole('combobox', { name: 'Choose a ChatGPT model' })).not.toBeDisabled())
    rejectOld(new Error('late failure'))
    await act(async () => { await Promise.resolve() })
    expect(screen.getByRole('combobox', { name: 'Choose a ChatGPT model' })).not.toBeDisabled()
  })

  it('reconciles a closed model PATCH after a signal-ignoring root fetch without corrupting the reopened dialog', async () => {
    await settleStart()
    api.pairingStatus.mockResolvedValue({ pairing_id: 'pair-1', state: 'completed', expires_at: started().expires_at, failure_code: null })
    api.test.mockResolvedValue({
      ok: true, detail: 'raw',
      models: [{ slug: 'gpt-a', display_name: 'GPT A' }, { slug: 'gpt-b', display_name: 'GPT B' }],
    })
    let resolveUpdate!: (value: LlmConnection) => void
    api.update.mockReturnValue(new Promise((resolve) => { resolveUpdate = resolve }))
    const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false, gcTime: Infinity } } })
    queryClient.setQueryData(AGENT_CONNECTIONS_KEY, [connection])
    const onReconciled = vi.fn()
    const view = renderWithProviders(
      <ChatGptPairingDialog browserLocation={secureLocation} open connection={connection} onOpenChange={vi.fn()} onReconciled={onReconciled} />,
      { queryClient },
    )
    const first = await screen.findByRole('combobox', { name: 'Choose a ChatGPT model' })
    await view.user.click(first)
    await view.user.click(await screen.findByRole('option', { name: 'GPT B' }))

    let resolveStale!: (value: LlmConnection[]) => void
    const staleFetch = queryClient.fetchQuery({
      queryKey: AGENT_CONNECTIONS_KEY,
      queryFn: () => new Promise<LlmConnection[]>((resolve) => { resolveStale = resolve }),
    }).catch(() => undefined)
    await waitFor(() => expect(queryClient.getQueryState(AGENT_CONNECTIONS_KEY)?.fetchStatus).toBe('fetching'))

    view.rerender(<ChatGptPairingDialog browserLocation={secureLocation} open={false} connection={connection} onOpenChange={vi.fn()} onReconciled={onReconciled} />)
    const reopened = { ...connection, default_model: 'gpt-a' }
    api.get.mockResolvedValue(reopened)
    view.rerender(<ChatGptPairingDialog browserLocation={secureLocation} open connection={reopened} onOpenChange={vi.fn()} onReconciled={onReconciled} />)
    await waitFor(() => expect(api.startPairing).toHaveBeenCalledTimes(2))
    const freshSelect = await screen.findByRole('combobox', { name: 'Choose a ChatGPT model' })
    expect(freshSelect).toHaveTextContent('GPT A')

    const authoritative = {
      ...connection, default_model: 'gpt-b', credential_status: 'active' as const,
      account_email: 'paired@example.com', scopes: ['openid'], access_token_expires_at: '2030-01-01T00:00:00Z',
    }
    const getsBeforeOldPatch = api.get.mock.calls.length
    api.get.mockResolvedValue(authoritative)
    resolveUpdate({ ...connection, default_model: 'stale-patch-response' })

    await waitFor(() => expect(api.get).toHaveBeenCalledTimes(getsBeforeOldPatch + 1))
    await waitFor(() => expect(queryClient.getQueryData([...AGENT_CONNECTIONS_KEY, connection.id])).toEqual(authoritative))
    expect(queryClient.getQueryData<LlmConnection[]>(AGENT_CONNECTIONS_KEY)).toEqual([authoritative])
    expect(onReconciled).toHaveBeenLastCalledWith(authoritative)
    expect(freshSelect).toHaveTextContent('GPT A')
    expect(freshSelect).not.toBeDisabled()

    resolveStale([{ ...connection, default_model: 'stale-root' }])
    await staleFetch
    expect(queryClient.getQueryData<LlmConnection[]>(AGENT_CONNECTIONS_KEY)).toEqual([authoritative])
  })

  it.each(['success', 'failure'] as const)(
    'reconciles authoritative credential metadata after a closed post-pairing test %s without corrupting the reopened dialog',
    async (outcome) => {
      api.startPairing
        .mockResolvedValueOnce(started())
        .mockResolvedValueOnce(started({ pairing_id: 'fresh', secret: 'fresh-secret' }))
      api.pairingStatus
        .mockResolvedValueOnce({ pairing_id: 'pair-1', state: 'completed', expires_at: started().expires_at, failure_code: null })
        .mockReturnValue(new Promise(() => undefined))
      let resolveTest!: (value: { ok: boolean; detail: string; models: string[] }) => void
      let rejectTest!: (reason: Error) => void
      api.test.mockReturnValueOnce(new Promise((resolve, reject) => { resolveTest = resolve; rejectTest = reject }))
      const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false, gcTime: Infinity } } })
      queryClient.setQueryData(AGENT_CONNECTIONS_KEY, [connection])
      const onReconciled = vi.fn()
      const view = renderWithProviders(
        <ChatGptPairingDialog browserLocation={secureLocation} open connection={connection} onOpenChange={vi.fn()} onReconciled={onReconciled} />,
        { queryClient },
      )
      await waitFor(() => expect(api.test).toHaveBeenCalledTimes(1))

      let resolveStale!: (value: LlmConnection[]) => void
      const staleFetch = queryClient.fetchQuery({
        queryKey: AGENT_CONNECTIONS_KEY,
        queryFn: () => new Promise<LlmConnection[]>((resolve) => { resolveStale = resolve }),
      }).catch(() => undefined)
      await waitFor(() => expect(queryClient.getQueryState(AGENT_CONNECTIONS_KEY)?.fetchStatus).toBe('fetching'))

      view.rerender(<ChatGptPairingDialog browserLocation={secureLocation} open={false} connection={connection} onOpenChange={vi.fn()} onReconciled={onReconciled} />)
      view.rerender(<ChatGptPairingDialog browserLocation={secureLocation} open connection={connection} onOpenChange={vi.fn()} onReconciled={onReconciled} />)
      expect(await screen.findByText('fresh-secret')).toBeVisible()

      const authoritative = {
        ...connection, default_model: 'gpt-authoritative', credential_status: 'active' as const,
        credential_error_code: null, account_email: 'paired@example.com', scopes: ['openid', 'profile'],
        access_token_expires_at: '2030-01-01T00:00:00Z', needs_reauthorization: false,
      }
      api.get.mockResolvedValue(authoritative)
      if (outcome === 'success') resolveTest({ ok: true, detail: 'raw success detail', models: ['gpt-old'] })
      else rejectTest(new Error('raw test failure secret'))

      await waitFor(() => expect(queryClient.getQueryData([...AGENT_CONNECTIONS_KEY, connection.id])).toEqual(authoritative))
      expect(queryClient.getQueryData<LlmConnection[]>(AGENT_CONNECTIONS_KEY)).toEqual([authoritative])
      expect(onReconciled).toHaveBeenLastCalledWith(authoritative)
      expect(screen.getByText('fresh-secret')).toBeVisible()
      expect(screen.queryByRole('combobox', { name: 'Choose a ChatGPT model' })).not.toBeInTheDocument()
      expect(document.body.textContent).not.toContain('raw test failure secret')
      expect(document.body.textContent).not.toContain('raw success detail')

      resolveStale([{ ...connection, credential_status: 'not_connected' }])
      await staleFetch
      expect(queryClient.getQueryData<LlmConnection[]>(AGENT_CONNECTIONS_KEY)).toEqual([authoritative])
    },
  )

  it.each([
    ['http 502', { response: { status: 502 } }],
    ['http 503', { response: { status: 503 } }],
    ['network', new Error('network detail')],
  ])('keeps the dialog open with fixed reconciliation copy after ambiguous cancel: %s', async (_name, failure) => {
    await settleStart()
    api.cancelPairing.mockRejectedValueOnce(failure)
    const onOpenChange = vi.fn()
    const onReconciled = vi.fn()
    const view = renderWithProviders(<ChatGptPairingDialog browserLocation={secureLocation} open connection={connection} onOpenChange={onOpenChange} onReconciled={onReconciled} />)
    await screen.findByText('one-time-secret')
    await view.user.click(screen.getByRole('button', { name: 'Cancel' }))

    expect(await screen.findByRole('alert')).toHaveTextContent('Could not confirm cancellation. Check the connection and try again.')
    expect(onOpenChange).not.toHaveBeenCalled()
    expect(api.get).toHaveBeenCalledWith('connection-1')
    expect(onReconciled).toHaveBeenCalledWith(connection)
    api.get.mockResolvedValue({ ...connection, credential_status: 'active' })
    await view.user.click(screen.getByRole('button', { name: 'Cancel' }))
    await waitFor(() => expect(onOpenChange).toHaveBeenCalledWith(false))
    expect(api.cancelPairing).toHaveBeenCalledTimes(1)
  })

  it('treats cancellation 404 as a completion race and reconciles before closing', async () => {
    await settleStart()
    api.cancelPairing.mockRejectedValue({ response: { status: 404 } })
    api.get.mockResolvedValue({ ...connection, credential_status: 'active' })
    const onOpenChange = vi.fn()
    const onReconciled = vi.fn()
    const view = renderWithProviders(<ChatGptPairingDialog browserLocation={secureLocation} open connection={connection} onOpenChange={onOpenChange} onReconciled={onReconciled} />)
    await screen.findByText('one-time-secret')
    await view.user.click(screen.getByRole('button', { name: 'Cancel' }))

    await waitFor(() => expect(onOpenChange).toHaveBeenCalledWith(false))
    expect(onReconciled).toHaveBeenCalledWith(expect.objectContaining({ credential_status: 'active' }))
  })

  it('best-effort reconciles on unmount without duplicate DELETEs', async () => {
    await settleStart()
    const onReconciled = vi.fn()
    const view = renderWithProviders(<ChatGptPairingDialog browserLocation={secureLocation} open connection={connection} onOpenChange={vi.fn()} onReconciled={onReconciled} />)
    await screen.findByText('one-time-secret')
    view.unmount()
    await waitFor(() => expect(api.cancelPairing).toHaveBeenCalledTimes(1))
    await waitFor(() => expect(api.get).toHaveBeenCalledWith('connection-1'))
    expect(onReconciled).toHaveBeenCalledWith(connection)
  })
})
