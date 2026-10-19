import { screen, waitFor } from '@testing-library/react'
import { QueryClient } from '@tanstack/react-query'
import { beforeEach, expect, it, vi } from 'vitest'
import { renderWithProviders } from '@/test/utils'
import { ConnectionFormDialog } from './connection-form-dialog'
import type { LlmConnection } from '@/lib/api'
import { AGENT_CONNECTIONS_KEY } from '@/lib/agent-connections-cache'

const api = vi.hoisted(() => ({ create: vi.fn(), update: vi.fn() }))
vi.mock('@/lib/api', async (original) => {
  const actual = await original<typeof import('@/lib/api')>()
  return { ...actual, agents: { connections: api } }
})

beforeEach(() => {
  vi.clearAllMocks()
  api.create.mockImplementation(async (payload) => ({ id: 'new', ...payload }))
})

it('creates ChatGPT with only safe fields and continues to pairing', async () => {
  const onSaved = vi.fn()
  const view = renderWithProviders(<ConnectionFormDialog open onOpenChange={vi.fn()} onSaved={onSaved} />)
  await view.user.type(screen.getByLabelText('Name'), 'My ChatGPT')
  await view.user.click(screen.getByRole('combobox', { name: 'Provider type' }))
  await view.user.click(await screen.findByRole('option', { name: 'Sign in with ChatGPT' }))

  expect(screen.queryByLabelText(/Endpoint URL/)).not.toBeInTheDocument()
  expect(screen.queryByLabelText(/API key/)).not.toBeInTheDocument()
  await view.user.click(screen.getByRole('button', { name: 'Create and continue' }))

  await waitFor(() => expect(api.create).toHaveBeenCalledWith({
    name: 'My ChatGPT', kind: 'openai_chatgpt', is_default: false,
  }))
  const payload = api.create.mock.calls[0][0]
  expect(payload).not.toHaveProperty('api_key')
  expect(payload).not.toHaveProperty('base_url')
  expect(payload).not.toHaveProperty('extra')
  expect(onSaved).toHaveBeenCalled()
})

it('guards create and default changes while save is pending', async () => {
  let resolveCreate!: (value: LlmConnection) => void
  api.create.mockReturnValue(new Promise((resolve) => { resolveCreate = resolve }))
  const view = renderWithProviders(<ConnectionFormDialog open onOpenChange={vi.fn()} />)
  await view.user.type(screen.getByLabelText('Name'), 'Pending provider')
  const create = screen.getByRole('button', { name: 'Create' })

  await view.user.click(create)

  expect(api.create).toHaveBeenCalledTimes(1)
  expect(create).toBeDisabled()
  expect(screen.getByRole('switch', { name: /default/i })).toBeDisabled()
  resolveCreate({ id: 'new', name: 'Pending provider' } as LlmConnection)
  await waitFor(() => expect(create).not.toBeDisabled())
})

it('preserves API-key provider payload behavior', async () => {
  const view = renderWithProviders(<ConnectionFormDialog open onOpenChange={vi.fn()} />)
  await view.user.type(screen.getByLabelText('Name'), 'OpenAI')
  await view.user.click(screen.getByRole('combobox', { name: 'Provider type' }))
  await view.user.click(await screen.findByRole('option', { name: 'OpenAI' }))
  await view.user.type(screen.getByLabelText(/API key/), 'sk-example')
  await view.user.click(screen.getByRole('button', { name: 'Create' }))
  await waitFor(() => expect(api.create).toHaveBeenCalled())
  expect(api.create.mock.calls[0][0]).toMatchObject({ kind: 'openai', api_key: 'sk-example', default_model: 'gpt-4o-mini' })
})

it('edits ChatGPT without forbidden generic-provider fields', async () => {
  const connection: LlmConnection = {
    id: 'c1', name: 'ChatGPT', kind: 'openai_chatgpt', base_url: null, default_model: 'gpt-old', extra: {},
    is_default: false, has_api_key: false, auth_method: 'chatgpt_oauth', credential_status: 'active',
    credential_error_code: null, account_email: null, scopes: [], access_token_expires_at: null,
    needs_reauthorization: false, created_at: '', updated_at: '',
  }
  api.update.mockResolvedValue({ ...connection, name: 'Renamed' })
  const view = renderWithProviders(<ConnectionFormDialog open connection={connection} onOpenChange={vi.fn()} />)
  await view.user.clear(screen.getByLabelText('Name'))
  await view.user.type(screen.getByLabelText('Name'), 'Renamed')
  await view.user.click(screen.getByRole('button', { name: 'Save' }))
  await waitFor(() => expect(api.update).toHaveBeenCalledWith('c1', {
    name: 'Renamed', default_model: 'gpt-old', is_default: false,
  }))
  const payload = api.update.mock.calls[0][1]
  expect(payload).not.toHaveProperty('api_key')
  expect(payload).not.toHaveProperty('base_url')
  expect(payload).not.toHaveProperty('extra')
  expect(payload).not.toHaveProperty('kind')
})

it('preserves OpenAI-compatible endpoint and optional-key payloads', async () => {
  const view = renderWithProviders(<ConnectionFormDialog open onOpenChange={vi.fn()} />)
  await view.user.type(screen.getByLabelText('Name'), 'Local provider')
  await view.user.click(screen.getByRole('combobox', { name: 'Provider type' }))
  await view.user.click(await screen.findByRole('option', { name: /OpenAI-compatible/ }))
  await view.user.type(screen.getByLabelText(/Endpoint URL/), 'https://models.example/v1')
  await view.user.type(screen.getByLabelText('Default model'), 'custom-model')
  await view.user.click(screen.getByRole('button', { name: 'Create' }))
  await waitFor(() => expect(api.create).toHaveBeenCalled())
  expect(api.create.mock.calls[0][0]).toMatchObject({
    kind: 'openai_compatible', base_url: 'https://models.example/v1', api_key: null, default_model: 'custom-model',
  })
})

it('keeps a created connection after a delayed root list fetch settles', async () => {
  const old = { id: 'old', name: 'Old', kind: 'openai', is_default: false } as LlmConnection
  const saved = { ...old, id: 'created', name: 'Created' }
  api.create.mockResolvedValue(saved)
  const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false, gcTime: Infinity } } })
  queryClient.setQueryData(AGENT_CONNECTIONS_KEY, [old])
  let resolveStale!: (value: LlmConnection[]) => void
  let signal: AbortSignal | undefined
  const staleFetch = queryClient.fetchQuery({
    queryKey: AGENT_CONNECTIONS_KEY,
    queryFn: ({ signal: currentSignal }) => {
      signal = currentSignal
      return new Promise<LlmConnection[]>((resolve) => { resolveStale = resolve })
    },
  }).catch(() => undefined)
  const view = renderWithProviders(<ConnectionFormDialog open onOpenChange={vi.fn()} />, { queryClient })
  await view.user.type(screen.getByLabelText('Name'), 'Created')
  await view.user.click(screen.getByRole('button', { name: 'Create' }))

  await waitFor(() => expect(api.create).toHaveBeenCalled())
  expect(signal?.aborted).toBe(true)
  resolveStale([old])
  await staleFetch

  await waitFor(() => expect(queryClient.getQueryData<LlmConnection[]>(AGENT_CONNECTIONS_KEY)?.map(({ id }) => id)).toEqual(['old', 'created']))
})

it('keeps an updated default authoritative after a delayed root list fetch settles', async () => {
  const oldDefault = {
    id: 'old-default', name: 'Old default', kind: 'openai', base_url: null, default_model: null, extra: {},
    is_default: true, has_api_key: false, auth_method: 'api_key', credential_status: null,
    credential_error_code: null, account_email: null, scopes: [], access_token_expires_at: null,
    needs_reauthorization: false, created_at: '', updated_at: '',
  } as LlmConnection
  const edited = { ...oldDefault, id: 'edited', name: 'Edited', is_default: false }
  const saved = { ...edited, is_default: true }
  api.update.mockResolvedValue(saved)
  const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false, gcTime: Infinity } } })
  queryClient.setQueryData(AGENT_CONNECTIONS_KEY, [oldDefault, edited])
  let resolveStale!: (value: LlmConnection[]) => void
  const staleFetch = queryClient.fetchQuery({
    queryKey: AGENT_CONNECTIONS_KEY,
    queryFn: () => new Promise<LlmConnection[]>((resolve) => { resolveStale = resolve }),
  }).catch(() => undefined)
  const view = renderWithProviders(<ConnectionFormDialog open connection={edited} onOpenChange={vi.fn()} />, { queryClient })
  await view.user.click(screen.getByRole('switch', { name: /default/i }))
  await view.user.click(screen.getByRole('button', { name: 'Save' }))

  await waitFor(() => expect(api.update).toHaveBeenCalled())
  resolveStale([oldDefault, edited])
  await staleFetch

  await waitFor(() => expect(queryClient.getQueryData<LlmConnection[]>(AGENT_CONNECTIONS_KEY)?.map(({ id, is_default }) => [id, is_default])).toEqual([
    ['old-default', false], ['edited', true],
  ]))
})
