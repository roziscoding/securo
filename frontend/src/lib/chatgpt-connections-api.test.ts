import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import http, { agents, WORKSPACE_STORAGE_KEY } from './api'

const signal = new AbortController().signal
const encoder = new TextEncoder()

function fetchResponse(
  payload: string,
  status = 201,
  contentType: string | null = 'application/json',
  parts?: Uint8Array[],
) {
  const source = encoder.encode(payload)
  const chunks = parts ?? [source]
  let delivered = 0
  const reader = {
    read: vi.fn(async () => {
      if (delivered >= chunks.length) return { done: true as const, value: undefined }
      return { done: false as const, value: chunks[delivered++] }
    }),
    cancel: vi.fn(async () => undefined),
    releaseLock: vi.fn(),
  }
  return {
    source,
    response: {
      ok: status >= 200 && status < 300,
      status,
      redirected: false,
      headers: new Headers(contentType ? { 'Content-Type': contentType } : {}),
      body: { getReader: () => reader },
    } as unknown as Response,
    reader,
  }
}

beforeEach(() => {
  const values = new Map<string, string>()
  vi.stubGlobal('localStorage', {
    getItem: (key: string) => values.get(key) ?? null,
    setItem: (key: string, value: string) => values.set(key, value),
    removeItem: (key: string) => values.delete(key),
    clear: () => values.clear(),
  })
  localStorage.setItem('token', 'auth-token')
  localStorage.setItem(WORKSPACE_STORAGE_KEY, 'workspace-1')
})
afterEach(() => {
  vi.restoreAllMocks()
  vi.unstubAllGlobals()
})

describe('ChatGPT connection API contracts', () => {
  it('passes the query cancellation signal to the connection list request', async () => {
    const get = vi.spyOn(http, 'get').mockResolvedValue({ data: [] })

    await agents.connections.list(signal)

    expect(get).toHaveBeenCalledWith('/agents/connections', { signal })
  })

  it('uses an isolated bounded fetch boundary and scrubs source bytes after returning the secret', async () => {
    const sentinel = 'one-time-secret-boundary-sentinel'
    const wire = fetchResponse(JSON.stringify({
      pairing_id: 'pair-1', secret: sentinel, expires_at: '2030-01-01T00:00:00Z',
    }))
    const fetchMock = vi.spyOn(globalThis, 'fetch').mockResolvedValue(wire.response)
    const axiosPost = vi.spyOn(http, 'post')

    const promise = agents.connections.startPairing('connection-1', signal)
    const started = await promise

    expect(started.pairing_id).toBe('pair-1')
    expect(new TextDecoder().decode(started.secretBytes)).toBe(sentinel)
    expect(wire.source.every((byte) => byte === 0)).toBe(true)
    expect(JSON.stringify(started)).not.toContain(sentinel)
    expect(JSON.stringify(promise)).not.toContain(sentinel)
    expect(JSON.stringify(fetchMock.mock.results)).not.toContain(sentinel)
    expect(axiosPost).not.toHaveBeenCalled()
    expect(fetchMock).toHaveBeenCalledWith('/api/agents/connections/connection-1/chatgpt/pairings', {
      method: 'POST',
      headers: {
        Accept: 'application/json',
        Authorization: 'Bearer auth-token',
        'X-Workspace-Id': 'workspace-1',
      },
      credentials: 'same-origin',
      cache: 'no-store',
      redirect: 'error',
      signal,
    })
  })

  it.each([
    'application/json',
    'Application/JSON',
    'APPLICATION/JSON; Charset=UTF-8',
    "application/json; !#$%&'*+-.^_`|~09AZaz=!#$%&'*+-.^_`|~09AZaz",
    'application/json \t;\tcharset=UTF-8 ; boundary=token',
    'application/json; empty=""',
    "application/json; note=\"\t !#$%&'()*+,-./:;<=>?@[]^_`{|}~\"",
    'application/json; note="escaped \\"quote\\" and \\\\ slash"',
    'application/json; pairs="\\' + '\t' + '\\ \\!\\' + String.fromCharCode(0x80) + '"',
    `application/json; note="latin-1 ${String.fromCharCode(0x80, 0xff)}"`,
  ])(
    'accepts exact 201 with JSON media type %s',
    async (contentType) => {
      const wire = fetchResponse(JSON.stringify({
        pairing_id: 'pair-1', secret: 'secret', expires_at: '2030-01-01T00:00:00Z',
      }), 201, contentType)
      vi.spyOn(globalThis, 'fetch').mockResolvedValue(wire.response)

      await expect(agents.connections.startPairing('connection-1', signal)).resolves.toMatchObject({ pairing_id: 'pair-1' })
    },
  )

  it.each([
    ' application/json',
    'application/json ',
    'application/json\n;charset=UTF-8',
    'application/json\r;charset=UTF-8',
    'application/json\v;charset=UTF-8',
    'application/json\f;charset=UTF-8',
    'application/json\u00a0;charset=UTF-8',
    'application/json; note="control \u0001"',
    'application/json; note="delete \u007f"',
    'application/json; note="unicode \u0100"',
    'application/json; note="escaped newline \\' + '\n' + '"',
    'application/json; note="unterminated',
    'application/json; note="dangling escape \\',
    'application/json; =token',
    'application/json; name=',
    'application/json; name =token',
    'application/json; name= token',
    'application/json; na(me=token',
    'application/json; name=to/ken',
    'application /json; name=token',
    'application/json, text/plain',
    'application/json; charset=UTF-8 junk',
    'application/json;',
    'application/json;; charset=UTF-8',
  ])('rejects malformed JSON media type before reading: %s', async (contentType) => {
    const wire = fetchResponse(JSON.stringify({
      pairing_id: 'pair-1', secret: 'secret', expires_at: '2030-01-01T00:00:00Z',
    }))
    Object.defineProperty(wire.response, 'headers', {
      value: { get: () => contentType },
    })
    vi.spyOn(globalThis, 'fetch').mockResolvedValue(wire.response)

    await expect(agents.connections.startPairing('connection-1', signal)).rejects.toThrow('pairing_start_invalid')
    expect(wire.reader.read).not.toHaveBeenCalled()
    expect(wire.reader.cancel).toHaveBeenCalledTimes(1)
  })

  it.each([
    [200, 'application/json'], [202, 'application/json'], [204, 'application/json'],
    [201, 'text/plain'], [201, 'text/html'], [201, null],
  ])('rejects status/content-type mismatch before reading: %s %s', async (status, contentType) => {
    const wire = fetchResponse(JSON.stringify({
      pairing_id: 'pair-1', secret: 'secret', expires_at: '2030-01-01T00:00:00Z',
    }), status, contentType)
    vi.spyOn(globalThis, 'fetch').mockResolvedValue(wire.response)

    await expect(agents.connections.startPairing('connection-1', signal)).rejects.toThrow('pairing_start_invalid')
    expect(wire.reader.read).not.toHaveBeenCalled()
    expect(wire.reader.cancel).toHaveBeenCalledTimes(1)
  })

  it('preserves UTF-8 decoding and multi-chunk bounds for an exact success response', async () => {
    const payload = JSON.stringify({ pairing_id: 'pair-1', secret: 'sëcret', expires_at: '2030-01-01T00:00:00Z' })
    const bytes = encoder.encode(payload)
    const parts = [bytes.slice(0, 11), bytes.slice(11, 29), bytes.slice(29)]
    const wire = fetchResponse(payload, 201, 'application/json; charset=utf-8', parts)
    vi.spyOn(globalThis, 'fetch').mockResolvedValue(wire.response)

    const result = await agents.connections.startPairing('connection-1', signal)
    expect(new TextDecoder().decode(result.secretBytes)).toBe('sëcret')
    expect(parts.every((part) => part.every((byte) => byte === 0))).toBe(true)
  })

  it('propagates abort during fetch without parsing a response', async () => {
    const controller = new AbortController()
    vi.spyOn(globalThis, 'fetch').mockImplementation(async (_url, init) => {
      controller.abort()
      throw new DOMException(init?.signal?.aborted ? 'Aborted' : 'unexpected', 'AbortError')
    })

    await expect(agents.connections.startPairing('connection-1', controller.signal)).rejects.toMatchObject({ name: 'AbortError' })
  })

  it('rejects and cancels on abort during body read', async () => {
    const wire = fetchResponse('{}')
    wire.reader.read.mockRejectedValueOnce(new DOMException('Aborted', 'AbortError'))
    vi.spyOn(globalThis, 'fetch').mockResolvedValue(wire.response)

    await expect(agents.connections.startPairing('connection-1', signal)).rejects.toMatchObject({ name: 'AbortError' })
    expect(wire.reader.cancel).toHaveBeenCalledTimes(1)
  })

  it.each([
    ['wrong type', JSON.stringify({ pairing_id: 4, secret: 'secret', expires_at: '2030-01-01T00:00:00Z' })],
    ['extra field', JSON.stringify({ pairing_id: 'pair', secret: 'secret', expires_at: '2030-01-01T00:00:00Z', extra: true })],
    ['oversized secret', JSON.stringify({ pairing_id: 'pair', secret: 'x'.repeat(5000), expires_at: '2030-01-01T00:00:00Z' })],
  ])('rejects and scrubs malformed start responses: %s', async (_name, payload) => {
    const wire = fetchResponse(payload)
    vi.spyOn(globalThis, 'fetch').mockResolvedValue(wire.response)

    await expect(agents.connections.startPairing('connection-1', signal)).rejects.toThrow('pairing_start_invalid')
    expect(wire.source.every((byte) => byte === 0)).toBe(true)
  })

  it('bounds non-success bodies and exposes only a safe error code', async () => {
    const wire = fetchResponse(JSON.stringify({ code: 'credential_busy', detail: 'provider-secret-detail' }), 503)
    vi.spyOn(globalThis, 'fetch').mockResolvedValue(wire.response)

    await expect(agents.connections.startPairing('connection-1', signal)).rejects.toThrow('pairing_start_credential_busy')
    expect(wire.source.every((byte) => byte === 0)).toBe(true)
  })

  it.each([
    ['unknown code', '{"code":"future_server_code"}'],
    ['malformed JSON', '{'],
    ['non-object JSON', '[]'],
  ])('maps a JSON failure without a safe code to unavailable: %s', async (_name, payload) => {
    const wire = fetchResponse(payload, 503)
    vi.spyOn(globalThis, 'fetch').mockResolvedValue(wire.response)

    await expect(agents.connections.startPairing('connection-1', signal)).rejects.toThrow('pairing_start_unavailable')
  })

  it.each(['text/plain', 'text/html', null])(
    'does not parse non-JSON failure media type %s',
    async (contentType) => {
      const wire = fetchResponse('{"code":"credential_busy"}', 503, contentType)
      vi.spyOn(globalThis, 'fetch').mockResolvedValue(wire.response)

      await expect(agents.connections.startPairing('connection-1', signal)).rejects.toThrow('pairing_start_unavailable')
      expect(wire.reader.read).not.toHaveBeenCalled()
      expect(wire.reader.cancel).toHaveBeenCalledTimes(1)
    },
  )

  it('polls, cancels, and revokes using the exact methods and paths with no secret payload', async () => {
    const get = vi.spyOn(http, 'get').mockResolvedValue({ data: { pairing_id: 'pair-1', state: 'pending', expires_at: '2030-01-01T00:00:00Z' } })
    const remove = vi.spyOn(http, 'delete').mockResolvedValue({})
    const post = vi.spyOn(http, 'post').mockResolvedValue({})

    await agents.connections.pairingStatus('connection-1', 'pair-1', signal)
    await agents.connections.cancelPairing('connection-1', 'pair-1')
    await agents.connections.revoke('connection-1')

    expect(get).toHaveBeenCalledWith('/agents/connections/connection-1/chatgpt/pairings/pair-1', { signal })
    expect(remove).toHaveBeenCalledWith('/agents/connections/connection-1/chatgpt/pairings/pair-1')
    expect(post).toHaveBeenCalledWith('/agents/connections/connection-1/revoke')
    expect(JSON.stringify([get.mock.calls, remove.mock.calls, post.mock.calls])).not.toContain('secret')
  })
})