import { useCallback, useEffect, useRef, useState } from 'react'
import { Copy, Loader2 } from 'lucide-react'
import { useQueryClient } from '@tanstack/react-query'
import { useTranslation } from 'react-i18next'
import { toast } from 'sonner'
import { Button } from '@/components/ui/button'
import { Dialog, DialogContent, DialogDescription, DialogFooter, DialogHeader, DialogTitle } from '@/components/ui/dialog'
import { Label } from '@/components/ui/label'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { agents } from '@/lib/api'
import type { LlmConnection, OAuthModel, PairingFailureCode, PairingState } from '@/lib/api'
import { cancelAgentConnectionsList, upsertAgentConnection } from '@/lib/agent-connections-cache'
import { chatGptHelperCommand, normalizeOAuthModels } from './chatgpt-pairing-utils'

interface Props {
  open: boolean
  connection: LlmConnection | null
  onOpenChange: (open: boolean) => void
  onCompleted?: () => void
  onReconciled?: (connection: LlmConnection) => void | Promise<void>
  /** Injectable only for deterministic browser-origin tests. */
  browserLocation?: Pick<Location, 'protocol' | 'origin'>
}

type Phase = 'idle' | 'starting' | PairingState | 'expired' | 'network' | 'reconciling'
type PublicPairing = { pairingId: string; expiresAt: string }
const TERMINAL = new Set<Phase>(['completed', 'failed', 'cancelled', 'expired'])
const POLL_DELAY_MS = 2000
const MAX_POLL_DELAY_MS = 15_000
const SAFE_FAILURES: Record<PairingFailureCode, string> = {
  transient: 'agents.connections.chatgpt.errors.transient', revoked: 'agents.connections.chatgpt.errors.revoked',
  scope_loss: 'agents.connections.chatgpt.errors.scope_loss', corrupt_token: 'agents.connections.chatgpt.errors.invalid',
  reauth_required: 'agents.connections.chatgpt.errors.reauth', too_early: 'agents.connections.chatgpt.errors.transient',
  race_lost: 'agents.connections.chatgpt.errors.transient', redirect_rejected: 'agents.connections.chatgpt.errors.invalid',
  malformed_response: 'agents.connections.chatgpt.errors.invalid', inactive: 'agents.connections.chatgpt.errors.reauth',
  invalid_client: 'agents.connections.chatgpt.errors.invalid', expired: 'agents.connections.chatgpt.errors.expired',
  refresh_outcome_unknown: 'agents.connections.chatgpt.errors.unknown', revocation_pending: 'agents.connections.chatgpt.errors.unknown',
  resource_cleanup_pending: 'agents.connections.chatgpt.errors.unknown', credential_busy: 'agents.connections.chatgpt.errors.busy',
  start_outcome_unknown: 'agents.connections.chatgpt.errors.unknown',
}

function errorStatus(error: unknown): number | undefined {
  return (error as { response?: { status?: number } })?.response?.status
}

export function ChatGptPairingDialog({ open, connection, onOpenChange, onCompleted, onReconciled, browserLocation }: Props) {
  const { t } = useTranslation()
  const queryClient = useQueryClient()
  const [phase, setPhase] = useState<Phase>('idle')
  const [pairing, setPairing] = useState<PublicPairing | null>(null)
  const [hasSecret, setHasSecret] = useState(false)
  const [failure, setFailure] = useState<PairingFailureCode | null>(null)
  const [reconcileError, setReconcileError] = useState(false)
  const [models, setModels] = useState<OAuthModel[]>([])
  const [selectedModel, setSelectedModel] = useState('')
  const [modelPending, setModelPending] = useState(false)
  const titleRef = useRef<HTMLHeadingElement>(null)
  const secretNodeRef = useRef<HTMLElement | null>(null)
  const secretBytesRef = useRef<Uint8Array | null>(null)
  const onCompletedRef = useRef(onCompleted)
  const onReconciledRef = useRef(onReconciled)
  const modelPendingRef = useRef(false)
  const modelOperationRef = useRef(0)
  const generation = useRef(0)
  const pairingRef = useRef<PublicPairing | null>(null)
  const connectionIdRef = useRef<string | null>(null)
  const abortRef = useRef<AbortController | null>(null)
  const pollTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null)
  const expiryTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null)
  const startTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null)
  const cancelInFlightRef = useRef<{ pairingId: string; promise: Promise<boolean> } | null>(null)
  const cancellationAttemptedRef = useRef(new Set<string>())
  const cancellationConfirmedRef = useRef(new Set<string>())
  onCompletedRef.current = onCompleted
  onReconciledRef.current = onReconciled

  const stopActivity = useCallback(() => {
    abortRef.current?.abort()
    abortRef.current = null
    if (pollTimerRef.current) clearTimeout(pollTimerRef.current)
    if (expiryTimerRef.current) clearTimeout(expiryTimerRef.current)
    if (startTimerRef.current) clearTimeout(startTimerRef.current)
    pollTimerRef.current = null
    expiryTimerRef.current = null
    startTimerRef.current = null
  }, [])

  const clearPairingRefs = useCallback(() => {
    pairingRef.current = null
    connectionIdRef.current = null
  }, [])

  const scrubVisible = useCallback((updateState = true) => {
    if (secretNodeRef.current) secretNodeRef.current.textContent = ''
    secretBytesRef.current?.fill(0)
    secretBytesRef.current = null
    if (updateState) {
      setHasSecret(false)
      setPairing(null)
    }
  }, [])

  const attachSecretNode = useCallback((node: HTMLElement | null) => {
    secretNodeRef.current = node
    if (!node) return
    const bytes = secretBytesRef.current
    node.textContent = bytes ? new TextDecoder().decode(bytes) : ''
  }, [])

  const reconcileConnection = useCallback(async (connectionId: string): Promise<LlmConnection | null> => {
    try {
      await cancelAgentConnectionsList(queryClient)
      const current = await agents.connections.get(connectionId)
      await cancelAgentConnectionsList(queryClient)
      await upsertAgentConnection(queryClient, current)
      await onReconciledRef.current?.(current)
      return current
    } catch {
      return null
    }
  }, [queryClient])

  // A live pairing has a server TTL backstop. Navigation/unmount performs this
  // same DELETE+GET sequence best-effort, without updating this component.
  const cancelAndReconcile = useCallback((current: PublicPairing, connectionId: string): Promise<boolean> => {
    const existing = cancelInFlightRef.current
    if (existing?.pairingId === current.pairingId) return existing.promise

    const promise = (async () => {
      if (!cancellationAttemptedRef.current.has(current.pairingId)) {
        // Consume the single cancellation attempt before awaiting I/O. An
        // ambiguous network outcome must never cause a duplicate DELETE.
        cancellationAttemptedRef.current.add(current.pairingId)
        try {
          await cancelAgentConnectionsList(queryClient)
          await agents.connections.cancelPairing(connectionId, current.pairingId)
          cancellationConfirmedRef.current.add(current.pairingId)
        } catch (error) {
          if (errorStatus(error) === 404) cancellationConfirmedRef.current.add(current.pairingId)
        }
      }
      const fetched = await reconcileConnection(connectionId)
      if (!fetched) return false
      if (cancellationConfirmedRef.current.has(current.pairingId)) return true
      if (fetched.credential_status === 'active' || fetched.credential_status === 'disabled') return true
      return Date.now() >= Date.parse(current.expiresAt)
    })().finally(() => {
      if (cancelInFlightRef.current?.pairingId === current.pairingId) cancelInFlightRef.current = null
    })
    cancelInFlightRef.current = { pairingId: current.pairingId, promise }
    return promise
  }, [queryClient, reconcileConnection])

  const finish = useCallback(async (
    ownGeneration: number,
    next: Phase,
    code: PairingFailureCode | null = null,
    cancelPending = false,
  ) => {
    if (ownGeneration !== generation.current) return
    const current = pairingRef.current
    const connectionId = connectionIdRef.current
    stopActivity()
    scrubVisible()
    clearPairingRefs()
    setFailure(code)
    setReconcileError(false)
    setPhase(next)

    let authoritative: LlmConnection | null = null
    if (current && connectionId && cancelPending) {
      void cancelAndReconcile(current, connectionId)
    } else if (connectionId) {
      authoritative = await reconcileConnection(connectionId)
    }
    if (ownGeneration !== generation.current || next !== 'completed' || !connectionId) return
    if (authoritative) setSelectedModel(authoritative.default_model ?? '')
    onCompletedRef.current?.()
    let testedModels: OAuthModel[] = []
    try {
      await cancelAgentConnectionsList(queryClient)
      const result = await agents.connections.test(connectionId)
      testedModels = normalizeOAuthModels(result.models)
    } catch {
      // Never expose server or transport details from the credential test.
    } finally {
      await reconcileConnection(connectionId)
    }
    if (ownGeneration === generation.current) setModels(testedModels)
  }, [cancelAndReconcile, clearPairingRefs, queryClient, reconcileConnection, scrubVisible, stopActivity])

  const connectionId = connection?.id ?? null
  const currentLocation = browserLocation ?? (typeof window !== 'undefined' ? window.location : null)
  const secureOrigin = currentLocation?.protocol === 'https:'

  useEffect(() => {
    if (!open || !connectionId) return
    const ownGeneration = ++generation.current
    connectionIdRef.current = connectionId
    modelPendingRef.current = false
    modelOperationRef.current += 1
    setModelPending(false)
    setPhase(secureOrigin ? 'starting' : 'idle')
    setFailure(null)
    setReconcileError(false)
    setModels([])
    setSelectedModel(connection?.default_model ?? '')
    scrubVisible()
    if (!secureOrigin) return () => { generation.current += 1; stopActivity(); scrubVisible(false); clearPairingRefs() }

    const poll = async (current: PublicPairing, consecutiveFailures = 0): Promise<void> => {
      if (ownGeneration !== generation.current) return
      if (Date.now() >= Date.parse(current.expiresAt)) {
        await finish(ownGeneration, 'expired', null, true)
        return
      }
      const controller = new AbortController()
      abortRef.current = controller
      try {
        const status = await agents.connections.pairingStatus(connectionId, current.pairingId, controller.signal)
        if (ownGeneration !== generation.current || controller.signal.aborted) return
        setFailure(status.failure_code ?? null)
        setPhase(status.state)
        if (TERMINAL.has(status.state)) {
          await finish(ownGeneration, status.state, status.failure_code ?? null)
          return
        }
        consecutiveFailures = 0
      } catch (error) {
        if (ownGeneration !== generation.current || controller.signal.aborted) return
        if (errorStatus(error) === 404) {
          await finish(ownGeneration, 'expired')
          return
        }
        consecutiveFailures += 1
        setPhase('network')
      }
      if (ownGeneration === generation.current) {
        const delay = Math.min(POLL_DELAY_MS * 2 ** Math.min(consecutiveFailures, 3), MAX_POLL_DELAY_MS)
        pollTimerRef.current = setTimeout(() => void poll(current, consecutiveFailures), delay)
      }
    }

    startTimerRef.current = setTimeout(() => {
      startTimerRef.current = null
      if (ownGeneration !== generation.current) return
      const controller = new AbortController()
      abortRef.current = controller
      void agents.connections.startPairing(connectionId, controller.signal).then((started) => {
        const current = { pairingId: started.pairing_id, expiresAt: started.expires_at }
        if (ownGeneration !== generation.current || controller.signal.aborted) {
          started.secretBytes.fill(0)
          void cancelAndReconcile(current, connectionId)
          return
        }
        secretBytesRef.current?.fill(0)
        secretBytesRef.current = started.secretBytes
        pairingRef.current = current
        setPairing(current)
        setHasSecret(true)
        setPhase('pending')
        const expiresIn = Date.parse(current.expiresAt) - Date.now()
        if (!Number.isFinite(expiresIn) || expiresIn <= 0) {
          void finish(ownGeneration, 'expired', null, true)
          return
        }
        expiryTimerRef.current = setTimeout(
          () => void finish(ownGeneration, 'expired', null, true),
          Math.min(expiresIn, 2_147_483_647),
        )
        void poll(current)
      }).catch(() => {
        if (ownGeneration === generation.current && !controller.signal.aborted) {
          stopActivity()
          scrubVisible()
          clearPairingRefs()
          setPhase('failed')
        }
      })
    }, 0)

    return () => {
      generation.current += 1
      modelOperationRef.current += 1
      modelPendingRef.current = false
      const current = pairingRef.current
      const currentConnectionId = connectionIdRef.current
      stopActivity()
      scrubVisible(false)
      clearPairingRefs()
      if (current && currentConnectionId) void cancelAndReconcile(current, currentConnectionId)
    }
  }, [cancelAndReconcile, clearPairingRefs, connection?.default_model, connectionId, finish, open, scrubVisible, secureOrigin, stopActivity])

  const requestClose = async () => {
    if (phase === 'starting' || phase === 'reconciling') return
    const ownGeneration = generation.current
    const current = pairingRef.current
    const currentConnectionId = connectionIdRef.current
    stopActivity()
    scrubVisible()
    setFailure(null)
    setReconcileError(false)

    if (current && currentConnectionId && !TERMINAL.has(phase)) {
      setPhase('reconciling')
      const reconciled = await cancelAndReconcile(current, currentConnectionId)
      if (ownGeneration !== generation.current) return
      if (!reconciled) {
        setReconcileError(true)
        setPhase('network')
        const remaining = Date.parse(current.expiresAt) - Date.now()
        if (Number.isFinite(remaining) && remaining > 0) {
          expiryTimerRef.current = setTimeout(() => { void requestClose() }, Math.min(remaining, 2_147_483_647))
        }
        return
      }
    }
    if (ownGeneration !== generation.current) return
    clearPairingRefs()
    setPhase('idle')
    onOpenChange(false)
  }

  const copy = async (value: string, label: string) => {
    try {
      await navigator.clipboard.writeText(value)
      toast.success(t('agents.connections.chatgpt.copied', { label }))
    } catch {
      toast.error(t('agents.connections.chatgpt.copyFailed'))
    }
  }

  const copySecret = async () => {
    const bytes = secretBytesRef.current
    if (!bytes) return
    let ephemeral = new TextDecoder().decode(bytes)
    try {
      await navigator.clipboard.writeText(ephemeral)
      toast.success(t('agents.connections.chatgpt.copied', { label: t('agents.connections.chatgpt.secret') }))
    } catch {
      toast.error(t('agents.connections.chatgpt.copyFailed'))
    } finally {
      ephemeral = ''
    }
  }

  const saveModel = async (slug: string) => {
    if (!connectionId || modelPendingRef.current) return
    const ownGeneration = generation.current
    const operation = ++modelOperationRef.current
    const prior = selectedModel
    modelPendingRef.current = true
    setModelPending(true)
    setSelectedModel(slug)
    let failed = false
    try {
      await cancelAgentConnectionsList(queryClient)
      await agents.connections.update(connectionId, { default_model: slug })
      if (ownGeneration === generation.current && operation === modelOperationRef.current) onCompletedRef.current?.()
    } catch {
      failed = true
      if (ownGeneration === generation.current && operation === modelOperationRef.current) toast.error(t('agents.connections.saveFailed'))
    } finally {
      const authoritative = await reconcileConnection(connectionId)
      if (ownGeneration === generation.current && operation === modelOperationRef.current) {
        if (authoritative) setSelectedModel(authoritative.default_model ?? '')
        else if (failed) setSelectedModel(prior)
        modelPendingRef.current = false
        setModelPending(false)
      }
    }
  }

  const command = pairing && currentLocation ? chatGptHelperCommand(pairing.pairingId, currentLocation) : null
  const statusKey = `agents.connections.chatgpt.status.${phase}`
  const failureKey = failure ? SAFE_FAILURES[failure] : null
  const nondismissible = phase === 'starting' || phase === 'reconciling'

  return (
    <Dialog open={open} onOpenChange={(next) => { if (!next) void requestClose() }}>
      <DialogContent
        className="flex max-h-[calc(100dvh-2rem)] max-w-2xl flex-col overflow-hidden p-4 sm:p-6"
        showCloseButton={!nondismissible}
        onOpenAutoFocus={(event) => { event.preventDefault(); titleRef.current?.focus() }}
        onEscapeKeyDown={(event) => { if (nondismissible) event.preventDefault() }}
        onPointerDownOutside={(event) => { if (nondismissible) event.preventDefault() }}
        onInteractOutside={(event) => { if (nondismissible) event.preventDefault() }}
      >
        <DialogHeader>
          <DialogTitle ref={titleRef} tabIndex={-1}>{t('agents.connections.chatgpt.title')}</DialogTitle>
          <DialogDescription>{t('agents.connections.chatgpt.description')}</DialogDescription>
        </DialogHeader>
        <div className="min-h-0 flex-1 space-y-4 overflow-y-auto">
          <div role="status" aria-live="polite" className="flex items-center gap-2 text-sm">
            {(phase === 'starting' || phase === 'pending' || phase === 'attempt_registered' || phase === 'completing' || phase === 'network' || phase === 'reconciling') && <Loader2 className="h-4 w-4 animate-spin" />}
            {t(statusKey)}
          </div>
          {!secureOrigin && <p role="alert" className="text-sm text-amber-700 dark:text-amber-300">{t('agents.connections.chatgpt.httpsRequired')}</p>}
          {failureKey && <p role="alert" className="text-sm text-rose-600 dark:text-rose-400">{t(failureKey)}</p>}
          {reconcileError && <p role="alert" className="text-sm text-rose-600 dark:text-rose-400">{t('agents.connections.chatgpt.reconcileFailed')}</p>}
          {pairing && hasSecret && (
            <div className="space-y-4">
              <p className="text-sm">{t('agents.connections.chatgpt.instructions')}</p>
              {command && <div className="space-y-2">
                <Label>{t('agents.connections.chatgpt.command')}</Label>
                <div className="flex items-start gap-2"><code className="min-w-0 flex-1 overflow-x-auto rounded bg-muted p-3 text-xs">{command}</code><Button variant="outline" size="icon" aria-label={t('agents.connections.chatgpt.copyCommand')} onClick={() => void copy(command, t('agents.connections.chatgpt.command'))}><Copy className="h-4 w-4" /></Button></div>
              </div>}
              <div className="space-y-2">
                <Label>{t('agents.connections.chatgpt.secret')}</Label>
                <p className="text-xs text-muted-foreground">{t('agents.connections.chatgpt.secretWarning')}</p>
                <div className="flex items-start gap-2"><code ref={attachSecretNode} data-testid="pairing-secret" className="min-w-0 flex-1 break-all rounded bg-muted p-3 text-xs" /><Button variant="outline" size="icon" aria-label={t('agents.connections.chatgpt.copySecret')} onClick={() => void copySecret()}><Copy className="h-4 w-4" /></Button></div>
              </div>
            </div>
          )}
          {phase === 'completed' && models.length > 0 && connectionId && <div className="space-y-2">
            <Label id="chatgpt-model-label">{t('agents.connections.chatgpt.chooseModel')}</Label>
            <Select value={selectedModel} disabled={modelPending} onValueChange={(slug) => void saveModel(slug)}>
              <SelectTrigger aria-labelledby="chatgpt-model-label"><SelectValue placeholder={t('agents.connections.chatgpt.chooseModel')} /></SelectTrigger>
              <SelectContent>{models.map((model) => <SelectItem key={model.slug} value={model.slug}>{model.display_name}</SelectItem>)}</SelectContent>
            </Select>
          </div>}
        </div>
        <DialogFooter>
          <Button variant="outline" onClick={() => void requestClose()} disabled={nondismissible}>{phase === 'completed' ? t('common.close') : t('agents.connections.cancel')}</Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  )
}
