import { useCallback, useEffect, useRef, useState } from 'react'
import { useQuery, useQueryClient } from '@tanstack/react-query'
import { Link } from 'react-router-dom'
import { useTranslation } from 'react-i18next'
import { ArrowLeft, CheckCircle2, Edit2, Plug, Plus, Star, Trash2, XCircle } from 'lucide-react'
import { toast } from 'sonner'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import { Dialog, DialogContent, DialogDescription, DialogFooter, DialogHeader, DialogTitle } from '@/components/ui/dialog'
import { PageHeader } from '@/components/page-header'
import { agents } from '@/lib/api'
import type { LlmConnection } from '@/lib/api'
import { AGENT_CONNECTIONS_KEY, cancelAgentConnectionsList, removeAgentConnection, upsertAgentConnection } from '@/lib/agent-connections-cache'
import { ConnectionFormDialog } from '@/components/agents/connection-form-dialog'
import { ChatGptPairingDialog } from '@/components/agents/chatgpt-pairing-dialog'
import { McpExternalPanel } from '@/components/agents/mcp-external-panel'


type ConfirmAction = { kind: 'reconnect' | 'disconnect' | 'delete'; connection: LlmConnection }
const PAIRABLE = new Set(['not_connected', 'revoked', 'reauth_required'])
const BLOCKING_ERRORS = new Set(['refresh_outcome_unknown', 'revocation_pending', 'resource_cleanup_pending'])
const CREDENTIAL_LABELS = new Set(['not_connected', 'active', 'disabled', 'revoked', 'reauth_required'])

function errorStatus(error: unknown): number | undefined {
  return (error as { response?: { status?: number } })?.response?.status
}

export default function AgentConnectionsPage() {
  const { t } = useTranslation()
  const qc = useQueryClient()

  const [editing, setEditing] = useState<LlmConnection | null>(null)
  const [createOpen, setCreateOpen] = useState(false)
  const [pairing, setPairing] = useState<LlmConnection | null>(null)
  const [confirmAction, setConfirmAction] = useState<ConfirmAction | null>(null)
  const [actionPending, setActionPending] = useState(false)
  const [testing, setTesting] = useState<Set<string>>(new Set())
  const [testResults, setTestResults] = useState<Record<string, { ok: boolean; detail: string }>>({})
  const actionPendingRef = useRef(false)
  const testPendingRef = useRef(new Set<string>())
  const testGenerationRef = useRef(new Map<string, number>())
  const previousListRef = useRef<LlmConnection[] | undefined>(undefined)

  const { data: list, isLoading } = useQuery({
    queryKey: AGENT_CONNECTIONS_KEY,
    queryFn: ({ signal }) => agents.connections.list(signal),
  })

  useEffect(() => {
    const previous = previousListRef.current
    previousListRef.current = list
    if (!previous || previous === list || testPendingRef.current.size === 0) return
    for (const id of testPendingRef.current) {
      testGenerationRef.current.set(id, (testGenerationRef.current.get(id) ?? 0) + 1)
    }
    testPendingRef.current.clear()
    setTesting(new Set())
  }, [list])

  const refresh = useCallback(async (id: string) => {
    try {
      const current = await agents.connections.get(id)
      await upsertAgentConnection(qc, current)
      return current
    } catch {
      return null
    }
  }, [qc])

  const canPair = (connection: LlmConnection) => PAIRABLE.has(connection.credential_status ?? '') && !BLOCKING_ERRORS.has(connection.credential_error_code ?? '')

  const revokeAndRefresh = async (connection: LlmConnection) => {
    await cancelAgentConnectionsList(qc)
    try { await agents.connections.revoke(connection.id) } catch { /* refetch determines ambiguous outcome */ }
    return refresh(connection.id)
  }

  const deleteAndReconcile = async (connection: LlmConnection) => {
    await cancelAgentConnectionsList(qc)
    try {
      await agents.connections.remove(connection.id)
    } catch {
      try {
        const survivor = await agents.connections.get(connection.id)
        await upsertAgentConnection(qc, survivor)
        toast.error(t('agents.connections.deleteFailed'))
        return
      } catch (reconcileError) {
        if (errorStatus(reconcileError) !== 404) {
          toast.error(t('agents.connections.deleteInconclusive', {
            defaultValue: 'Could not confirm whether the connection was removed. Refresh before trying again.',
          }))
          return
        }
      }
    }
    await removeAgentConnection(qc, connection.id)
    toast.success(t('agents.connections.deleted'))
    setConfirmAction(null)
  }

  const runConfirmedAction = async () => {
    if (!confirmAction || actionPendingRef.current) return
    actionPendingRef.current = true
    setActionPending(true)
    const { connection, kind } = confirmAction
    try {
      const after = await revokeAndRefresh(connection)
      if (!after || !canPair(after)) {
        toast.error(t('agents.connections.chatgpt.revokeUnresolved'))
        return
      }
      if (kind === 'reconnect') setPairing(after)
      if (kind === 'disconnect') toast.success(t('agents.connections.chatgpt.disconnected'))
      if (kind === 'delete') {
        await deleteAndReconcile(connection)
        return
      }
      setConfirmAction(null)
    } catch {
      toast.error(t(kind === 'delete' ? 'agents.connections.deleteFailed' : 'agents.connections.chatgpt.revokeUnresolved'))
    } finally {
      actionPendingRef.current = false
      setActionPending(false)
    }
  }

  const removePlain = async (connection: LlmConnection) => {
    if (actionPendingRef.current) return
    actionPendingRef.current = true
    setActionPending(true)
    try {
      await deleteAndReconcile(connection)
    } finally {
      actionPendingRef.current = false
      setActionPending(false)
    }
  }

  const testConnection = async (connection: LlmConnection) => {
    if (testPendingRef.current.has(connection.id)) return
    testPendingRef.current.add(connection.id)
    const ownGeneration = (testGenerationRef.current.get(connection.id) ?? 0) + 1
    testGenerationRef.current.set(connection.id, ownGeneration)
    setTesting((current) => new Set(current).add(connection.id))
    try {
      await cancelAgentConnectionsList(qc)
      const result = await agents.connections.test(connection.id)
      if (testGenerationRef.current.get(connection.id) !== ownGeneration) return
      await refresh(connection.id)
      if (testGenerationRef.current.get(connection.id) !== ownGeneration) return
      const detail = connection.kind === 'openai_chatgpt'
        ? t(result.ok ? 'agents.connections.chatgpt.testSuccess' : 'agents.connections.chatgpt.testFailed')
        : result.detail
      setTestResults((current) => ({ ...current, [connection.id]: { ok: result.ok, detail } }))
    } catch {
      if (testGenerationRef.current.get(connection.id) !== ownGeneration) return
      setTestResults((current) => ({ ...current, [connection.id]: { ok: false, detail: t('agents.connections.chatgpt.testFailed') } }))
    } finally {
      if (testGenerationRef.current.get(connection.id) === ownGeneration) {
        testPendingRef.current.delete(connection.id)
        setTesting((current) => {
          const next = new Set(current)
          next.delete(connection.id)
          return next
        })
      }
    }
  }

  const statusLabel = (connection: LlmConnection) => {
    const status = connection.credential_status ?? 'not_connected'
    return CREDENTIAL_LABELS.has(status)
      ? t(`agents.connections.chatgpt.credential.${status}`)
      : t('agents.connections.chatgpt.credential.not_connected')
  }
  const providerLabel = (connection: LlmConnection) => connection.kind === 'openai_chatgpt' ? t('agents.connections.chatgpt.provider') : connection.kind

  return <div>
    <Link to="/agents" className="mb-2 inline-flex items-center gap-1 text-xs text-muted-foreground transition-colors hover:text-foreground">
      <ArrowLeft className="h-3.5 w-3.5" /> {t('agents.title')}
    </Link>
    {/* Connection credentials are requester-owned and current_user-scoped on the backend;
        workspace write roles intentionally do not gate these controls. */}
    <PageHeader section={t('agents.title')} title={t('agents.connections.title')} action={
      <Button size="sm" className="h-8 gap-1.5" disabled={isLoading} onClick={() => { if (!isLoading) setCreateOpen(true) }}><Plus size={13} /> {t('agents.connections.add')}</Button>
    } />
    <p className="mb-6 max-w-2xl text-sm text-muted-foreground">{t('agents.connections.subtitle')}</p>

    {isLoading ? <div className="rounded-xl border bg-card p-6 text-sm text-muted-foreground">{t('agents.loading')}</div> : (list?.length ?? 0) === 0 ?
      <div className="rounded-xl border bg-card p-10 text-center shadow-sm"><Plug className="mx-auto mb-3 h-8 w-8 text-muted-foreground" /><h2 className="font-semibold">{t('agents.connections.empty.title')}</h2><p className="mt-1 text-sm text-muted-foreground">{t('agents.connections.empty.subtitle')}</p></div> :
      <div className="overflow-hidden rounded-xl border bg-card shadow-sm"><div className="divide-y">
        {list?.map((connection) => {
          const chatgpt = connection.kind === 'openai_chatgpt'
          const active = connection.credential_status === 'active' || connection.credential_status === 'disabled'
          const tested = testResults[connection.id]
          return <div key={connection.id} className="flex flex-col gap-3 px-4 py-4 sm:flex-row sm:items-start sm:px-5">
            <div className="flex min-w-0 flex-1 gap-3">
              <div className="flex h-10 w-10 shrink-0 items-center justify-center rounded-md bg-muted"><Plug className="h-5 w-5 text-muted-foreground" /></div>
              <div className="min-w-0 flex-1">
                <div className="flex flex-wrap items-center gap-2"><span className="truncate text-sm font-semibold">{connection.name}</span><Badge variant="secondary">{providerLabel(connection)}</Badge>{chatgpt && <Badge variant={active ? 'default' : 'outline'}>{statusLabel(connection)}</Badge>}{connection.is_default && <Badge className="gap-1 bg-amber-100 text-amber-800 hover:bg-amber-100 dark:bg-amber-900/40 dark:text-amber-200"><Star className="h-3 w-3" />{t('agents.connections.default')}</Badge>}</div>
                <div className="mt-1 space-y-0.5 text-xs text-muted-foreground">
                  {chatgpt ? <><div>{connection.account_email ?? t('agents.connections.chatgpt.noAccount')}</div><div>{connection.default_model ?? t('agents.connections.noDefaultModel')}</div>{connection.access_token_expires_at && <div>{t('agents.connections.chatgpt.expires', { date: new Date(connection.access_token_expires_at).toLocaleString() })}</div>}{connection.credential_error_code && <div className="text-rose-600">{t(`agents.connections.chatgpt.errors.${connection.credential_error_code}`, { defaultValue: t('agents.connections.chatgpt.errors.unknown') })}</div>}</> : <div>{connection.base_url || t('agents.connections.providerDefaultEndpoint')} · {connection.default_model || t('agents.connections.noDefaultModel')}</div>}
                </div>
                {tested && <div role="status" className={`mt-2 inline-flex items-center gap-1.5 text-xs ${tested.ok ? 'text-emerald-600' : 'text-rose-600'}`}>{tested.ok ? <CheckCircle2 className="h-3.5 w-3.5" /> : <XCircle className="h-3.5 w-3.5" />}{tested.detail}</div>}
              </div>
            </div>
            <div className="flex flex-wrap items-center gap-2 sm:justify-end">
              {chatgpt && !active && canPair(connection) && <Button size="sm" onClick={() => setPairing(connection)}>{connection.needs_reauthorization ? t('agents.connections.chatgpt.signInAgain') : t('agents.connections.chatgpt.signIn')}</Button>}
              {chatgpt && active && <><Button size="sm" variant="outline" onClick={() => setConfirmAction({ kind: 'reconnect', connection })}>{t('agents.connections.chatgpt.reconnect')}</Button><Button size="sm" variant="outline" onClick={() => setConfirmAction({ kind: 'disconnect', connection })}>{t('agents.connections.chatgpt.disconnect')}</Button></>}
              <Button size="sm" variant="outline" disabled={testing.has(connection.id)} onClick={() => void testConnection(connection)}>{t('agents.connections.test')}</Button>
              <Button size="icon" variant="ghost" aria-label={t('common.edit')} onClick={() => setEditing(connection)}><Edit2 className="h-4 w-4" /></Button>
              <Button size="icon" variant="ghost" aria-label={t('common.delete')} onClick={() => setConfirmAction({ kind: 'delete', connection })}><Trash2 className="h-4 w-4" /></Button>
            </div>
          </div>
        })}
      </div></div>}

    <McpExternalPanel />
    <ConnectionFormDialog open={createOpen} onOpenChange={setCreateOpen} onSaved={(saved) => { if (saved.kind === 'openai_chatgpt') setPairing(saved) }} />
    <ConnectionFormDialog open={!!editing} onOpenChange={(next) => { if (!next) setEditing(null) }} connection={editing} />
    <ChatGptPairingDialog open={!!pairing} connection={pairing} onOpenChange={(next) => { if (!next) setPairing(null) }} onReconciled={(current) => upsertAgentConnection(qc, current)} />
    <Dialog open={!!confirmAction} onOpenChange={(next) => { if (!next && !actionPending) setConfirmAction(null) }}>
      <DialogContent showCloseButton={!actionPending}>
        <DialogHeader><DialogTitle>{confirmAction && t(`agents.connections.chatgpt.confirm.${confirmAction.kind}.title`)}</DialogTitle><DialogDescription>{confirmAction && t(`agents.connections.chatgpt.confirm.${confirmAction.kind}.description`, { name: confirmAction.connection.name })}</DialogDescription></DialogHeader>
        <DialogFooter><Button variant="outline" disabled={actionPending} onClick={() => setConfirmAction(null)}>{t('common.cancel')}</Button><Button variant={confirmAction?.kind === 'delete' ? 'destructive' : 'default'} disabled={actionPending} onClick={() => { if (confirmAction?.kind === 'delete' && confirmAction.connection.kind !== 'openai_chatgpt') void removePlain(confirmAction.connection); else void runConfirmedAction() }}>{confirmAction && t(`agents.connections.chatgpt.confirm.${confirmAction.kind}.action`)}</Button></DialogFooter>
      </DialogContent>
    </Dialog>
  </div>
}
