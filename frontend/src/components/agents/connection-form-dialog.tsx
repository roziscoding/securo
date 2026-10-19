import { useRef, useState } from 'react'
import { useMutation, useQueryClient } from '@tanstack/react-query'
import { useTranslation } from 'react-i18next'
import { toast } from 'sonner'
import { Button } from '@/components/ui/button'
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from '@/components/ui/dialog'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { Switch } from '@/components/ui/switch'
import { agents } from '@/lib/api'
import type { LlmConnection, LlmConnectionKind } from '@/lib/api'
import { cancelAgentConnectionsList, upsertAgentConnection } from '@/lib/agent-connections-cache'

interface Props {
  open: boolean
  onOpenChange: (open: boolean) => void
  connection?: LlmConnection | null
  onSaved?: (connection: LlmConnection) => void
}

const KIND_LABELS: Record<LlmConnectionKind, { label: string; needsBaseUrl: boolean; needsKey: boolean; modelHint: string; urlHint: string; defaultModel?: string }> = {
  ollama: { label: 'Ollama', needsBaseUrl: false, needsKey: false, modelHint: 'llama3.1:8b', urlHint: 'http://host.docker.internal:11434' },
  // OpenAI + Anthropic ship a sensible default model so non-tech users
  // don't need to know which id to type. Self-hosted kinds (Ollama,
  // OpenAI-compatible) intentionally have NO default — the right model
  // depends on what the user actually has installed/loaded.
  openai: { label: 'OpenAI', needsBaseUrl: false, needsKey: true, modelHint: 'gpt-4o-mini', urlHint: '', defaultModel: 'gpt-4o-mini' },
  anthropic: { label: 'Anthropic', needsBaseUrl: false, needsKey: true, modelHint: 'claude-haiku-4-5', urlHint: '', defaultModel: 'claude-haiku-4-5' },
  openai_compatible: { label: 'OpenAI-compatible (LM Studio, vLLM, Groq, Together, …)', needsBaseUrl: true, needsKey: false, modelHint: 'llama3.1-70b', urlHint: 'http://192.168.1.142:1234' },
  openai_chatgpt: { label: 'Sign in with ChatGPT', needsBaseUrl: false, needsKey: false, modelHint: 'Select after signing in', urlHint: '' },
}

export function ConnectionFormDialog({ open, onOpenChange, connection, onSaved }: Props) {
  const { t } = useTranslation()
  const qc = useQueryClient()
  const isEdit = !!connection

  const [name, setName] = useState('')
  const [kind, setKind] = useState<LlmConnectionKind>('ollama')
  const [baseUrl, setBaseUrl] = useState('')
  const [apiKey, setApiKey] = useState('')
  const [keyTouched, setKeyTouched] = useState(false)
  const [defaultModel, setDefaultModel] = useState('')
  // Track whether the user typed in the model field. If they didn't,
  // switching kinds re-seeds the field with the new kind's default
  // model (where one exists). Once they type, we never overwrite.
  const [modelTouched, setModelTouched] = useState(false)
  const [isDefault, setIsDefault] = useState(false)
  const savingRef = useRef(false)

  /** Switch the connector kind and, when the user is in the create
   *  flow and hasn't customized the model field, auto-fill the new
   *  kind's preset (only OpenAI + Anthropic ship one). */
  const handleKindChange = (newKind: LlmConnectionKind) => {
    setKind(newKind)
    if (isEdit) return
    if (modelTouched) return
    setDefaultModel(KIND_LABELS[newKind].defaultModel ?? '')
  }

  const [formSource, setFormSource] = useState<{ connection: typeof connection; open: typeof open } | null>(null)
  if (!formSource || formSource.connection !== connection || formSource.open !== open) {
    setFormSource({ connection, open })
    if (connection) {
      setName(connection.name)
      setKind(connection.kind)
      setBaseUrl(connection.base_url ?? '')
      setApiKey('')
      setKeyTouched(false)
      setDefaultModel(connection.default_model ?? '')
      // Editing a saved connection — treat the field as already
      // user-touched so kind swaps don't clobber what's stored.
      setModelTouched(true)
      setIsDefault(connection.is_default)
    } else {
      setName('')
      setKind('ollama')
      setBaseUrl('')
      setApiKey('')
      setKeyTouched(false)
      // Seed the model field with the initial kind's default (none
      // for ollama). Switching kinds inside the form will re-seed via
      // handleKindChange.
      setDefaultModel(KIND_LABELS['ollama'].defaultModel ?? '')
      setModelTouched(false)
      setIsDefault(false)
    }
  }

  const meta = KIND_LABELS[kind]

  const saveMut = useMutation({
    mutationFn: async () => {
      await cancelAgentConnectionsList(qc)
      if (kind === 'openai_chatgpt') {
        if (isEdit && connection) {
          return agents.connections.update(connection.id, {
            name: name.trim(),
            default_model: defaultModel.trim() || null,
            is_default: isDefault,
          })
        }
        return agents.connections.create({ name: name.trim(), kind, is_default: isDefault })
      }
      const payload: Parameters<typeof agents.connections.create>[0] = {
        name: name.trim(),
        kind,
        base_url: meta.needsBaseUrl ? baseUrl.trim() || null : baseUrl.trim() || null,
        default_model: defaultModel.trim() || null,
        is_default: isDefault,
      }
      // Only include api_key in the payload if user actually typed one. This
      // preserves the existing key on edits where the user didn't touch the field.
      if (!isEdit || keyTouched) {
        payload.api_key = apiKey || null
      }
      if (isEdit && connection) return agents.connections.update(connection.id, payload)
      return agents.connections.create(payload)
    },
    onSuccess: async (saved) => {
      await upsertAgentConnection(qc, saved)
      onOpenChange(false)
      toast.success(isEdit ? t('agents.connections.updated') : t('agents.connections.created'))
      onSaved?.(saved)
    },
    onError: () => {
      toast.error(t('agents.connections.saveFailed'))
    },
    onSettled: () => {
      savingRef.current = false
    },
  })

  const submit = () => {
    if (savingRef.current) return
    if (!name.trim()) {
      toast.error(t('agents.connections.nameRequired'))
      return
    }
    if (meta.needsBaseUrl && !baseUrl.trim()) {
      toast.error(t('agents.connections.baseUrlRequired'))
      return
    }
    savingRef.current = true
    saveMut.mutate()
  }

  return (
    <Dialog open={open} onOpenChange={(next) => { if (!savingRef.current) onOpenChange(next) }}>
      <DialogContent className="max-w-xl" showCloseButton={!saveMut.isPending}>
        <DialogHeader>
          <DialogTitle>{isEdit ? t('agents.connections.editTitle') : t('agents.connections.createTitle')}</DialogTitle>
          <DialogDescription className="sr-only">{t('agents.connections.subtitle')}</DialogDescription>
        </DialogHeader>
        <div className="grid gap-4 py-2">
          <div className="grid gap-2">
            <Label htmlFor="connection-name">{t('agents.connections.name')}</Label>
            <Input id="connection-name" disabled={saveMut.isPending} value={name} onChange={(e) => setName(e.target.value)} placeholder={t('agents.connections.namePlaceholder')} />
          </div>
          <div className="grid gap-2">
            <Label id="connection-kind-label">{t('agents.connections.kind')}</Label>
            <Select value={kind} onValueChange={(v) => handleKindChange(v as LlmConnectionKind)} disabled={isEdit || saveMut.isPending}>
              <SelectTrigger aria-labelledby="connection-kind-label">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                {(Object.keys(KIND_LABELS) as LlmConnectionKind[]).map((k) => (
                  <SelectItem key={k} value={k}>
                    {KIND_LABELS[k].label}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>
          {kind !== 'openai_chatgpt' && <div className="grid gap-2">
            <Label htmlFor="connection-base-url">
              {t('agents.connections.baseUrl')}
              {!meta.needsBaseUrl && <span className="text-xs text-muted-foreground ml-1">({t('agents.connections.optional')})</span>}
            </Label>
            <Input
              id="connection-base-url"
              disabled={saveMut.isPending}
              value={baseUrl}
              onChange={(e) => setBaseUrl(e.target.value)}
              placeholder={meta.urlHint || t('agents.connections.baseUrlPlaceholder')}
            />
            {kind === 'ollama' && (
              <p className="text-xs text-muted-foreground">
                {t('agents.connections.ollamaHint')}
              </p>
            )}
            {kind === 'openai_compatible' && (
              <p className="text-xs text-muted-foreground">
                {t('agents.connections.openaiCompatHint')}
              </p>
            )}
          </div>}
          {kind !== 'openai_chatgpt' && <div className="grid gap-2">
            <Label htmlFor="connection-api-key">
              {t('agents.connections.apiKey')}
              {!meta.needsKey && <span className="text-xs text-muted-foreground ml-1">({t('agents.connections.optional')})</span>}
            </Label>
            <Input
              id="connection-api-key"
              disabled={saveMut.isPending}
              type="password"
              value={apiKey}
              onChange={(e) => {
                setApiKey(e.target.value)
                setKeyTouched(true)
              }}
              placeholder={isEdit && connection?.has_api_key ? t('agents.connections.apiKeyPlaceholderEdit') : t('agents.connections.apiKeyPlaceholder')}
            />
            <p className="text-xs text-muted-foreground">{t('agents.connections.apiKeyHint')}</p>
          </div>}
          {(kind !== 'openai_chatgpt' || isEdit) && <div className="grid gap-2">
            <Label htmlFor="connection-model">{t('agents.connections.defaultModel')}</Label>
            <Input
              id="connection-model"
              disabled={saveMut.isPending}
              value={defaultModel}
              onChange={(e) => {
                setDefaultModel(e.target.value)
                setModelTouched(true)
              }}
              placeholder={meta.modelHint}
            />
          </div>}
          <div className="flex items-center gap-2">
            <Switch id="connection-default" disabled={saveMut.isPending} checked={isDefault} onCheckedChange={(v) => setIsDefault(!!v)} />
            <Label htmlFor="connection-default" className="cursor-pointer">
              {t('agents.connections.setAsDefault')}
            </Label>
          </div>
        </div>
        <DialogFooter>
          <Button variant="outline" disabled={saveMut.isPending} onClick={() => onOpenChange(false)}>
            {t('agents.connections.cancel')}
          </Button>
          <Button onClick={submit} disabled={saveMut.isPending}>
            {saveMut.isPending ? t('agents.connections.saving') : isEdit ? t('agents.connections.save') : kind === 'openai_chatgpt' ? t('agents.connections.createContinue') : t('agents.connections.create')}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  )
}
