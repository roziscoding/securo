import type { OAuthModel } from '@/lib/api'

export function chatGptHelperCommand(pairingId: string, location: Pick<Location, 'protocol' | 'origin'>): string | null {
  return location.protocol === 'https:' ? `securo-siwc ${pairingId} --server ${location.origin}` : null
}

export function normalizeOAuthModels(models: Array<string | OAuthModel> = []): OAuthModel[] {
  return models.map((model) => typeof model === 'string' ? { slug: model, display_name: model } : model)
}
