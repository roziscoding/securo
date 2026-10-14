# Sign in with ChatGPT no Securo — estudo de implementação

**Repositório analisado:** `securo-finance/securo`
**Commit:** `b0ad4cf20350621881a870744468e979752e58a8`
**Escopo:** permitir que uma conexão de agent use a assinatura ChatGPT do usuário sem API key.

## Resumo executivo

É tecnicamente possível, mas **não é apenas trocar uma API key por um access token**.

O caminho oficial para projetos open source usa registro dinâmico (`client_id=dynamic_agent_client`), Authorization Code + PKCE, um `client_id` emitido por usuário/workspace e não exige client secret nem partner API key.[6] O token deve ser usado no endpoint público `POST https://api.openai.com/v1/responses`; a documentação manda explicitamente não usar endpoints privados `chatgpt.com/backend-api`.[8]

Há dois impactos arquiteturais importantes:

1. O provider atual do Securo implementa Chat Completions (`/v1/chat/completions`). SIWC exige Responses API, `store: false`, `stream: true`, contexto completo em `input` e um conjunto relevante de limitações de preview.[8][13]
2. O callback OSS deve ser um listener HTTP em `127.0.0.1` na máquina que abriu o navegador. Num Securo rodando num servidor/VM, o navegador normalmente está em outra máquina; a própria OpenAI orienta concluir OAuth localmente e transferir as credenciais protegidas para a VM.[6][10]

### Recomendação

Implementar uma conexão separada, internamente chamada `openai_chatgpt`, com:

- um provider dedicado de Responses API;
- uma tabela separada de credenciais OAuth;
- refresh/revogação serializados;
- um **helper local de pareamento** para o fluxo OAuth quando o Securo estiver remoto;
- UI que apresente “OpenAI — ChatGPT plan” ao lado de “OpenAI — API key”.

Não recomendo copiar o callback web usado hoje pelas conexões bancárias: ele conflita com a exigência de loopback do fluxo OSS. Também não recomendo copiar o fluxo interno do Codex ou usar `chatgpt.com/backend-api`; isso contradiz o contrato SIWC oficial atual.[8]

## 1. Elegibilidade e limites do produto

A documentação é voltada a apps open source e localmente hospedados. Para um app pago ou remotamente hospedado, a OpenAI pede o preenchimento do interest form.[16] Isso significa:

| Cenário | Situação |
|---|---|
| Securo OSS instalado pelo usuário em máquina local | encaixe direto no fluxo documentado |
| Securo OSS self-hosted em servidor/VM do usuário | suportado, mas requer OAuth local + transferência protegida de credenciais[10] |
| SaaS hospedado pelos mantenedores do Securo | tratar como dependente de contato/aprovação da OpenAI[16] |
| Usar client ID fixo do fluxo de website | fluxo distinto; website SIWC espera client ID fornecido pela OpenAI[17] |

O registro OSS é dinâmico. Na primeira autorização, o app envia `dynamic_agent_client`; o callback devolve um `client_id` emitido, normalmente `oaiapp_...`, que deve ser salvo e reutilizado para aquela identidade/workspace. `dynamic_agent_client` nunca deve ser persistido como o client ID real.[6]

## 2. Contrato oficial que precisa ser implementado

### 2.1 Autorização

Endpoint de autorização:

```text
https://auth.openai.com/api/accounts/authorize
```

A primeira tentativa precisa enviar:

```text
client_id=dynamic_agent_client
agent_name_hint=Securo
ext_agent_host_id=<ID ESTÁVEL DO HOST>
response_type=code
redirect_uri=http://127.0.0.1:<PORTA>/auth/callback
scope=openid profile email offline_access resource.invoke chatgpt.tokens.use.direct
resource=https://api.openai.com/v1
state=<ALEATÓRIO>
nonce=<ALEATÓRIO>
code_challenge_method=S256
code_challenge=<BASE64URL_SHA256_DO_VERIFIER_SEM_PADDING>
```

O callback deve usar literalmente `127.0.0.1`, não `localhost`; esquema, host e path permanecem iguais e apenas a porta pode mudar. A URI usada na troca do código deve ser exatamente a mesma daquela tentativa.[6]

O `ext_agent_host_id` identifica o runtime/host e deve ser estável, opaco e persistido antes do primeiro login. A OpenAI recomenda um JWK thumbprint URI e também aceita `urn:uuid:<UUIDv4>`.[16]

### 2.2 Troca do código

Depois de validar `state`, o helper envia form-urlencoded para:

```text
POST https://auth.openai.com/api/accounts/oauth/token
```

com:

```text
grant_type=authorization_code
client_id=<CLIENT_ID EMITIDO>
code=<CODE>
code_verifier=<PKCE VERIFIER>
redirect_uri=<URI EXATA DA TENTATIVA>
resource=https://api.openai.com/v1
```

Não há client secret no fluxo OSS.[6] O discovery document atual confirma Authorization Code, refresh token, PKCE S256, ID tokens RS256 e autenticação pública `none` no token endpoint.[15]

### 2.3 Validação

Validar o ID token, não apenas decodificá-lo:

- assinatura contra o JWKS publicado;
- `iss == https://auth.openai.com`;
- `aud` igual ao client ID emitido;
- expiração;
- `nonce` da tentativa;
- `sub` presente e válido.

Depois, verificar no **token response** se `chatgpt.tokens.use.direct` foi realmente concedido. Um ID token válido sozinho não autoriza uso da assinatura.[6]

### 2.4 Tokens, refresh e revogação

O token response inclui access token, refresh token, ID token, tipo, `expires_in`, scopes e `earliest_refresh_at`. Access tokens duram uma hora; refresh tokens duram 30 dias e cada refresh bem-sucedido devolve um refresh token substituto com nova janela de 30 dias.[11]

O refresh usa o client ID emitido, nunca `dynamic_agent_client`, e deve ser serializado por sessão para impedir reutilização concorrente de um token rotativo.[7]

A revogação é um POST form-urlencoded ao `revocation_endpoint`, enviando o refresh token, `token_type_hint=refresh_token` e o client ID emitido. Depois da revogação, os tokens locais devem ser removidos; falha de rede deve ser reportada sem reter credenciais indefinidamente como se o logout tivesse sido confirmado.[7]

### 2.5 Inference

Modelo disponível:

```http
GET https://api.openai.com/v1/models
Authorization: Bearer <ACCESS_TOKEN>
```

A resposta SIWC documentada usa `models[]`; a UI deve manter itens com `visibility == "list"`, mostrar `display_name` e enviar `slug` como model.[8]

Inference:

```http
POST https://api.openai.com/v1/responses
Authorization: Bearer <ACCESS_TOKEN>
Content-Type: application/json
```

Todo request HTTP desse fluxo precisa usar `store: false`, `stream: true` e enviar o contexto necessário em `input`. O sucesso só ocorre ao receber `response.completed`; `response.failed`, `response.incomplete` e stream encerrado sem completion são falhas separadas.[8]

Durante o preview, o Securo precisa omitir campos que hoje fazem parte do contrato Chat Completions, especialmente `temperature` e `max_output_tokens`. Também não pode usar `previous_response_id`; deve reenviar o histórico necessário. Ferramentas function/custom locais são possíveis, mas vários hosted tools e `tool_search` não são suportados.[13]

## 3. Estado atual do Securo

### Persistência

`backend/app/agents/models/connection.py::LlmConnection` contém:

- `kind`, `base_url`, `default_model` e `extra`;
- apenas um segredo, `api_key_encrypted: String(500)`;
- escopo por usuário;
- referência opcional dos agents por `connection_id`.

`extra` é devolvido integralmente por `ConnectionRead`, portanto **não pode guardar tokens**. `String(500)` também não é adequado para um bundle OAuth cifrado com JWTs.

A cifragem existente em `backend/app/agents/services/crypto.py` usa Fernet derivado de `SECRET_KEY`. Ela pode ser reutilizada, mas `decrypt()` converte ciphertext inválido em `None`; para OAuth isso precisa virar um estado explícito de “reauthorization required”, não uma credencial simplesmente ausente.

### Provider e runtime

`backend/app/agents/providers/openai.py::OpenAIProvider` usa:

- `POST /chat/completions`;
- payload `messages`;
- eventos SSE `choices[].delta`;
- ferramentas no formato Chat Completions;
- `temperature` em todo request.

Isso é incompatível com o fluxo SIWC. Apenas injetar o OAuth token em `api_key` faria o Securo chamar o protocolo errado.

O contrato neutro em `backend/app/agents/providers/base.py` (`ChatMessage`, `ToolDefinition`, `ChatChunk`) é suficiente para receber um novo adapter. O loop em `AgentExecutor` pode permanecer quase todo inalterado se o adapter converter eventos Responses para `ChatChunk`.

Há um problema independente a corrigir: `AgentExecutor.run()` captura qualquer erro de resolução do provider e faz fallback para credenciais de ambiente. Uma conexão OAuth corrompida ou expirada deve falhar fechada; não pode mudar silenciosamente de provider/conta.

### Frontend

A tela atual está concentrada em:

- `frontend/src/pages/agent-connections.tsx`;
- `frontend/src/components/agents/connection-form-dialog.tsx`;
- tipos e client em `frontend/src/lib/api.ts`.

O formulário pressupõe uma API key estática. O callback bancário em `frontend/src/pages/oauth-callback.tsx` é uma boa referência para state one-shot e proteção contra double mount do React, mas **não é um callback válido para o fluxo OSS SIWC remoto**, pois o redirect precisa terminar no loopback da máquina do navegador.[6][10]

## 4. Arquitetura proposta

### 4.1 Tipo de conexão

Adicionar um kind técnico separado:

```text
openai_chatgpt
```

Na UI ele pode aparecer agrupado sob OpenAI:

- **OpenAI — ChatGPT plan**
- **OpenAI — API key**

A separação evita condicionais espalhadas e deixa explícito que o wire protocol, catálogo de modelos, lifecycle de credenciais, erros e billing são diferentes. Ela também se encaixa no registry atual (`_PROVIDERS`) e no estilo modular do projeto.

### 4.2 Credenciais em tabela separada

Criar uma relação 1:1, por exemplo `agent_llm_oauth_credentials`:

```text
id                          uuid PK
connection_id               uuid UNIQUE FK -> agent_llm_connections ON DELETE CASCADE
issuer                      varchar(255)
subject                     varchar(255)
issued_client_id            varchar(255)
account_email               varchar(320) nullable
scopes                      jsonb
status                      varchar(32)   # active, disabled, expired, revoked, error
token_bundle_encrypted      text          # access + refresh + id token, atomicamente
access_token_expires_at     timestamptz
refresh_token_expires_at    timestamptz nullable
credential_version          integer
last_refreshed_at           timestamptz nullable
last_error_code             varchar(120) nullable
created_at / updated_at     timestamptz
```

O host ID deve ser uma configuração persistente da instância/runtime do Securo, não um valor escolhido por conexão nem copiado do laptop. Para uma VM, o helper usa o host ID da própria instância durante o OAuth; a documentação exige preservar o host ID da VM ao importar as credenciais.[10]

Um bundle JSON cifrado em uma única coluna `Text` torna a rotação de access/refresh/ID token atômica. Metadados necessários para listagem, status e scheduling permanecem fora do ciphertext; tokens nunca entram em `extra`, respostas da API, logs ou browser storage. A OpenAI exige manter access, refresh e retained ID tokens fora de browser storage, logs e analytics.[7]

### 4.3 Serviço OAuth

Criar `backend/app/agents/services/openai_siwc.py` com responsabilidades pequenas:

```text
build_authorization_url()
validate_id_token()
exchange_authorization_code()
import_credential_bundle()
ensure_fresh_access_token()
refresh_tokens()
revoke_tokens()
list_models()
mark_reauthorization_required()
```

Reutilizar a ideia de `oauth_state.py` para tokens one-shot, mas com namespace separado e payload vinculado a:

- user ID;
- connection draft/ID;
- pairing ID;
- PKCE/nonce apenas no helper local;
- host ID esperado;
- timestamp e expiração.

Refresh precisa de lock distribuído por connection ID, seguido de re-read/version check. Redis já existe no Securo e é adequado; PostgreSQL advisory lock também seria aceitável. Em um `401`, forçar no máximo um refresh e um retry.

### 4.4 Fluxo local/VM recomendado

O navegador não consegue criar sozinho um listener nativo confiável. Portanto, a experiência robusta é:

1. Na tela Connections, o usuário escolhe **OpenAI — ChatGPT plan**.
2. O frontend cria uma conexão pendente e um pairing one-shot no backend.
3. A UI mostra o pairing ID público e, separadamente, um pairing secret de uso único. O usuário inicia o helper somente com o ID público:

   ```text
   securo siwc login --pair <PAIRING_ID>
   ```

4. O helper solicita o pairing secret interativamente com entrada sem eco (`getpass`/TTY ou stdin protegido). O secret não entra na URL, command line, shell history ou process listing; automações podem fornecê-lo por stdin a partir de um secret store local.
5. O helper consulta o Securo via HTTPS, enviando o secret apenas no header `Authorization`, obtém o host ID público da instância e abre um listener temporário em `127.0.0.1`.
6. O helper gera state, nonce e PKCE; abre o navegador no authorize endpoint.
7. O callback retorna ao helper local.
8. O helper troca o código, valida ID token e scopes e envia o credential record ao backend reutilizando o bearer pairing secret no header — nunca na URL.
9. O backend revalida o ID token/scopes, cifra e persiste o bundle, consome o pairing e testa `GET /v1/models`.
10. O frontend faz polling curto ou recebe conclusão, invalida `['agent-connections']` e mostra a conta/status. O pairing secret deixa de ser exibido assim que o fluxo começa e nunca é persistido em browser storage.

Esse desenho implementa literalmente a orientação para self-hosted VM: OAuth local, transferência por canal protegido e refresh posterior pertencente à VM.[10]

#### Modo local opcional

Se frontend, navegador e backend estiverem comprovadamente na mesma máquina, um modo local pode deixar o próprio processo Securo abrir o loopback. Não deve ser o default em Docker nem ser inferido apenas pelo hostname da URL.

#### O que não fazer

- Não redirecionar a OpenAI para `/agents/connections/oauth/callback` no domínio do Securo usando o dynamic client.
- Não colocar access/refresh/ID tokens em query string, `localStorage`, `sessionStorage` ou estado React.
- Não pedir ao usuário para colar tokens manualmente.
- Não usar o client ID do Codex CLI.
- Não apontar requests para `chatgpt.com/backend-api`.[8]

### 4.5 Provider Responses

Criar `backend/app/agents/providers/openai_chatgpt.py::OpenAIChatGPTProvider`.

Ele deve:

- obter credenciais frescas por um resolver async, não receber um token estático no construtor;
- chamar exatamente `https://api.openai.com/v1/responses`;
- sempre definir `store=false` e `stream=true`;
- não enviar `temperature`, `max_output_tokens` nem outros campos bloqueados no preview;
- serializar o histórico completo em `input`;
- converter system prompt para `instructions` ou developer message;
- converter tools para o formato Responses function/custom;
- converter assistant tool calls e resultados em `function_call` / `function_call_output`, preservando `call_id`;
- traduzir `response.output_text.delta`, eventos de function call, usage, `response.completed`, `response.failed` e `response.incomplete` para o contrato `ChatChunk`;
- considerar a chamada bem-sucedida apenas após `response.completed`;
- definir `supports_embeddings = False`.

A configuração de embeddings do Securo já é instance-wide e separada das conexões de chat, então SIWC não deve ser reutilizado para RAG embeddings.

`connection_service.test_connection()` precisa de um branch dedicado: refresh se necessário, `GET /v1/models`, parse de `models[]`, filtro por `visibility`, e erros específicos de entitlement/usage.

### 4.6 API backend

Sugestão de rotas:

```text
POST   /api/agents/connections/chatgpt/pairings
GET    /api/agents/connections/chatgpt/pairings/{id}
POST   /api/agents/connections/chatgpt/pairings/{id}/complete   # helper
POST   /api/agents/connections/{id}/reauthorize
POST   /api/agents/connections/{id}/refresh
POST   /api/agents/connections/{id}/disconnect
GET    /api/agents/connections/{id}/models
```

A conclusão do pairing deve exigir dois fatores de posse:

- ID público do pairing;
- secret aleatório de alta entropia enviado em `Authorization`, armazenado somente como hash no backend.

`ConnectionRead` deve ganhar apenas dados sanitizados:

```text
auth_method: "chatgpt_oauth"
credential_status
account_email
scopes/granted_plan_access
access_token_expires_at
needs_reauthorization
```

### 4.7 Frontend

Mudanças principais:

- `frontend/src/lib/api.ts`
  - adicionar `openai_chatgpt` e tipos de pairing/status/model;
  - adicionar métodos start/status/reauthorize/disconnect/models.
- `frontend/src/components/agents/connection-form-dialog.tsx`
  - apresentar ChatGPT plan como opção própria;
  - esconder API key e base URL;
  - iniciar pairing;
  - renderizar comando/deep link do helper;
  - bloquear double submit e dismissal durante transições críticas.
- `frontend/src/pages/agent-connections.tsx`
  - mostrar conta, status, expiração, reauthorize/disconnect e Manage usage.
- novo componente/modal de pairing
  - progresso, timeout, retry e instruções diferentes para local/VM.
- locales
  - labels e erros em todos os arquivos de idioma existentes.

A ação deve usar o texto **Continue with ChatGPT**. Após o primeiro sucesso, mostrar uma confirmação equivalente a **You're using your ChatGPT plan**; durante o uso, mostrar **Using ChatGPT plan** e um link **Manage usage** para as configurações do ChatGPT.[14]

## 5. Tratamento de falhas e segurança

### Falhas permanentes

- refresh token reutilizado/expirado/inválido: limpar tokens inutilizáveis e pedir reautorização com o client ID emitido;
- `invalid_client`: erro de configuração, não loop de login;
- ausência de `chatgpt.tokens.use.direct`: conexão de identidade válida, mas plan usage desativado;
- identidade/client ID/workspace inesperado: rejeitar sem substituir a conexão ativa.[6][7][12]

### Falhas de inference

Mapear explicitamente:

- `subscription_sharing_user_not_eligible` → indisponível para user/workspace/policy; não repetir OAuth;
- `subscription_sharing_usage_limit_exceeded` → pausar requests e oferecer Manage usage;
- `subscription_sharing_unsupported_capability` → remover o campo/capability indicado, não repetir o mesmo body;
- falhas 503/temporárias → preservar credenciais e usar backoff limitado.

A OpenAI não faz fallback automático para outra forma de cobrança.[12] O Securo também não deve fazer fallback silencioso para a API key global.

### Hardening necessário

- sanitizar bodies de erro antes de SSE/UI/logs;
- nunca logar authorization URLs que incluam `id_token_hint`;
- limitar issuer, token endpoint e inference base URL a constantes/allowlist;
- não seguir redirects que reenviem Authorization para outro host;
- lock de refresh por conexão;
- troca atômica do bundle rotativo;
- vínculo explícito entre `connection_id` e proprietário do agent;
- revogação best-effort antes da exclusão, com aviso se não confirmada;
- status distinto para ciphertext ilegível após rotação de `SECRET_KEY`.

## 6. Migração e arquivos afetados

Backend:

```text
backend/alembic/versions/097_*.py
backend/app/agents/models/connection.py
backend/app/agents/models/oauth_credential.py             # novo
backend/app/agents/schemas/connection.py
backend/app/agents/api/connections.py
backend/app/agents/services/openai_siwc.py                 # novo
backend/app/agents/providers/openai_chatgpt.py             # novo
backend/app/agents/providers/registry.py
backend/app/agents/runtime/executor.py
backend/app/agents/config.py
```

Helper local:

```text
cli/securo_siwc/*                                           # novo, distribuição a decidir
```

Frontend:

```text
frontend/src/lib/api.ts
frontend/src/components/agents/connection-form-dialog.tsx
frontend/src/components/agents/chatgpt-pairing-dialog.tsx   # novo
frontend/src/pages/agent-connections.tsx
frontend/src/locales/*.json
```

## 7. Ordem de implementação

### Fase 0 — prova de contrato

Antes de integrar a UI:

1. criar um spike CLI mínimo que execute dynamic registration em `127.0.0.1`;
2. validar um token e listar os modelos;
3. executar uma chamada `store=false`, `stream=true` até `response.completed`;
4. testar um function call real;
5. confirmar o comportamento de `earliest_refresh_at` com uma resposta real;
6. testar conta Plus e, se possível, Pro e workspace distinto.

Critério de saída: endpoint público Responses funcionando sem client ID pré-provisionado no cenário OSS alvo.

### Fase 1 — backend e provider

- migration + credential table;
- SIWC service;
- provider Responses;
- refresh/revoke/locking;
- API de pairing;
- falha fechada no executor.

### Fase 2 — helper local

- listener loopback;
- PKCE/nonce/state;
- ID-token validation;
- transferência protegida one-shot;
- empacotamento/distribuição para Linux/macOS/Windows.

### Fase 3 — frontend

- opção de conexão;
- pairing modal;
- status/model picker;
- reauthorize/disconnect;
- UX de usage/limits.

### Fase 4 — hardening

- concorrência de refresh;
- rotação/corrupção de segredo;
- multi-account;
- revogação e network failures;
- observabilidade sem secrets;
- documentação de self-hosted VM.

## 8. Testes obrigatórios

Backend:

- PKCE, nonce, state replay/expiry e callback errors;
- validação JWKS/issuer/audience/expiry/subject;
- escopo direto ausente;
- import de pairing por usuário errado e secret repetido;
- token bundle nunca serializado;
- refresh rotation concorrente;
- retry único após 401;
- revogação confirmada e não confirmada;
- Responses serialization;
- SSE de texto, function calls, usage, completed/failed/incomplete;
- ausência de campos proibidos (`temperature`, `previous_response_id`, etc.);
- nenhum fallback de auth OAuth para credencial global;
- ownership de conexão entre usuários/workspaces.

Frontend/helper:

- API key e ChatGPT plan como caminhos independentes;
- double-click/StrictMode não duplicam pairing;
- pairing expirado e retry;
- helper fecha listener e apaga material temporário;
- conta/status/modelos corretos;
- reauthorize/disconnect;
- usage limit e link Manage usage;
- build, typecheck e suíte Vitest.

## 9. Decisão go/no-go

**Spike concluído e validado.** Dynamic registration funcionou com um client ID emitido, os seis scopes esperados, catálogo de sete modelos, inference pública em `gpt-6.1-sol` até `response.completed` e um function call real. O callback remoto funcionou por listener fixo em `127.0.0.1:1455` atravessando um túnel SSH `-L`; isso confirma a necessidade do helper local proposto.

**GO para implementar backend/provider/helper.** Ainda não começaria pela tela de Connections: refresh rotativo, revogação e pairing seguro devem existir antes da UX final. O castelo agora tem fundação; continua sendo prudente instalar as portas antes das cortinas.

## Sources

[6] https://developers.openai.com/siwc/token-sharing-open-source/sign-in
[7] https://developers.openai.com/siwc/token-sharing-open-source/profiles-and-sessions
[8] https://developers.openai.com/siwc/token-sharing-open-source/models-and-inference
[10] https://developers.openai.com/siwc/token-sharing-open-source/self-hosted-vms
[11] https://developers.openai.com/siwc/token-sharing-open-source/token-reference
[12] https://developers.openai.com/siwc/token-sharing-open-source/errors-and-recovery
[13] https://developers.openai.com/siwc/token-sharing-open-source/preview-limitations
[14] https://developers.openai.com/siwc/ui-ux-guidelines
[15] https://auth.openai.com/.well-known/openid-configuration
[16] https://developers.openai.com/siwc/token-sharing-open-source — OpenAI — SIWC token sharing for open-source projects
[17] https://developers.openai.com/siwc/website — OpenAI — Sign in with ChatGPT on your website
