# pomollu-tauri

Pomollu의 로컬 모바일 앱 (Tauri 2 + GeckoView, Android). layream의 검증된
로컬 기능을 acvus 워크스페이스에 정식 이식한 것.

## 구조

```
pomollu-tauri/
├── pomollu-core/        # Tauri 의존 없는 핵심 (호스트에서 테스트 가능)
│   └── src/
│       ├── oauth.rs         # Vertex AI OAuth (PKCE) + 토큰 리프레시 + GCP 프로젝트 목록
│       ├── crypto.rs        # AES-256-GCM 토큰 저장소 암호화 (랜덤 논스, 레거시 폴백 없음)
│       ├── persistence.rs   # 원자적 JSON 쓰기(tmp+rename), settings/session/workspaces
│       ├── retry.rs         # 취소 토큰 + 지수 백오프 재시도 (429/5xx, Retry-After)
│       ├── providers/       # 스트리밍 클라이언트 (UI 경로)
│       │   ├── mistral.rs   #   SSE 스트리밍 + capability 필터된 모델 목록
│       │   └── vertex.rs    #   streamGenerateContent SSE + thinking config
│       └── fetch.rs         # ReqwestFetch: acvus_ext_llm::Fetch 네이티브 구현 (엔진 훅)
├── src/                 # Tauri 레이어: 커맨드 + 상태
│   ├── commands_auth.rs     # vertex_oauth_* (PKCE 시작/콜백/상태/해제), 딥링크 핸드오프
│   ├── commands_chat.rs     # chat_mistral / chat_vertex / chat_gca, stream_id별 버퍼+폴링·취소
│   ├── commands_settings.rs # 설정/세션 저장·로드
│   └── commands_workspace.rs# 워크스페이스 CRUD (디렉터리 격리)
├── android-overlay/     # GeckoView 통합 (tauri android init 위에 덮어씀)
└── build-apk.sh         # 프론트 빌드 → 오버레이 적용 → APK
```

같은 프로바이더가 두 층에 존재하는 이유:
- **acvus-ext-llm** `mistral_registry` / `vertex_registry` — 엔진(acvus 언어
  런타임)용 non-streaming ExternFn. `pomollu_core::fetch::ReqwestFetch`로
  네이티브 transport가 이미 준비돼 있음.
- **pomollu-core providers** — 채팅 UI용 스트리밍·취소 클라이언트.

엔진 연결 잔여 조건: acvus-orchestration `Session`의 턴 실행이 아직
`TurnResult { /* TODO */ }` 상태. 실행이 구현되면 lib.rs에서 registry 등록만
하면 된다.

## LLM 프로바이더

| Provider | 인증 | 엔진 경로 (ext-llm) | UI 경로 (streaming) |
|----------|------|--------------------|---------------------|
| Mistral | API key | `mistral_chat` | `chat_mistral` |
| Vertex AI | OAuth PKCE (자동 리프레시) | `vertex_llm` | `chat_vertex` |
| Gemini Code Assist (GCA) | OAuth client_secret (무료 티어) | (미노출) | `chat_gca` |
| OpenAI 호환 / Anthropic / Gemini | API key | 기존 registry | (미노출) |

Vertex OAuth 클라이언트는 settings의 `oauthClientId`/`oauthRedirectUri`로
오버라이드 가능 (기본값은 동일 저자의 GCP 클라이언트).

GCA(`cloudcode-pa.googleapis.com/v1internal`)는 Gemini를 무료 티어로 호출하는
비공식 경로다. Google의 공개 installed-app 클라이언트를 쓰며 — `client_secret`이
OAuth 의미의 비밀이 아님 — Vertex와 동일한 Gemini 요청/응답 스키마를
`{model, request, project}`로 래핑한다. 비용: `v1internal`은 비공식 API라
예고 없이 바뀔 수 있음. `chat_gca`는 Vertex와 같은 Gemini 타입(thinking config
포함)을 공유하고, 토큰은 같은 암호화 저장소의 `gca` 필드에 들어간다
(`#[serde(default)]`로 기존 vertex-only `tokens.json` 전방 호환).

## 빌드 (Android)

```sh
rustup target add aarch64-linux-android
# NDK: aarch64 리눅스 호스트는 HomuHomu833/android-ndk-custom r30 사용 가능
tauri android init       # 1회 — gen/ 생성
# android-overlay/README.md 의 1회 수동 단계 (gradle, manifest) 적용
./build-apk.sh           # 프론트 빌드 + 오버레이 + APK
```

## 검증 상태 (이 환경에서 관찰된 범위)

- `cargo test -p acvus-ext-llm` — 30 passed (mistral/vertex registry 포함)
- `cargo test -p pomollu-core` — crypto/oauth/persistence/providers 단위 테스트
- `cargo check --target aarch64-linux-android` — Tauri 레이어 타입 체크
- APK 실기기 검증은 미수행 (이 환경에 Android SDK 없음) — residual

알려진 핀: `time = 0.3.44`, `serde_with = 3.12`, `plist = 1.7` (time 0.3.47+가
tauri-utils 2.9.2 블랭킷 impl과 E0119 코히어런스 충돌).

## 멀티 스트림 (acvus 멀티세션 구조 정합)

채팅 스트림은 호출자가 주는 `stream_id`로 격리된다 (`StreamBufferState` /
`StreamCancelState`가 `HashMap<stream_id, _>`). 워크스페이스마다, 혹은
Vertex·GCA·Mistral을 동시에 — 서로의 버퍼나 취소 토큰을 건드리지 않고 병렬
실행된다. 프론트는 `startChat()`이 `stream_id`를 반환하므로 그 id로 개별
취소(`cancelChat(streamId)`)·폴링(`pollStreamChunks(streamId)`)이 가능하다.
완료 시 취소 토큰을 제거하고, 마지막 폴링이 빈 버퍼 엔트리를 회수한다 (누수 없음).

프로바이더별 모델 카탈로그는 **소스가 다르다** (공유 집합 아님): Mistral은
라이브 `/v1/models`(chat 필터+최신 dedup), Vertex는 리전별 라이브
`publishers/google/models`, GCA는 정적 카탈로그(`v1internal`에 목록 엔드포인트
없음). 활성 프로바이더의 목록을 조회해야 하고, 한쪽 모델이 다른 쪽에 있다고
가정하면 안 된다.

## 설계 메모 — 검토 중인 다음 단계

### axum 로컬 서버 (검토)

현재 IPC는 GeckoView WebExtension 브리지(`window.ipc` → native messaging →
`Rust.ipc`)이고 스트리밍 응답은 버퍼+폴링이다. 대안: Rust 백엔드에서 axum을
127.0.0.1 랜덤 포트에 띄워 (a) 정적 자산, (b) REST 커맨드, (c) SSE 스트리밍을
직접 서빙하면 Kotlin AssetServer와 WebExtension 브리지가 모두 사라진다.

- 장점: 폴링 제거(진짜 SSE), Kotlin 글루 최소화, 데스크톱/모바일 동일 경로
- 비용: Tauri invoke 보안 모델(capabilities) 우회 — localhost 포트는 같은
  기기의 타 앱에서 접근 가능하므로 토큰 기반 요청 인증 필요
- 판단: 구조적으로 우월하나 별도 PR 규모. 현재 폴링 구현은 layream에서
  동작이 입증된 경로라 먼저 출하 가능.

### 저장 시스템 — acvus 설계를 로컬로 (`pomollu-core/store.rs`, 구현됨)

엔진 리팩토링이 저장소를 **content-addressed + append-only + 메타데이터
compare-and-swap (+ CRDT)** 위에 다시 정초하는 중이다 (구 `IdbBlobStore`와
동형). 그 IndexedDB 버전과 forward-compatible한 파일 기반 `FsBlobStore`를
구현했다:

```
store/
├── blobs/{sha256}   content-addressed, immutable — G-Set (join-semilattice)
├── journal          append-only entry DAG (parent 링크 → undo/branch/goto)
└── refs/{name}      CAS 포인터 (cursor / HEAD)
```

설계 근거 (CLAUDE.md Full Example 구조):
- **Claim 1 (단일 기기, `P(contention)≪1`)** — blob layer는 무조건 수렴하는
  CRDT, ref layer는 single-writer 하에 linearizable한 CAS. 모바일은 단일
  기기이므로 이걸로 충분. → 지금 구현한 범위.
- **Claim 2 (다기기 sync, `P(contention)→1`)** — blob layer(G-Set)는 그대로
  공짜로 merge되고, pointer layer에만 commutativity를 추가하면 된다 (migration
  path, 미구현).

caveat: SHA-256 collision resistance는 cryptographic 가정. `ref_cas`는
single-writer 가정 하에서만 linearizable (다중 프로세스면 file lock 필요 —
앱이 유일 writer).

세션/워크스페이스를 이 store 위로 올리면 (현재는 mutable JSON) 엔진 부활 시
히스토리·undo·분기가 충돌이 아니라 합류한다. 엔진의 BlobStore trait가 확정되면
이 primitives(put/get/ref_cas/journal) 위에 그 trait를 구현하면 된다.

### RisuAI 포맷 + CBS 포팅 (검토)

layream-core에 이미 Rust 구현이 있다: `.risup`/`.risum`(rpack+msgpack),
`.charx`(zip), CBS(LALRPOP 파서+evaluator+highlighter, ~2.6k LOC).

- 1단계 (기계적): `pomollu-risu` 크레이트로 모듈 복사 — preset.rs, charx.rs,
  rpack.rs, crypto(risup 복호화에 레거시 제로-논스 경로 필요), cbs/ 전체 +
  lalrpop build dep. 커맨드 4개 노출 (load_preset, load_character,
  evaluate_cbs, highlight_cbs).
- 2단계 (의미적, 미해결): CBS 템플릿 → acvus 템플릿 변환기. CBS의
  `{{char}}`/조건/랜덤 블록을 acvus 노드 스펙으로 사상. 양쪽 의미론이
  1:1이 아니라서 (CBS는 문자열 치환, acvus는 타입 있는 템플릿) lossy 변환
  경계를 명시해야 함 — 설계 후 별도 작업.
