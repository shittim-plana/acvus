# pomollu-tauri — 미완 / 잔여 (RESIDUALS)

구현됐지만 **통합·검증이 안 끝난** 것들을 명시적으로 모은 목록.
(README의 설계 노트가 산문으로 흩어져 있어 여기 한곳에 박제한다.)

상태 범례: `DONE` 끝남 / `IMPL` 구현됐으나 미통합 / `TODO` 미착수 / `UNVERIFIED` 검증 안 됨

---

## 저장 시스템 (FsBlobStore)

- `IMPL` `pomollu-core/src/store.rs` — content-addressed blobs + append-only
  journal(parent DAG) + CAS pointer + 멀티윈도(cas_lock, commit, branch-on-conflict).
  30 테스트 통과. 구현 자체는 끝.
- `TODO` **세션/워크스페이스를 store 위로 마이그레이션.** 현재 `persistence.rs`는
  mutable JSON(tmp+rename, last-write-wins). store로 올리면 history·undo·분기가
  충돌이 아니라 합류. 엔진 부활 시 정합.
- `TODO` **엔진 `BlobStore` trait 정합.** 현재 FsBlobStore는 독립 API
  (put/get/ref_cas/journal/commit). acvus 리팩토링이 BlobStore trait를 재도입하면
  이 primitives 위에 그 trait를 구현.
- `TODO` **Claim 2 (다기기 sync).** blob layer(G-Set)는 공짜로 merge되고
  pointer에만 commutativity 추가하면 됨. migration path만 문서화, 미구현.
- ★ **재사용 가능성**: 이 store는 acvus엔 없던 새 능력. **layream도 같은 mutable
  JSON 문제**가 있어 layream-core로 포팅하면 동일 이득. layream 설계 문서
  (`layream/docs/risu-fidelity-and-converter.md`)의 해당 항목 참조.

## 엔진 연결

- `IMPL` `pomollu-core/src/fetch.rs::ReqwestFetch` — `acvus_ext_llm::Fetch` 네이티브
  구현 (엔진 훅 준비됨).
- `IMPL` acvus-ext-llm `mistral_registry` / `vertex_registry` — 엔진이 LLM 부르는
  통로. 단 **speculative** — 동작하는 엔진이 없음.
- `TODO` `acvus-orchestration::Session`의 턴 실행이 `TurnResult { /* TODO */ }`.
  실행이 살아나면 lib.rs에서 registry 등록만 하면 엔진이 네이티브로 LLM 호출.
- 참고: `pomollu-core/src/providers/`의 스트리밍 클라이언트(mistral/vertex/gca)는
  엔진이 살면 registry와 **중복 가능** — 엔진 없는 동안의 직접 채팅용. 정리 대상.

## 실행/전송 (선택 설계)

- `TODO` **axum 로컬 서버** — 현재 GeckoView WebExtension IPC(`window.ipc`) +
  버퍼+폴링 스트리밍. axum을 127.0.0.1에 띄우면 SSE 직접 + Kotlin AssetServer/
  브리지 제거 가능. 별도 PR 규모. (README 설계 노트 참조)

## Android / 빌드 검증

- `IMPL` `android-overlay/` — GeckoView 통합 파일(MainActivity, AssetServer,
  OAuthDialog, BrowserPlugin, ipc-extension) + `build-apk.sh`. 코드만 있음.
- `TODO` **APK 실기기 빌드/검증** — 이 환경에 Android SDK 없음. `tauri android init`
  + `android-overlay/README.md`의 수동 단계(gradle geckoview dep, manifest deep-link)
  적용 후 `build-apk.sh` 필요.
- `UNVERIFIED` **Tauri 레이어 Android 컴파일** — pomollu-core는 30테스트 통과(DONE).
  Tauri 커맨드 레이어(GCA + stream_id + generate_handler! 최종본)의
  `cargo check --target aarch64-linux-android`는 **이 박스에서 검증 불가**.
  원인: ring/blake3의 cc-rs `detect_compiler_family.c` 프로브가 커스텀 NDK r30
  clang에서 hang (반복 재현 — 죽인 run의 좀비가 16시간 잔존하는 것으로 확인).
  → **정상 Android 툴체인 서버에서 `cargo check --target aarch64-linux-android`로
  확인 필요.** 코드상 risk는 낮음(커맨드 시그니처 ↔ generate_handler! 정합은
  pomollu-core 30테스트와 별개로 눈으로 확인됨)지만 컴파일은 미증명.

## 버전 핀 (빌드 주의)

- `pomollu-tauri/Cargo.lock`: `time=0.3.44`, `serde_with=3.12.0`, `plist=1.7.4`.
  time 0.3.47+가 tauri-utils 2.9.2와 E0119 코히어런스 충돌. `cargo update` 시 주의.
- 이 박스는 C 컴파일러 없어 zig cc 우회 + Android는 커스텀 NDK r30. 정상 툴체인
  서버에선 불필요.
