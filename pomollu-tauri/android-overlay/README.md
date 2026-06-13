# Android overlay — GeckoView integration

Files here are applied on top of the Tauri-generated Android project
(`tauri android init` → `gen/android/`) by `../build-apk.sh`. Pattern and
files ported from layream (same author), package renamed to
`com.shittimplana.pomollu`.

| File | Role |
|------|------|
| `java/.../MainActivity.kt` | GeckoRuntime + GeckoSession setup, IPC WebExtension install, deep-link OAuth code capture |
| `java/.../AssetServer.kt` | localhost HTTP server (127.0.0.1, random port) serving `assets/` to GeckoView |
| `java/.../OAuthDialog.kt` | dedicated GeckoSession dialog that intercepts the OAuth redirect and returns the `code` |
| `java/.../BrowserPlugin.kt` | open-in-browser / custom tabs / permissions plugin |
| `assets/ipc-extension/` | WebExtension bridge: page `window.ipc.postMessage` → native messaging → `Rust.ipc()` (Tauri invoke transport) |

## Manual one-time steps after `tauri android init`

1. `gen/android/app/build.gradle.kts` — add inside `repositories {}` and
   `dependencies {}`:

   ```kotlin
   repositories {
       maven { url = uri("https://maven.mozilla.org/maven2") }
   }
   dependencies {
       implementation("org.mozilla.geckoview:geckoview-arm64-v8a:128.0.20240725162350")
   }
   ```

2. `gen/android/app/src/main/AndroidManifest.xml` — add a deep-link intent
   filter per OAuth redirect scheme (reverse client ID). Vertex and GCA use
   different OAuth clients, so register both schemes:

   ```xml
   <!-- Vertex AI -->
   <intent-filter>
       <action android:name="android.intent.action.VIEW" />
       <category android:name="android.intent.category.DEFAULT" />
       <category android:name="android.intent.category.BROWSABLE" />
       <data android:scheme="com.googleusercontent.apps.317210024447-v4g6e0e1q5933vogajp0651vhkrgal06" />
   </intent-filter>
   <!-- Gemini Code Assist -->
   <intent-filter>
       <action android:name="android.intent.action.VIEW" />
       <category android:name="android.intent.category.DEFAULT" />
       <category android:name="android.intent.category.BROWSABLE" />
       <data android:scheme="com.googleusercontent.apps.681255809395-oo8ft2oprdrnp9e3aqf6av3hmdib135j" />
   </intent-filter>
   ```

3. ProGuard (release builds) — keep GeckoView classes:

   ```
   -keep class org.mozilla.geckoview.** { *; }
   ```

Caveat: `tauri android init` regenerates `gen/` — never edit `gen/` directly;
edit the overlay and re-run `build-apk.sh`. APK grows ~70–170 MB from the
bundled GeckoView (per-ABI artifact, aarch64 only here).
