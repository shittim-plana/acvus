package com.shittimplana.pomollu

import android.content.Intent
import android.os.Bundle
import android.widget.FrameLayout
import androidx.activity.enableEdgeToEdge
import org.mozilla.geckoview.GeckoResult
import org.mozilla.geckoview.GeckoRuntime
import org.mozilla.geckoview.GeckoRuntimeSettings
import org.mozilla.geckoview.GeckoSession
import org.mozilla.geckoview.GeckoSessionSettings
import org.mozilla.geckoview.GeckoView
import org.mozilla.geckoview.WebExtension

class MainActivity : TauriActivity() {
  private var geckoView: GeckoView? = null
  private var geckoSession: GeckoSession? = null
  private var assetServer: AssetServer? = null

  companion object {
    @Volatile
    private var sRuntime: GeckoRuntime? = null

    fun getOrCreateRuntime(activity: MainActivity): GeckoRuntime {
      return sRuntime ?: synchronized(this) {
        sRuntime ?: run {
          val settings = GeckoRuntimeSettings.Builder()
            .javaScriptEnabled(true)
            .consoleOutput(BuildConfig.DEBUG)
            .remoteDebuggingEnabled(BuildConfig.DEBUG)
            .build()
          val rt = GeckoRuntime.create(activity.applicationContext, settings)
          sRuntime = rt
          rt
        }
      }
    }
  }

  override fun onCreate(savedInstanceState: Bundle?) {
    enableEdgeToEdge()
    val oauthUri = if (isOAuthIntent(intent)) intent.data else null
    if (oauthUri != null) intent.data = null
    super.onCreate(savedInstanceState)
    if (oauthUri != null) saveOAuthCode(oauthUri)

    initGeckoView()
  }

  private fun initGeckoView() {
    val server = AssetServer(applicationContext)
    val port = server.start()
    assetServer = server

    val runtime = getOrCreateRuntime(this)
    val session = GeckoSession(
      GeckoSessionSettings.Builder()
        .usePrivateMode(false)
        .build()
    )
    session.open(runtime)
    geckoSession = session

    val view = GeckoView(this)
    view.setSession(session)
    geckoView = view

    val container = FrameLayout(this)
    container.fitsSystemWindows = true
    container.addView(view, FrameLayout.LayoutParams(
      FrameLayout.LayoutParams.MATCH_PARENT,
      FrameLayout.LayoutParams.MATCH_PARENT
    ))
    setContentView(container)

    // Install IPC WebExtension and register native message handler
    runtime.webExtensionController
      .ensureBuiltIn("resource://android/assets/ipc-extension/", "tauri-ipc@pomollu")
      .accept(
        { extension ->
          extension?.setMessageDelegate(
            object : WebExtension.MessageDelegate {
              override fun onMessage(
                nativeApp: String,
                message: Any,
                sender: WebExtension.MessageSender
              ): GeckoResult<Any>? {
                if (nativeApp == "ipc") {
                  @Suppress("UNCHECKED_CAST")
                  val map = message as? Map<String, Any>
                  val payload = map?.get("payload")?.toString()
                  val url = map?.get("url")?.toString() ?: "http://localhost:$port/"
                  payload?.let { Rust.ipc("main", url, it) }
                }
                return null
              }
            },
            "ipc"
          )
          session.loadUri("http://localhost:$port/")
        },
        { throwable ->
          Logger.error("IPC WebExtension install failed: ${throwable?.message}")
          session.loadUri("http://localhost:$port/")
        }
      )
  }

  override fun onNewIntent(intent: Intent) {
    if (isOAuthIntent(intent)) {
      saveOAuthCode(intent.data!!)
    } else {
      super.onNewIntent(intent)
    }
  }

  override fun onPause() {
    super.onPause()
    geckoSession?.setActive(false)
  }

  override fun onResume() {
    super.onResume()
    geckoSession?.setActive(true)
  }

  override fun onDestroy() {
    assetServer?.stop()
    geckoSession?.close()
    super.onDestroy()
  }

  private fun isOAuthIntent(intent: Intent?): Boolean {
    return intent?.data?.scheme?.startsWith("com.googleusercontent.apps.") == true
  }

  private fun saveOAuthCode(uri: android.net.Uri) {
    val scheme = uri.scheme ?: return
    val code = uri.getQueryParameter("code") ?: return
    filesDir.resolve("pending_oauth.txt").writeText("$scheme|$code")
  }
}
