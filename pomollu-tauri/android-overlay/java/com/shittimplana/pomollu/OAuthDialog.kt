package com.shittimplana.pomollu

import android.app.Dialog
import android.content.Context
import android.graphics.Color
import android.net.Uri
import android.os.Bundle
import android.view.ViewGroup
import android.view.Window
import android.widget.FrameLayout
import android.widget.TextView
import org.mozilla.geckoview.AllowOrDeny
import org.mozilla.geckoview.GeckoResult
import org.mozilla.geckoview.GeckoRuntime
import org.mozilla.geckoview.GeckoSession
import org.mozilla.geckoview.GeckoSessionSettings
import org.mozilla.geckoview.GeckoView

/**
 * A full-screen dialog that opens a Google OAuth consent page inside a dedicated GeckoSession.
 *
 * The dialog intercepts navigation to [redirectUriPrefix] using GeckoSession.NavigationDelegate.
 * When the redirect is detected, the authorization code is extracted from the query string
 * and delivered to [onResult]. The dialog then dismisses itself.
 *
 * Each dialog creates its own GeckoSession but reuses the singleton GeckoRuntime from
 * [MainActivity.getOrCreateRuntime].
 */
class OAuthDialog(
    context: Context,
    private val runtime: GeckoRuntime,
    private val authUrl: String,
    private val redirectUriPrefix: String,
    private val onResult: (code: String?, error: String?) -> Unit,
) : Dialog(context, android.R.style.Theme_Black_NoTitleBar_Fullscreen) {

    private var session: GeckoSession? = null
    private var resolved = false

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        requestWindowFeature(Window.FEATURE_NO_TITLE)
        setCancelable(true)

        val container = FrameLayout(context)

        // --- GeckoView ---
        val geckoView = GeckoView(context)
        val geckoSession = GeckoSession(
            GeckoSessionSettings.Builder()
                .usePrivateMode(true)   // isolate cookies from main session
                .build()
        )

        geckoSession.navigationDelegate = object : GeckoSession.NavigationDelegate {
            override fun onLoadRequest(
                session: GeckoSession,
                request: GeckoSession.NavigationDelegate.LoadRequest
            ): GeckoResult<AllowOrDeny>? {
                val url = request.uri
                if (url.startsWith(redirectUriPrefix)) {
                    val uri = Uri.parse(url)
                    val code = uri.getQueryParameter("code")
                    val error = uri.getQueryParameter("error")
                    resolve(code, error ?: if (code == null) "no code in redirect" else null)
                    dismiss()
                    return GeckoResult.fromValue(AllowOrDeny.DENY)
                }
                return GeckoResult.fromValue(AllowOrDeny.ALLOW)
            }
        }

        geckoSession.progressDelegate = object : GeckoSession.ProgressDelegate {
            // Could add a progress bar here if desired
        }

        geckoSession.open(runtime)
        geckoView.setSession(geckoSession)
        session = geckoSession

        container.addView(geckoView, FrameLayout.LayoutParams(
            FrameLayout.LayoutParams.MATCH_PARENT,
            FrameLayout.LayoutParams.MATCH_PARENT
        ))

        // --- Loading indicator (shown underneath GeckoView initially) ---
        val loadingText = TextView(context).apply {
            text = "Loading..."
            setTextColor(Color.WHITE)
            textSize = 14f
            setPadding(32, 32, 32, 32)
        }
        container.addView(loadingText, 0, FrameLayout.LayoutParams(
            FrameLayout.LayoutParams.WRAP_CONTENT,
            FrameLayout.LayoutParams.WRAP_CONTENT
        ).apply {
            gravity = android.view.Gravity.CENTER
        })

        setContentView(container, ViewGroup.LayoutParams(
            ViewGroup.LayoutParams.MATCH_PARENT,
            ViewGroup.LayoutParams.MATCH_PARENT
        ))

        setOnCancelListener {
            resolve(null, "cancelled")
        }

        // Navigate to OAuth consent page
        geckoSession.loadUri(authUrl)
    }

    private fun resolve(code: String?, error: String?) {
        if (resolved) return
        resolved = true
        onResult(code, error)
    }

    override fun dismiss() {
        try {
            session?.close()
            session = null
        } catch (_: Exception) {}
        if (!resolved) {
            resolve(null, "dismissed")
        }
        super.dismiss()
    }
}
