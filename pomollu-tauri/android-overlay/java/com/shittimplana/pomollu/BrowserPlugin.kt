package com.shittimplana.pomollu

import android.Manifest
import android.app.Activity
import android.content.Intent
import android.content.pm.PackageManager
import android.net.Uri
import android.os.Build
import android.os.Environment
import android.provider.Settings
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import app.tauri.annotation.Command
import app.tauri.annotation.TauriPlugin
import app.tauri.plugin.Invoke
import app.tauri.plugin.JSObject
import app.tauri.plugin.Plugin
import androidx.browser.customtabs.CustomTabsIntent
import org.json.JSONArray
import org.json.JSONObject

@TauriPlugin
class BrowserPlugin(private val activity: Activity) : Plugin(activity) {

    @Command
    fun listBrowsers(invoke: Invoke) {
        try {
            val probe = Intent(Intent.ACTION_VIEW, Uri.parse("https://example.com"))
            probe.addCategory(Intent.CATEGORY_BROWSABLE)
            val apps = activity.packageManager
                .queryIntentActivities(probe, PackageManager.MATCH_ALL)
                .filter { it.activityInfo.packageName != activity.packageName }

            val arr = JSONArray()
            for (ri in apps) {
                val obj = JSONObject()
                obj.put("package", ri.activityInfo.packageName)
                obj.put("label", ri.loadLabel(activity.packageManager).toString())
                arr.put(obj)
            }
            val result = JSObject()
            result.put("browsers", arr)
            invoke.resolve(result)
        } catch (ex: Exception) {
            invoke.reject(ex.message)
        }
    }

    @Command
    fun openBrowser(invoke: Invoke) {
        try {
            val url = invoke.parseArgs(String::class.java)
            val intent = Intent(Intent.ACTION_VIEW, Uri.parse(url))
            intent.addCategory(Intent.CATEGORY_BROWSABLE)
            intent.addFlags(Intent.FLAG_ACTIVITY_NEW_TASK)
            val chooser = Intent.createChooser(intent, "브라우저 선택")
            activity.startActivity(chooser)
            invoke.resolve()
        } catch (ex: Exception) {
            invoke.reject(ex.message)
        }
    }

    @Command
    fun requestStoragePermission(invoke: Invoke) {
        try {
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.R) {
                if (!Environment.isExternalStorageManager()) {
                    val intent = Intent(Settings.ACTION_MANAGE_APP_ALL_FILES_ACCESS_PERMISSION)
                    intent.data = Uri.parse("package:${activity.packageName}")
                    activity.startActivity(intent)
                }
            }
            val result = JSObject()
            result.put("granted", Build.VERSION.SDK_INT < Build.VERSION_CODES.R || Environment.isExternalStorageManager())
            invoke.resolve(result)
        } catch (ex: Exception) {
            invoke.reject(ex.message)
        }
    }

    @Command
    fun openInBrowser(invoke: Invoke) {
        try {
            val combined = invoke.parseArgs(String::class.java)
            val sep = combined.indexOf("|")
            val url: String
            val pkg: String
            if (sep >= 0) {
                pkg = combined.substring(0, sep)
                url = combined.substring(sep + 1)
            } else {
                url = combined
                pkg = ""
            }
            val intent = Intent(Intent.ACTION_VIEW, Uri.parse(url))
            intent.addCategory(Intent.CATEGORY_BROWSABLE)
            if (pkg.isNotEmpty()) {
                intent.setPackage(pkg)
            }
            intent.addFlags(Intent.FLAG_ACTIVITY_NEW_TASK)
            activity.startActivity(intent)
            invoke.resolve()
        } catch (ex: Exception) {
            invoke.reject(ex.message)
        }
    }

    @Command
    fun openCustomTab(invoke: Invoke) {
        try {
            val url = invoke.parseArgs(String::class.java)
            val customTabsIntent = CustomTabsIntent.Builder()
                .build()
            customTabsIntent.launchUrl(activity, Uri.parse(url))
            invoke.resolve()
        } catch (ex: Exception) {
            invoke.reject(ex.message)
        }
    }

    @Command
    fun requestNotificationPermission(invoke: Invoke) {
        try {
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
                if (ContextCompat.checkSelfPermission(activity, Manifest.permission.POST_NOTIFICATIONS)
                    != PackageManager.PERMISSION_GRANTED) {
                    ActivityCompat.requestPermissions(
                        activity,
                        arrayOf(Manifest.permission.POST_NOTIFICATIONS),
                        2001
                    )
                }
            }
            val result = JSObject()
            result.put("granted", Build.VERSION.SDK_INT < Build.VERSION_CODES.TIRAMISU ||
                ContextCompat.checkSelfPermission(activity, Manifest.permission.POST_NOTIFICATIONS) == PackageManager.PERMISSION_GRANTED)
            invoke.resolve(result)
        } catch (ex: Exception) {
            invoke.reject(ex.message)
        }
    }

    @Command
    fun openGeckoViewOAuth(invoke: Invoke) {
        try {
            val combined = invoke.parseArgs(String::class.java)
            // Format: "authUrl|redirectUriPrefix"
            val sep = combined.indexOf("|")
            if (sep < 0) {
                invoke.reject("Expected format: authUrl|redirectUriPrefix")
                return
            }
            val authUrl = combined.substring(0, sep)
            val redirectUriPrefix = combined.substring(sep + 1)

            val mainActivity = activity as? MainActivity
            if (mainActivity == null) {
                invoke.reject("Activity is not MainActivity — cannot access GeckoRuntime")
                return
            }
            val runtime = MainActivity.getOrCreateRuntime(mainActivity)

            activity.runOnUiThread {
                try {
                    val dialog = OAuthDialog(
                        context = activity,
                        runtime = runtime,
                        authUrl = authUrl,
                        redirectUriPrefix = redirectUriPrefix,
                        onResult = { code, error ->
                            val result = JSObject()
                            if (code != null) {
                                result.put("code", code)
                            }
                            if (error != null) {
                                result.put("error", error)
                            }
                            invoke.resolve(result)
                        }
                    )
                    dialog.show()
                } catch (ex: Exception) {
                    invoke.reject("Failed to open OAuth dialog: ${ex.message}")
                }
            }
        } catch (ex: Exception) {
            invoke.reject(ex.message)
        }
    }

    @Command
    fun getPendingOAuth(invoke: Invoke) {
        try {
            val file = activity.filesDir.resolve("pending_oauth.txt")
            if (file.exists()) {
                val content = file.readText()
                file.delete()
                val sep = content.indexOf("|")
                if (sep >= 0) {
                    val result = JSObject()
                    result.put("scheme", content.substring(0, sep))
                    result.put("code", content.substring(sep + 1))
                    invoke.resolve(result)
                    return
                }
            }
            invoke.resolve(JSObject())
        } catch (ex: Exception) {
            invoke.resolve(JSObject())
        }
    }
}
