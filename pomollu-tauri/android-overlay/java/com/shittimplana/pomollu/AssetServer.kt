package com.shittimplana.pomollu

import android.content.Context
import java.io.BufferedReader
import java.io.InputStreamReader
import java.net.InetAddress
import java.net.ServerSocket
import java.net.Socket
import java.net.URLDecoder
import java.util.concurrent.Executors

class AssetServer(private val context: Context) {
    var port: Int = 0
        private set

    private var serverSocket: ServerSocket? = null
    @Volatile
    private var running = false

    private val executor = Executors.newFixedThreadPool(4)

    fun start(): Int {
        serverSocket = ServerSocket(0, 50, InetAddress.getByName("127.0.0.1"))
        port = serverSocket!!.localPort
        running = true

        Thread {
            while (running) {
                try {
                    val client = serverSocket?.accept() ?: break
                    executor.submit { handleClient(client) }
                } catch (e: Exception) {
                    if (running) Logger.error("AssetServer accept error: ${e.message}")
                }
            }
        }.start()

        Logger.debug("AssetServer started on localhost:$port")
        return port
    }

    fun stop() {
        running = false
        try { serverSocket?.close() } catch (_: Exception) {}
        executor.shutdownNow()
    }

    private fun handleClient(client: Socket) {
        try {
            val reader = BufferedReader(InputStreamReader(client.getInputStream()))
            val requestLine = reader.readLine() ?: return
            val parts = requestLine.split(" ")
            if (parts.size < 2) return

            var path = URLDecoder.decode(parts[1].split("?")[0], "UTF-8")
            if (path == "/" || path.isEmpty()) path = "/index.html"
            val assetPath = path.removePrefix("/")

            if (assetPath.contains("..")) {
                sendError(client, 403, "Forbidden")
                return
            }

            try {
                val inputStream = context.assets.open(assetPath)
                val data = inputStream.readBytes()
                inputStream.close()

                val mime = getMimeType(assetPath)
                val headers = "HTTP/1.1 200 OK\r\n" +
                    "Content-Type: $mime\r\n" +
                    "Content-Length: ${data.size}\r\n" +
                    "Cache-Control: no-cache\r\n" +
                    "Access-Control-Allow-Origin: *\r\n" +
                    "Connection: close\r\n" +
                    "\r\n"

                val out = client.getOutputStream()
                out.write(headers.toByteArray())
                out.write(data)
                out.flush()
            } catch (e: java.io.FileNotFoundException) {
                sendError(client, 404, "Not Found: $assetPath")
            }
        } catch (e: Exception) {
            Logger.error("AssetServer handle error: ${e.message}")
        } finally {
            try { client.close() } catch (_: Exception) {}
        }
    }

    private fun sendError(client: Socket, code: Int, body: String) {
        val status = if (code == 404) "Not Found" else "Forbidden"
        val response = "HTTP/1.1 $code $status\r\n" +
            "Content-Type: text/plain\r\n" +
            "Content-Length: ${body.length}\r\n" +
            "\r\n$body"
        try {
            client.getOutputStream().write(response.toByteArray())
        } catch (_: Exception) {}
    }

    private fun getMimeType(path: String): String {
        val ext = path.substringAfterLast('.', "").lowercase()
        return when (ext) {
            "html", "htm" -> "text/html; charset=utf-8"
            "css" -> "text/css; charset=utf-8"
            "js", "mjs" -> "application/javascript; charset=utf-8"
            "json" -> "application/json; charset=utf-8"
            "svg" -> "image/svg+xml"
            "png" -> "image/png"
            "jpg", "jpeg" -> "image/jpeg"
            "gif" -> "image/gif"
            "webp" -> "image/webp"
            "ico" -> "image/x-icon"
            "woff" -> "font/woff"
            "woff2" -> "font/woff2"
            "ttf" -> "font/ttf"
            "wasm" -> "application/wasm"
            else -> "application/octet-stream"
        }
    }
}
