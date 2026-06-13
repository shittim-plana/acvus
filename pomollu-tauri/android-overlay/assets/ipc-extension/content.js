(function() {
  "use strict";

  var ipcObject = cloneInto({ postMessage: null }, window, { cloneFunctions: true });

  exportFunction(function(message) {
    browser.runtime.sendNativeMessage("ipc", {
      payload: message,
      url: window.location.href
    });
  }, ipcObject, { defineAs: "postMessage" });

  Object.defineProperty(window.wrappedJSObject, "ipc", {
    value: ipcObject,
    writable: false,
    configurable: false,
    enumerable: true
  });
})();
