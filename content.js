chrome.runtime.onMessage.addListener(
    function(request, sender, sendResponse) {
      if (request.action === "extractText") {
        let text = document.body.innerText;
        sendResponse({text: text});
      }
      return true;  // Will respond asynchronously
    }
  );