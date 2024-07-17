var ws = new WebSocket('ws://127.0.0.1:50007');

function communicate() {
  var txt = document.getElementById('tweet').value;
  ws.send(txt);
  ws.onmessage = function (evt) {
    alert(evt.data);
  };
}
