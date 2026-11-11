document.addEventListener("DOMContentLoaded", () => {
  const cameraVideo = document.getElementById("camera");
  const canvas = document.getElementById("canvas");
  const cameraButton = document.getElementById("camera-button");
  const cameraMessage = document.getElementById("camera-message");
  const resultText = document.getElementById("result");
  const copyButton = document.getElementById("copy-button");
  
  // Find the mascot/image inside camera-panel to hide it when camera starts
  const panelImage = document.querySelector(".camera-panel img");

  if (!canvas || !cameraVideo || !cameraButton) {
    console.error("[CRITICAL] Required DOM elements not found.");
    return;
  }

  const ctx = canvas.getContext("2d");
  let socket = null;
  let streamInterval = null;
  let isStreaming = false;
  let currentLandmarks = [];

  const HAND_CONNECTIONS = [
    [0, 1], [1, 2], [2, 3], [3, 4],
    [0, 5], [5, 6], [6, 7], [7, 8],
    [0, 9], [9, 10], [10, 11], [11, 12],
    [0, 13], [13, 14], [14, 15], [15, 16],
    [0, 17], [17, 18], [18, 19], [19, 20],
    [5, 9], [9, 13], [13, 17]
  ];

  cameraButton.addEventListener("click", async () => {
    if (isStreaming) {
      stopCamera();
      return;
    }

    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: 640, height: 480 },
        audio: false
      });

      cameraVideo.srcObject = stream;
      await cameraVideo.play();

      canvas.width = 640;
      canvas.height = 480;

      // Adjust UI visibility
      if (cameraMessage) cameraMessage.style.display = "none";
      if (panelImage) panelImage.style.display = "none";
      canvas.hidden = false;
      canvas.style.display = "block";
      canvas.style.width = "100%";
      canvas.style.borderRadius = "12px";

      cameraButton.innerText = "Stop Camera";
      isStreaming = true;

      // 1. Start rendering the camera to the canvas immediately
      renderLoop();

      // 2. Connect WebSocket to stream frames for ML inference
      connectWebSocket();
    } catch (err) {
      console.error("[ERROR] Camera initialization failed:", err);
      alert("Unable to access camera: " + err.message);
    }
  });

  // Continuous smooth render loop (runs at full screen refresh rate)
  function renderLoop() {
    if (!isStreaming) return;

    if (cameraVideo.readyState >= cameraVideo.HAVE_CURRENT_DATA) {
      // Draw mirrored camera feed
      ctx.save();
      ctx.scale(-1, 1);
      ctx.drawImage(cameraVideo, -canvas.width, 0, canvas.width, canvas.height);
      ctx.restore();

      // Overlay MediaPipe tracking points if detected
      if (currentLandmarks && currentLandmarks.length > 0) {
        drawSkeleton(currentLandmarks);
      }
    }

    requestAnimationFrame(renderLoop);
  }

  function connectWebSocket() {
    socket = new WebSocket("ws://localhost:8000/ws");

    socket.onopen = () => {
      console.log("[WS] Connected to MudraSetu backend");

      // Stream a compressed frame to ML backend every 66ms (~15 FPS)
      streamInterval = setInterval(() => {
        if (socket.readyState === WebSocket.OPEN && isStreaming) {
          const frameData = canvas.toDataURL("image/jpeg", 0.5);
          socket.send(frameData);
        }
      }, 66);
    };

    socket.onmessage = (event) => {
      const data = JSON.parse(event.data);

      // Store landmarks so the renderLoop paints them
      currentLandmarks = data.landmarks || [];

      // Update recognized sign text
      if (resultText) {
        if (data.detected && data.sign) {
          resultText.innerText = `${data.sign} (${data.confidence}%)`;
        } else {
          resultText.innerText = "Detecting sign...";
        }
      }
    };

    socket.onerror = (err) => {
      console.warn("[WS] Backend connection error. Make sure uvicorn is running on port 8000.");
    };

    socket.onclose = () => {
      console.log("[WS] Connection closed");
      clearInterval(streamInterval);
    };
  }

  function drawSkeleton(landmarks) {
    // Green bone connectors
    ctx.strokeStyle = "#00FF00";
    ctx.lineWidth = 3;
    for (const [startIdx, endIdx] of HAND_CONNECTIONS) {
      const p1 = landmarks[startIdx];
      const p2 = landmarks[endIdx];
      if (p1 && p2) {
        ctx.beginPath();
        ctx.moveTo(p1.x * canvas.width, p1.y * canvas.height);
        ctx.lineTo(p2.x * canvas.width, p2.y * canvas.height);
        ctx.stroke();
      }
    }

    // Green landmark points
    ctx.fillStyle = "#00FF00";
    for (const pt of landmarks) {
      ctx.beginPath();
      ctx.arc(pt.x * canvas.width, pt.y * canvas.height, 5, 0, 2 * Math.PI);
      ctx.fill();
    }
  }

  function stopCamera() {
    if (cameraVideo.srcObject) {
      cameraVideo.srcObject.getTracks().forEach((track) => track.stop());
    }
    clearInterval(streamInterval);
    if (socket) {
      socket.close();
    }

    currentLandmarks = [];
    canvas.hidden = true;
    canvas.style.display = "none";
    if (cameraMessage) cameraMessage.style.display = "block";
    if (panelImage) panelImage.style.display = "block";

    cameraButton.innerText = "Start Camera";
    if (resultText) resultText.innerText = "Your translation will appear here.";
    isStreaming = false;
  }

  if (copyButton) {
    copyButton.addEventListener("click", () => {
      const text = resultText ? resultText.innerText : "";
      if (text && text !== "Your translation will appear here." && text !== "Detecting sign...") {
        navigator.clipboard.writeText(text);
        alert("Copied to clipboard!");
      }
    });
  }
});