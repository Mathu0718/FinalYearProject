// ── DOM References ──
const dropZone = document.getElementById('drop-zone');
const fileInput = document.getElementById('file-input');
const previewContainer = document.getElementById('preview-container');
const imagePreview = document.getElementById('image-preview');
const videoPreview = document.getElementById('video-preview');
const clearBtn = document.getElementById('clear-btn');
const analyzeBtn = document.getElementById('analyze-btn');

const mainInterface = document.getElementById('main-interface');
const resultsDashboard = document.getElementById('results-dashboard');
const analyzeAnotherBtn = document.getElementById('analyze-another-btn');
const modelTypeSelect = document.getElementById('model-type-select');

const resultImage = document.getElementById('result-image');
const resultVideo = document.getElementById('result-video');
const loadingOverlay = document.getElementById('loading');
const detectionsList = document.getElementById('detections-list');
const detectionCount = document.getElementById('detection-count');

const modal = document.getElementById('image-modal');
const modalImg = document.getElementById('modal-img');
const closeModalBtn = document.getElementById('close-modal');

const aiMessages = document.getElementById('ai-messages');
const copilotStatus = document.querySelector('.status-text');
const copilotDot = document.querySelector('.dot');

// Advisory elements
const advisoryBanner = document.getElementById('advisory-banner');
const advisoryText = document.getElementById('advisory-text');
const advisoryTags = document.getElementById('advisory-tags');
const speakAdvisoryBtn = document.getElementById('speak-advisory-btn');

// Weather elements
const weatherStrip = document.getElementById('weather-strip');
const weatherLoading = document.getElementById('weather-loading');
const weatherData = document.getElementById('weather-data');

let currentFile = null;
let currentFileType = null;
let advisoriesMap = {};
let currentWeather = null;
let userLat = 13.0827;  // Default: Chennai
let userLon = 80.2707;

// ── ROI Drawing State ──
const roiCanvas = document.getElementById('roi-canvas');
const roiCtx = roiCanvas ? roiCanvas.getContext('2d') : null;
const roiInstruction = document.getElementById('roi-instruction');
const roiSkipBtn = document.getElementById('roi-skip-btn');
let roiDrawing = false;
let roiStart = { x: 0, y: 0 };
let roiEnd = { x: 0, y: 0 };
let roiRect = null;  // { x, y, w, h } in canvas-pixel coords
let roiReady = false;  // true once a box has been drawn
let roiMode = false;   // true when we're in ROI drawing mode

// Speedometer
const speedCanvas = document.getElementById('speedometer-canvas');
const speedCtx = speedCanvas.getContext('2d');
const speedValueEl = document.getElementById('speed-value');
const speedStatusEl = document.getElementById('speed-status');
let currentSpeed = 60;  // default cruising speed
let targetSpeed = 60;
let speedAnimFrame = null;

// ── Initialization ──
(async function initSystem() {
    // Load advisories
    try {
        const response = await fetch('/advisories');
        if (response.ok) {
            advisoriesMap = await response.json();
            console.log("[System] Advisory database loaded.");
        }
    } catch (e) {
        console.warn("[System] Using fallback advisories.");
    }

    // Get user location
    if (navigator.geolocation) {
        navigator.geolocation.getCurrentPosition(
            (pos) => {
                userLat = pos.coords.latitude;
                userLon = pos.coords.longitude;
                console.log(`[GPS] Location: ${userLat}, ${userLon}`);
                fetchWeather();
            },
            () => {
                console.warn("[GPS] Permission denied. Using default location.");
                fetchWeather();
            },
            { timeout: 5000 }
        );
    } else {
        fetchWeather();
    }
})();

// ── Weather Functions ──
async function fetchWeather() {
    try {
        const response = await fetch(`/weather?lat=${userLat}&lon=${userLon}`);
        if (response.ok) {
            currentWeather = await response.json();
            if (currentWeather.status === 'ok') {
                renderWeather(currentWeather);
                addBotMessage(`Weather Report: ${currentWeather.description}, ${currentWeather.temp}°C in ${currentWeather.city}`);
            }
        }
    } catch (e) {
        console.warn("[Weather] API unavailable:", e);
    }
}

function renderWeather(w) {
    weatherLoading.classList.add('hidden');
    weatherData.classList.remove('hidden');

    document.getElementById('weather-temp').textContent = `${w.temp}°C`;
    document.getElementById('weather-desc').textContent = w.description;
    document.getElementById('weather-humidity').textContent = `${w.humidity}%`;
    document.getElementById('weather-visibility').textContent = `${(w.visibility / 1000).toFixed(1)}km`;
    document.getElementById('weather-wind').textContent = `${w.wind_speed}m/s`;
    document.getElementById('weather-city').textContent = w.city;
    document.getElementById('weather-icon').src = `https://openweathermap.org/img/wn/${w.icon}@2x.png`;
}

// ── Modal Logic ──
if (closeModalBtn) {
    closeModalBtn.addEventListener('click', () => modal.classList.add('hidden'));
}
window.openModal = function(imgSrc) {
    modal.classList.remove('hidden');
    modalImg.src = imgSrc;
};

// ── Drag & Drop ──
dropZone.addEventListener('dragover', (e) => {
    e.preventDefault();
    dropZone.classList.add('dragover');
});
dropZone.addEventListener('dragleave', () => dropZone.classList.remove('dragover'));
dropZone.addEventListener('drop', (e) => {
    e.preventDefault();
    dropZone.classList.remove('dragover');
    if (e.dataTransfer.files && e.dataTransfer.files[0]) handleFile(e.dataTransfer.files[0]);
});

dropZone.addEventListener('click', () => fileInput.click());

fileInput.addEventListener('change', (e) => {
    if (e.target.files && e.target.files[0]) handleFile(e.target.files[0]);
});

function handleFile(file) {
    if (file.type.startsWith('image/')) currentFileType = 'image';
    else if (file.type.startsWith('video/')) currentFileType = 'video';
    else { alert('Unsupported format.'); return; }

    currentFile = file;
    const url = URL.createObjectURL(file);

    imagePreview.classList.add('hidden');
    videoPreview.classList.add('hidden');

    if (currentFileType === 'image') {
        imagePreview.src = url;
        imagePreview.classList.remove('hidden');
        // Wait for image to load, then init ROI canvas
        imagePreview.onload = () => initROICanvas(imagePreview);
    } else {
        videoPreview.src = url;
        videoPreview.classList.remove('hidden');
        // Wait for video metadata to get dimensions
        videoPreview.onloadeddata = () => {
            // Pause at first frame for drawing
            videoPreview.pause();
            videoPreview.currentTime = 0;
            initROICanvas(videoPreview);
        };
    }

    dropZone.classList.add('hidden');
    previewContainer.classList.remove('hidden');

    addBotMessage(`Stream loaded: ${file.name}. Draw an ROI box on the preview, then press Space to start analysis.`);
}

// ── ROI Canvas Initialization ──
function initROICanvas(mediaEl) {
    if (!roiCanvas || !roiCtx) return;

    roiMode = true;
    roiReady = false;
    roiRect = null;

    // Match canvas to the rendered media size
    const rect = mediaEl.getBoundingClientRect();
    roiCanvas.width = rect.width;
    roiCanvas.height = rect.height;
    roiCanvas.style.width = rect.width + 'px';
    roiCanvas.style.height = rect.height + 'px';
    roiCanvas.classList.remove('hidden');

    // Show instruction banner
    if (roiInstruction) {
        roiInstruction.classList.remove('roi-ready-banner');
        roiInstruction.querySelector('.flex-1').innerHTML = `
            <strong class="text-white text-sm">Draw Detection Region</strong><br>
            Click and drag on the preview to draw a bounding box for AI detection zone.
            Then press <span class="space-key-hint">⎵ Space</span> to start analysis.
        `;
    }

    // Clear any previous drawing
    roiCtx.clearRect(0, 0, roiCanvas.width, roiCanvas.height);

    // Draw crosshair guides
    drawROIGuides();
}

function drawROIGuides() {
    if (!roiCtx) return;
    const w = roiCanvas.width;
    const h = roiCanvas.height;
    roiCtx.clearRect(0, 0, w, h);

    // Rule-of-thirds grid
    roiCtx.strokeStyle = 'rgba(59, 130, 246, 0.15)';
    roiCtx.lineWidth = 1;
    roiCtx.setLineDash([4, 4]);
    for (let i = 1; i <= 2; i++) {
        // Vertical
        roiCtx.beginPath();
        roiCtx.moveTo((w / 3) * i, 0);
        roiCtx.lineTo((w / 3) * i, h);
        roiCtx.stroke();
        // Horizontal
        roiCtx.beginPath();
        roiCtx.moveTo(0, (h / 3) * i);
        roiCtx.lineTo(w, (h / 3) * i);
        roiCtx.stroke();
    }
    roiCtx.setLineDash([]);

    // If ROI is drawn, redraw it
    if (roiRect) {
        drawROIBox(roiRect.x, roiRect.y, roiRect.w, roiRect.h);
    }
}

function drawROIBox(x, y, w, h) {
    // Dim area outside ROI
    roiCtx.fillStyle = 'rgba(0, 0, 0, 0.5)';
    // Top strip
    roiCtx.fillRect(0, 0, roiCanvas.width, y);
    // Bottom strip
    roiCtx.fillRect(0, y + h, roiCanvas.width, roiCanvas.height - y - h);
    // Left strip
    roiCtx.fillRect(0, y, x, h);
    // Right strip
    roiCtx.fillRect(x + w, y, roiCanvas.width - x - w, h);

    // ROI border with animated dashes
    roiCtx.strokeStyle = '#3b82f6';
    roiCtx.lineWidth = 2;
    roiCtx.setLineDash([6, 3]);
    roiCtx.strokeRect(x, y, w, h);
    roiCtx.setLineDash([]);

    // Corner markers
    const cornerLen = 12;
    roiCtx.strokeStyle = '#60a5fa';
    roiCtx.lineWidth = 3;
    // Top-left
    roiCtx.beginPath(); roiCtx.moveTo(x, y + cornerLen); roiCtx.lineTo(x, y); roiCtx.lineTo(x + cornerLen, y); roiCtx.stroke();
    // Top-right
    roiCtx.beginPath(); roiCtx.moveTo(x + w - cornerLen, y); roiCtx.lineTo(x + w, y); roiCtx.lineTo(x + w, y + cornerLen); roiCtx.stroke();
    // Bottom-left
    roiCtx.beginPath(); roiCtx.moveTo(x, y + h - cornerLen); roiCtx.lineTo(x, y + h); roiCtx.lineTo(x + cornerLen, y + h); roiCtx.stroke();
    // Bottom-right
    roiCtx.beginPath(); roiCtx.moveTo(x + w - cornerLen, y + h); roiCtx.lineTo(x + w, y + h); roiCtx.lineTo(x + w, y + h - cornerLen); roiCtx.stroke();

    // Label
    roiCtx.fillStyle = 'rgba(59, 130, 246, 0.85)';
    roiCtx.font = 'bold 11px JetBrains Mono, monospace';
    roiCtx.fillText(`ROI: ${Math.abs(Math.round(w))}×${Math.abs(Math.round(h))}px`, x + 4, y - 6);
}

// ── ROI Mouse Events ──
if (roiCanvas) {
    roiCanvas.addEventListener('mousedown', (e) => {
        if (!roiMode) return;
        const rect = roiCanvas.getBoundingClientRect();
        roiStart = { x: e.clientX - rect.left, y: e.clientY - rect.top };
        roiDrawing = true;
        roiReady = false;
        roiRect = null;
    });

    roiCanvas.addEventListener('mousemove', (e) => {
        if (!roiDrawing || !roiMode) return;
        const rect = roiCanvas.getBoundingClientRect();
        roiEnd = { x: e.clientX - rect.left, y: e.clientY - rect.top };

        // Clamp to canvas bounds
        roiEnd.x = Math.max(0, Math.min(roiCanvas.width, roiEnd.x));
        roiEnd.y = Math.max(0, Math.min(roiCanvas.height, roiEnd.y));

        const x = Math.min(roiStart.x, roiEnd.x);
        const y = Math.min(roiStart.y, roiEnd.y);
        const w = Math.abs(roiEnd.x - roiStart.x);
        const h = Math.abs(roiEnd.y - roiStart.y);

        roiCtx.clearRect(0, 0, roiCanvas.width, roiCanvas.height);
        if (w > 5 && h > 5) {
            drawROIBox(x, y, w, h);
        }
    });

    roiCanvas.addEventListener('mouseup', (e) => {
        if (!roiDrawing || !roiMode) return;
        roiDrawing = false;

        const rect = roiCanvas.getBoundingClientRect();
        roiEnd = { x: e.clientX - rect.left, y: e.clientY - rect.top };
        roiEnd.x = Math.max(0, Math.min(roiCanvas.width, roiEnd.x));
        roiEnd.y = Math.max(0, Math.min(roiCanvas.height, roiEnd.y));

        const x = Math.min(roiStart.x, roiEnd.x);
        const y = Math.min(roiStart.y, roiEnd.y);
        const w = Math.abs(roiEnd.x - roiStart.x);
        const h = Math.abs(roiEnd.y - roiStart.y);

        if (w > 10 && h > 10) {
            roiRect = { x, y, w, h };
            roiReady = true;

            // Update instruction banner to green "ready" state
            if (roiInstruction) {
                roiInstruction.classList.add('roi-ready-banner');
                const wPct = ((w / roiCanvas.width) * 100).toFixed(0);
                const hPct = ((h / roiCanvas.height) * 100).toFixed(0);
                roiInstruction.querySelector('.flex-1').innerHTML = `
                    <strong class="text-white text-sm">✓ Detection Region Set</strong>
                    <span class="roi-coords-badge">${Math.round(w)}×${Math.round(h)}px (${wPct}%×${hPct}%)</span><br>
                    Press <span class="space-key-hint">⎵ Space</span> to start analysis, or redraw to adjust.
                `;
            }

            addBotMessage(`ROI region set: ${Math.round(w)}×${Math.round(h)}px. Press Space to begin analysis.`);
        } else {
            // Too small, reset
            roiRect = null;
            roiReady = false;
            roiCtx.clearRect(0, 0, roiCanvas.width, roiCanvas.height);
            drawROIGuides();
        }
    });
}

// ── Space Key Listener for ROI → Analysis ──
document.addEventListener('keydown', (e) => {
    // Only trigger when in ROI mode and not typing in an input
    if (e.code !== 'Space') return;
    if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA' || e.target.tagName === 'SELECT') return;
    if (!roiMode) return;
    if (!currentFile) return;

    e.preventDefault();
    roiMode = false;

    // Start analysis (with or without ROI)
    triggerAnalysis();
});

// ── Skip ROI Button ──
if (roiSkipBtn) {
    roiSkipBtn.addEventListener('click', () => {
        roiRect = null;
        roiReady = false;
        roiMode = false;
        triggerAnalysis();
    });
}

clearBtn.addEventListener('click', resetInterface);
analyzeAnotherBtn.addEventListener('click', resetInterface);

function resetInterface() {
    currentFile = null;
    currentFileType = null;
    fileInput.value = '';
    imagePreview.src = '';
    videoPreview.src = '';

    previewContainer.classList.add('hidden');
    resultsDashboard.classList.add('hidden');
    mainInterface.classList.remove('hidden');
    dropZone.classList.remove('hidden');

    // Reset ROI state
    roiRect = null;
    roiReady = false;
    roiMode = false;
    roiDrawing = false;
    if (roiCanvas) roiCanvas.classList.add('hidden');
    if (roiCtx) roiCtx.clearRect(0, 0, roiCanvas.width, roiCanvas.height);
    if (roiInstruction) roiInstruction.classList.remove('roi-ready-banner');

    aiMessages.innerHTML = '<div class="msg ai">Awaiting new data stream.</div>';

    // Reset speedometer & speech
    setSpeed(60, 'Cruising');
    speechQueue.length = 0;
    if (currentAudio) {
        currentAudio.pause();
        currentAudio = null;
    }
    if ('speechSynthesis' in window) window.speechSynthesis.cancel();
    isSpeaking = false;
}

// ── Analysis (triggered by Space key or Skip button) ──
function triggerAnalysis() {
    // Redirect to the actual analysis handler
    if (!currentFile) return;
    analyzeWithROI();
}

async function analyzeWithROI() {
    if (!currentFile) return;

    mainInterface.classList.add('hidden');
    resultsDashboard.classList.remove('hidden');

    resultImage.classList.add('hidden');
    resultVideo.classList.add('hidden');
    loadingOverlay.classList.remove('hidden');
    advisoryBanner.classList.add('loading');

    detectionsList.innerHTML = '';
    detectionCount.textContent = "Analyzing...";
    advisoryText.textContent = "Combining detection + weather data...";
    advisoryTags.innerHTML = '';

    copilotStatus.textContent = "Processing...";
    copilotDot.style.background = "#EAB308";

    // Reset speedometer for new analysis
    setSpeed(60, 'Cruising');
    speechQueue.length = 0;
    if (currentAudio) {
        currentAudio.pause();
        currentAudio = null;
    }
    if ('speechSynthesis' in window) window.speechSynthesis.cancel();
    isSpeaking = false;

    const roiLabel = roiRect ? ` with ROI region (${Math.round(roiRect.w)}×${Math.round(roiRect.h)}px)` : ' (Full Frame)';
    addBotMessage(`Initiating ${modelTypeSelect.options[modelTypeSelect.selectedIndex].text} analysis${roiLabel}...`, false);

    const formData = new FormData();
    formData.append('file', currentFile);
    formData.append('model_type', modelTypeSelect.value);
    formData.append('lat', userLat);
    formData.append('lon', userLon);

    // Send ROI coordinates as normalized ratios (0-1)
    if (roiRect && roiCanvas) {
        const roiNorm = {
            x: roiRect.x / roiCanvas.width,
            y: roiRect.y / roiCanvas.height,
            w: roiRect.w / roiCanvas.width,
            h: roiRect.h / roiCanvas.height,
        };
        formData.append('roi_x', roiNorm.x.toFixed(4));
        formData.append('roi_y', roiNorm.y.toFixed(4));
        formData.append('roi_w', roiNorm.w.toFixed(4));
        formData.append('roi_h', roiNorm.h.toFixed(4));
    }

    // Use streaming endpoint for videos, regular for images
    const isVideo = currentFileType === 'video';
    const endpoint = isVideo ? '/predict-stream' : '/predict';

    try {
        const response = await fetch(endpoint, {
            method: 'POST',
            body: formData
        });

        if (!response.ok) throw new Error('Inference failed');

        if (isVideo && response.headers.get('content-type')?.includes('ndjson')) {
            // Stream real-time detections
            const reader = response.body.getReader();
            const decoder = new TextDecoder();
            let buffer = '';
            const streamSpoken = new Set();

            while (true) {
                const { done, value } = await reader.read();
                if (done) break;

                buffer += decoder.decode(value, { stream: true });
                const lines = buffer.split('\n');
                buffer = lines.pop(); // Keep incomplete line in buffer

                for (const line of lines) {
                    if (!line.trim()) continue;
                    try {
                        const event = JSON.parse(line);
                        handleStreamEvent(event, streamSpoken);
                    } catch (e) {
                        console.warn('[Stream] Parse error:', e);
                    }
                }
            }
        } else {
            // Regular JSON response (images)
            const data = await response.json();
            renderResults(data);
        }

    } catch (error) {
        console.error('[Error]', error);
        addBotMessage("Inference failed. Please check connection.", true);
        loadingOverlay.classList.add('hidden');
        advisoryBanner.classList.remove('loading');
        copilotStatus.textContent = "Error";
        copilotDot.style.background = "#EF4444";
    }
}

// Keep the old analyze button working too (now hidden, but just in case)
analyzeBtn.addEventListener('click', () => {
    roiMode = false;
    triggerAnalysis();
});

// ── Handle Stream Events (real-time) ──
function handleStreamEvent(event, spokenSet) {
    if (event.event === 'start') {
        const total = event.total_frames;
        advisoryText.textContent = `Scanning ${total} frames live...`;
        if (event.weather && event.weather.status === 'ok') {
            renderWeather(event.weather);
        }
        addBotMessage(`Live streaming started. Processing ${total} frames...`);

    } else if (event.event === 'detection') {
        // REAL-TIME: announce immediately!
        event.detections.forEach(det => {
            const rawName = det.class_name;
            const isCoco = det.source === 'coco_road';
            const isRoadDmg = det.source === 'road_damage';
            const color = isCoco ? 'hsl(180, 80%, 65%)' : isRoadDmg ? 'hsl(30, 80%, 65%)' : `hsl(${[...rawName].reduce((a,c) => a + c.charCodeAt(0), 0) % 360}, 80%, 65%)`;
            const sourceMap = { 'coco_road': 'COCO', 'gtsrb': 'GTSRB', 'gtsrb_v2': 'GTSRB v2', 'gtsrb_fast': 'GTSRB Fast', 'road_damage': 'Road DMG', 'indian_signs': 'Indian' };
            const sourceLabel = sourceMap[det.source] || det.source || 'AI';
            const confPercent = (det.confidence * 100).toFixed(1);

            // Add to detection list immediately
            const li = document.createElement('li');
            li.className = 'neural-card';
            li.style.setProperty('--brand-primary', color);

            let thumb = '';
            if (det.representative_url) {
                const zoomUrl = det.full_frame_url || det.representative_url;
                thumb = `<img src="${det.representative_url}" class="det-thumb" onclick="openModal('${zoomUrl}')">`;
            }

            li.innerHTML = `
                ${thumb}
                <div class="det-body">
                    <div class="det-title">${rawName}</div>
                    <div class="det-meta"><span class="source-badge ${isCoco ? 'source-coco' : isRoadDmg ? 'source-damage' : 'source-custom'}">${sourceLabel}</span> Frame ${det.frame} · ${confPercent}%</div>
                </div>
                <div class="det-conf" style="color: ${color}; background: ${color}20; border-color: ${color}40;">LIVE</div>
            `;
            detectionsList.appendChild(li);

            // Update count
            const count = detectionsList.children.length;
            detectionCount.textContent = `${count} Entities`;

            // Track announced classes
            if (!spokenSet.has(rawName)) {
                spokenSet.add(rawName);
            }
        });

    } else if (event.event === 'immediate_action') {
        // Only show text, do not force speech immediately over conversations
        addBotMessage(`🔴 IMMEDIATE: ${event.action}`, true);
        if (!isSpeaking && !isProcessingVoice && !window.blockAutoSpeech) {
            window.blockAutoSpeech = true;
            setTimeout(() => window.blockAutoSpeech = false, 15000); // 15s cooldown
            speak(event.action);
        }

    } else if (event.event === 'predictive_advisory') {
        if (event.advisory) {
            advisoryText.textContent = event.advisory.combined_advisory;
            
            // Wait 15 seconds before explaining the analysis, if another event or completion doesn't cancel it
            if (window.predictiveTimeout) clearTimeout(window.predictiveTimeout);
            
            window.predictiveTimeout = setTimeout(() => {
                if (!isProcessingVoice) {
                    addBotMessage(`📊 PREDICTIVE: ${event.advisory.combined_advisory}`);
                    speak(event.advisory.combined_advisory);
                }
            }, 15000);
        }

    } else if (event.event === 'speed_sync') {
        const recommendation = calculateContextualSpeed(event.classes);
        setSpeed(recommendation.speed, recommendation.status);

    } else if (event.event === 'progress') {
        const pct = Math.round((event.frame / event.total) * 100);
        advisoryText.textContent = `Scanning frame ${event.frame}/${event.total} (${pct}%)...`;
        copilotStatus.textContent = `Frame ${event.frame}`;

    } else if (event.event === 'complete') {
        // Video done — show final results
        loadingOverlay.classList.add('hidden');
        advisoryBanner.classList.remove('loading');
        copilotStatus.textContent = "Live Sync";
        copilotDot.style.background = "#22C55E";

        const mediaUrl = event.media_url + '?t=' + Date.now();
        resultVideo.src = mediaUrl;
        resultVideo.classList.remove('hidden');

        // Render advisory
        if (event.advisory) {
            advisoryText.textContent = event.advisory.combined_advisory;
            advisoryTags.innerHTML = '';
            if (event.advisory.speed_limit_detected) addTag(`🚗 ${event.advisory.speed_limit_detected} km/h`, 'speed');
            if (event.advisory.weather_risk) addTag(`⚠️ ${event.advisory.weather_risk.level} risk`, `risk-${event.advisory.weather_risk.level}`);
            if (event.advisory.damage_alerts?.length) addTag(`🛣️ ${event.advisory.damage_alerts.length} road issue(s)`, 'damage');
            if (event.advisory.sign_alerts?.length) addTag(`🚦 ${event.advisory.sign_alerts.length} sign(s)`, 'sign');

            // Clear pending midway explanations if video fully complete
            if (window.predictiveTimeout) clearTimeout(window.predictiveTimeout);
            
            // Speak final combined advisory but give it a 10-second delay so it doesn't interrupt chat
            setTimeout(() => {
                if (!isProcessingVoice) {
                    addBotMessage(`📊 FINAL ADVISORY: ${event.advisory.combined_advisory}`);
                    speak(event.advisory.combined_advisory);
                }
            }, 10000);
        }

        if (event.weather && event.weather.status === 'ok') {
            renderWeather(event.weather);
            const risk = event.advisory?.weather_risk;
            if (risk) {
                document.getElementById('weather-risk-text').textContent = risk.level.toUpperCase() + ' RISK';
                document.getElementById('weather-risk-badge').className = `weather-risk-badge risk-${risk.level}`;
            }
        }

        // Save analysis context for chat
        updateAnalysisContext(event);

        addBotMessage(`✅ Analysis complete. ${event.total_frames_processed} frames processed. ${event.detections.length} unique objects detected.`);
    }
}

// ── Render Results ──
function renderResults(data) {
    // Save analysis context for chat
    updateAnalysisContext(data);

    loadingOverlay.classList.add('hidden');
    advisoryBanner.classList.remove('loading');
    copilotStatus.textContent = "Live Sync";
    copilotDot.style.background = "#22C55E";

    if (data.type === 'video') {
        const mediaUrl = data.result_url + '?t=' + Date.now();
        resultVideo.src = mediaUrl;
        resultImage.classList.add('hidden');
        resultVideo.classList.remove('hidden');
    } else {
        resultImage.src = data.result_url;
        resultImage.classList.remove('hidden');
        resultVideo.classList.add('hidden');
        
        // For static images, update the speedometer once with all detected classes
        const allClassNames = data.detections.map(d => d.class_name.split(" (Frame")[0].trim());
        const recommendation = calculateContextualSpeed(allClassNames);
        setSpeed(recommendation.speed, recommendation.status);
    }

    // Update weather risk badge
    if (data.weather && data.weather.status === 'ok') {
        renderWeather(data.weather);
        const risk = data.advisory?.weather_risk;
        if (risk) {
            const riskBadge = document.getElementById('weather-risk-badge');
            const riskText = document.getElementById('weather-risk-text');
            riskText.textContent = risk.level.toUpperCase() + ' RISK';
            riskBadge.className = `weather-risk-badge risk-${risk.level}`;
        }
    }

    // Render advisory
    if (data.advisory) {
        advisoryText.textContent = data.advisory.combined_advisory;

        advisoryTags.innerHTML = '';

        if (data.advisory.speed_limit_detected) {
            addTag(`🚗 ${data.advisory.speed_limit_detected} km/h`, 'speed');
        }

        if (data.advisory.weather_risk) {
            const risk = data.advisory.weather_risk;
            addTag(`⚠️ ${risk.level} risk (${risk.factor}x)`, `risk-${risk.level}`);
        }

        if (data.advisory.damage_alerts?.length) {
            addTag(`🛣️ ${data.advisory.damage_alerts.length} road issue(s)`, 'damage');
        }

        if (data.advisory.sign_alerts?.length) {
            addTag(`🚦 ${data.advisory.sign_alerts.length} sign(s)`, 'sign');
        }

        // Speak the advisory
        setTimeout(() => speak(data.advisory.combined_advisory), 800);
        addBotMessage(data.advisory.combined_advisory, data.advisory.weather_risk?.level === 'high');
    }

    // Render detections
    const totalDets = data.detections.length;
    detectionCount.textContent = `${totalDets} Entities`;

    if (totalDets === 0) {
        detectionsList.innerHTML = `
            <li class="neural-card" style="border-left-color: #475569;">
                <div class="det-body">
                    <div class="det-title">Zero Confident Detections</div>
                    <div class="det-meta">No objects triggered the detection threshold.</div>
                </div>
            </li>
        `;
        return;
    }

    const spokenClasses = new Set();

    data.detections.forEach((det, idx) => {
        const confPercent = (det.confidence * 100).toFixed(1);
        const li = document.createElement('li');
        li.className = 'neural-card';
        li.style.animationDelay = `${idx * 0.06}s`;

        const isCoco = det.source === 'coco_road';
        const isRoadDmg = det.source === 'road_damage';
        const nameHash = [...det.class_name].reduce((acc, c) => acc + c.charCodeAt(0), 0);
        const color = isCoco ? 'hsl(180, 80%, 65%)' : isRoadDmg ? 'hsl(30, 80%, 65%)' : `hsl(${nameHash % 360}, 80%, 65%)`;
        li.style.setProperty('--brand-primary', color);

        const rawClassName = det.class_name.split(" (Frame")[0].trim();
        const sourceMap = { 'coco_road': 'COCO', 'gtsrb': 'GTSRB', 'gtsrb_v2': 'GTSRB v2', 'gtsrb_fast': 'GTSRB Fast', 'road_damage': 'Road DMG', 'indian_signs': 'Indian' };
        const sourceLabel = sourceMap[det.source] || det.source || 'AI';

        if (!spokenClasses.has(rawClassName)) {
            spokenClasses.add(rawClassName);
            const advisory = advisoriesMap[rawClassName] || `Detected: ${rawClassName}. Navigate safely.`;
            addBotMessage(advisory, advisory.includes("Warning") || advisory.includes("Danger"));
            speak(advisory);
        }

        let thumb = "";
        if (det.representative_url) {
            const zoomUrl = det.full_frame_url || det.representative_url;
            thumb = `<img src="${det.representative_url}" class="det-thumb" onclick="openModal('${zoomUrl}')">`;
        }

        li.innerHTML = `
            ${thumb}
            <div class="det-body">
                <div class="det-title">${det.class_name}</div>
                <div class="det-meta"><span class="source-badge ${isCoco ? 'source-coco' : isRoadDmg ? 'source-damage' : 'source-custom'}">${sourceLabel}</span> ${confPercent}%</div>
            </div>
            <div class="det-conf" style="color: ${color}; background: ${color}20; border-color: ${color}40;">
                ${data.type === 'video' ? 'TRK' : confPercent + '%'}
            </div>
        `;

        detectionsList.appendChild(li);
    });
}

function addTag(text, type) {
    const tag = document.createElement('span');
    tag.className = `advisory-tag tag-${type}`;
    tag.textContent = text;
    advisoryTags.appendChild(tag);
}

// ── Speak Advisory Button ──
if (speakAdvisoryBtn) {
    speakAdvisoryBtn.addEventListener('click', () => {
        const text = advisoryText.textContent;
        if (text && text !== "Combining detection + weather data..." && text !== "Analyzing conditions...") {
            speak(text);
        }
    });
}

// ── AI Co-pilot Messages ──
function addBotMessage(text, isAlert = false) {
    const msg = document.createElement('div');
    msg.className = `msg ${isAlert ? 'ai-alert' : 'ai'}`;
    msg.textContent = text;
    aiMessages.appendChild(msg);
    aiMessages.scrollTop = aiMessages.scrollHeight;
}

// ── AI Voice System (Kokoro TTS / Fallback) ──
const speechQueue = [];
let isSpeaking = false;
let ttsAvailable = false;
let ttsEngine = null;
let currentAudio = null;

// Check server TTS status on load
(async function checkTTS() {
    try {
        const res = await fetch('/tts/status');
        if (res.ok) {
            const status = await res.json();
            ttsAvailable = status.available;
            ttsEngine = status.engine;
            if (ttsAvailable) {
                console.log(`[TTS] AI Voice online: ${ttsEngine} engine`);
                addBotMessage(`🔊 AI Voice System online (${ttsEngine === 'kokoro' ? 'Kokoro Neural TTS' : 'System TTS'})`);
            }
        }
    } catch (e) {
        console.warn('[TTS] Status check failed, using browser fallback');
    }
})();

function speak(text) {
    if (!text || text.trim().length === 0) return;
    speechQueue.push(text);
    processQueue();
}

async function processQueue() {
    if (isSpeaking || speechQueue.length === 0) return;
    isSpeaking = true;
    const text = speechQueue.shift();

    // Show speaking indicator
    const pulseEl = document.querySelector('.pulse-indicator');
    if (pulseEl) pulseEl.classList.add('speaking');

    if (ttsAvailable) {
        // Use server-side AI TTS
        try {
            const res = await fetch('/tts', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ text, voice: 'af_heart', speed: 1.0 })
            });

            if (res.ok) {
                const audioBlob = await res.blob();
                const audioUrl = URL.createObjectURL(audioBlob);
                currentAudio = new Audio(audioUrl);
                currentAudio.volume = 1.0;

                currentAudio.onended = () => {
                    URL.revokeObjectURL(audioUrl);
                    currentAudio = null;
                    isSpeaking = false;
                    if (pulseEl) pulseEl.classList.remove('speaking');
                    processQueue();
                };
                currentAudio.onerror = () => {
                    URL.revokeObjectURL(audioUrl);
                    currentAudio = null;
                    isSpeaking = false;
                    if (pulseEl) pulseEl.classList.remove('speaking');
                    processQueue();
                };

                await currentAudio.play();
                return;
            }
        } catch (e) {
            console.warn('[TTS] Server error, falling back to browser:', e);
        }
    }

    // Fallback: Browser SpeechSynthesis
    if ('speechSynthesis' in window) {
        const utter = new SpeechSynthesisUtterance(text);
        utter.rate = 1.0;
        utter.pitch = 1.05;
        utter.onend = () => {
            isSpeaking = false;
            if (pulseEl) pulseEl.classList.remove('speaking');
            processQueue();
        };
        utter.onerror = () => {
            isSpeaking = false;
            if (pulseEl) pulseEl.classList.remove('speaking');
            processQueue();
        };
        window.speechSynthesis.speak(utter);
    } else {
        isSpeaking = false;
        if (pulseEl) pulseEl.classList.remove('speaking');
        processQueue();
    }
}

// ── Speedometer ──

function drawSpeedometer(speed) {
    const W = speedCanvas.width;
    const H = speedCanvas.height;
    const cx = W / 2;
    const cy = H - 20;
    const radius = 110;

    speedCtx.clearRect(0, 0, W, H);

    // BMW distinctive style:
    // Outer Metallic Ring
    speedCtx.beginPath();
    speedCtx.arc(cx, cy, radius, Math.PI, 0, false);
    speedCtx.lineWidth = 4;
    speedCtx.strokeStyle = 'rgba(200,200,210,0.8)';
    speedCtx.stroke();

    // Middle thick black/grey band
    speedCtx.beginPath();
    speedCtx.arc(cx, cy, radius - 8, Math.PI, 0, false);
    speedCtx.lineWidth = 14;
    speedCtx.strokeStyle = 'rgba(20,20,25,0.9)';
    speedCtx.stroke();

    // BMW blue/red M-accents on the arc
    // Blue segment for 0-60
    speedCtx.beginPath();
    const angle60 = Math.PI + (60 / 140) * Math.PI;
    speedCtx.arc(cx, cy, radius - 8, Math.PI, angle60, false);
    speedCtx.lineWidth = 14;
    speedCtx.strokeStyle = 'rgba(0, 114, 206, 0.4)'; // M Blue
    speedCtx.stroke();

    // Red segment for 100-140
    speedCtx.beginPath();
    const angle100 = Math.PI + (100 / 140) * Math.PI;
    speedCtx.arc(cx, cy, radius - 8, angle100, Math.PI * 2, false);
    speedCtx.lineWidth = 14;
    speedCtx.strokeStyle = 'rgba(226, 35, 26, 0.4)'; // M Red
    speedCtx.stroke();

    // Tick marks and Labels
    for (let i = 0; i <= 140; i += 20) {
        const angle = Math.PI + (i / 140) * Math.PI;
        const inner = radius - 20;
        const outer = radius;
        const x1 = cx + Math.cos(angle) * inner;
        const y1 = cy + Math.sin(angle) * inner;
        const x2 = cx + Math.cos(angle) * outer;
        const y2 = cy + Math.sin(angle) * outer;

        speedCtx.beginPath();
        speedCtx.moveTo(x1, y1);
        speedCtx.lineTo(x2, y2);
        speedCtx.lineWidth = i % 40 === 0 ? 3 : 1;
        speedCtx.strokeStyle = i % 40 === 0 ? '#fff' : 'rgba(255,255,255,0.5)';
        speedCtx.stroke();

        // Labels
        if (i % 40 === 0 || i === 0 || i === 140) {
            const lx = cx + Math.cos(angle) * (inner - 14);
            const ly = cy + Math.sin(angle) * (inner - 14);
            speedCtx.fillStyle = '#fff';
            speedCtx.font = 'bold 12px Inter, sans-serif';
            speedCtx.textAlign = 'center';
            speedCtx.textBaseline = 'middle';
            speedCtx.fillText(i, lx, ly);
        }
    }

    // BMW Orange/Red Needle
    const needleAngle = Math.PI + (Math.min(Math.max(speed, 0), 140) / 140) * Math.PI;

    // Needle Trail (glowing arc)
    speedCtx.beginPath();
    speedCtx.arc(cx, cy, radius - 20, Math.PI, needleAngle, false);
    speedCtx.lineWidth = 4;
    speedCtx.strokeStyle = 'rgba(255, 60, 0, 0.8)';
    speedCtx.stroke();

    // The Needle Point
    const needleLen = radius - 15;
    const nx = cx + Math.cos(needleAngle) * needleLen;
    const ny = cy + Math.sin(needleAngle) * needleLen;

    speedCtx.beginPath();
    speedCtx.moveTo(cx, cy);
    speedCtx.lineTo(nx, ny);
    speedCtx.lineWidth = 4;
    speedCtx.strokeStyle = '#FF3C00';
    speedCtx.shadowColor = '#FF3C00';
    speedCtx.shadowBlur = 10;
    speedCtx.lineCap = 'round';
    speedCtx.stroke();
    speedCtx.shadowBlur = 0; // reset

    // Center cap
    speedCtx.beginPath();
    speedCtx.arc(cx, cy, 12, 0, Math.PI * 2);
    speedCtx.fillStyle = '#111';
    speedCtx.fill();
    speedCtx.lineWidth = 2;
    speedCtx.strokeStyle = '#555';
    speedCtx.stroke();

    speedCtx.beginPath();
    speedCtx.arc(cx, cy, 4, 0, Math.PI * 2);
    speedCtx.fillStyle = '#FF3C00';
    speedCtx.fill();
}

function animateSpeed() {
    if (Math.abs(currentSpeed - targetSpeed) < 0.5) {
        currentSpeed = targetSpeed;
        drawSpeedometer(currentSpeed);
        speedValueEl.textContent = Math.round(currentSpeed);
        updateSpeedStatus();
        return;
    }
    currentSpeed += (targetSpeed - currentSpeed) * 0.08;
    drawSpeedometer(currentSpeed);
    speedValueEl.textContent = Math.round(currentSpeed);
    updateSpeedStatus();
    speedAnimFrame = requestAnimationFrame(animateSpeed);
}

let currentSpeedReason = 'Cruising';

function setSpeed(newSpeed, reason) {
    targetSpeed = Math.max(0, Math.min(140, newSpeed));
    if (reason) currentSpeedReason = reason;
    if (speedAnimFrame) cancelAnimationFrame(speedAnimFrame);
    animateSpeed();
}

function updateSpeedStatus() {
    const s = Math.round(currentSpeed);
    speedStatusEl.className = 'speed-status';
    
    // Always show the dynamic text reason, just change the color based on severity
    speedStatusEl.textContent = currentSpeedReason.toUpperCase();
    
    if (s <= 0) {
        speedStatusEl.classList.add('stopped');
    } else if (s <= 40 && s < 50) {
        speedStatusEl.classList.add('warning');
    } else if (s > 100) {
        speedStatusEl.classList.add('danger');
    }
}

// Context-Aware Algorithmic Speed Calculation
function calculateContextualSpeed(classesArray) {
    if (!classesArray || classesArray.length === 0) return { speed: 60, status: 'Cruising (Clear)' };

    let counts = {
        person: 0,
        vehicle: 0,
        crack: 0,
        pothole: 0,
        stop: 0,
        trafficLight: 0,
        speedLimit: null
    };

    classesArray.forEach(c => {
        const cl = c.toLowerCase();
        if (cl.includes('person') || cl.includes('pedestrian') || cl.includes('children')) counts.person++;
        else if (cl.includes('car') || cl.includes('truck') || cl.includes('bus') || cl.includes('motorcycle') || cl.includes('vehicle') || cl.includes('bicycle')) counts.vehicle++;
        else if (cl.includes('pothole') || cl.includes('damage') || cl.includes('bumpy')) counts.pothole++;
        else if (cl.includes('crack')) counts.crack++;
        else if (cl.includes('stop') || cl.includes('no entry')) counts.stop++;
        else if (cl.includes('traffic light')) counts.trafficLight++;
        else {
            const limitMatch = cl.match(/speed limit.*?(\d+)/);
            if (limitMatch && !counts.speedLimit) counts.speedLimit = parseInt(limitMatch[1]);
        }
    });

    if (counts.stop > 0) return { speed: 0, status: 'STOP REQUIRED' };

    let baseSpeed = counts.speedLimit || 60;
    let factors = [];

    // The Algorithmic Reductions (User requested 1 car (-10) + 2 person (-30) + 1 crack (-10) = 10 km/h)
    if (counts.person > 0) {
        baseSpeed -= (counts.person * 15);
        factors.push(`${counts.person} Person(s)`);
    }
    if (counts.vehicle > 0) {
        baseSpeed -= (counts.vehicle * 10);
        factors.push(`${counts.vehicle} Vehicle(s)`);
    }
    if (counts.crack > 0) {
        baseSpeed -= (counts.crack * 10);
        factors.push(`${counts.crack} Crack(s)`);
    }
    if (counts.pothole > 0) {
        baseSpeed -= (counts.pothole * 30);
        factors.push(`${counts.pothole} Pothole(s)`);
    }
    if (counts.trafficLight > 0) {
        baseSpeed = Math.min(baseSpeed, 30);
        factors.push('Traffic Light');
    }

    if (factors.length === 0) {
        return { speed: baseSpeed - 2 + Math.random() * 4, status: 'Cruising (Adjusting)' };
    }

    // Never drop below 10 km/h unless strictly stopped
    baseSpeed = Math.max(10, baseSpeed);
    
    return { speed: baseSpeed, status: `ENV: ${factors.join(' + ')}` };
}

// Initialize speedometer
drawSpeedometer(60);

// ── Conversational AI Chat System ──
let lastAnalysisData = { detections: [], advisory: null, weather: null };

// Track analysis results for chat context
function updateAnalysisContext(data) {
    if (data.detections) lastAnalysisData.detections = data.detections;
    if (data.advisory) lastAnalysisData.advisory = data.advisory;
    if (data.weather) lastAnalysisData.weather = data.weather;
}

const chatInput = document.getElementById('chat-input');
const chatSendBtn = document.getElementById('chat-send-btn');

function addUserMessage(text) {
    const msg = document.createElement('div');
    msg.className = 'msg user';
    msg.textContent = text;
    aiMessages.appendChild(msg);
    aiMessages.scrollTop = aiMessages.scrollHeight;
}

function showTypingIndicator() {
    const indicator = document.createElement('div');
    indicator.className = 'typing-indicator';
    indicator.id = 'typing-indicator';
    indicator.innerHTML = '<div class="dot-bounce"></div><div class="dot-bounce"></div><div class="dot-bounce"></div>';
    aiMessages.appendChild(indicator);
    aiMessages.scrollTop = aiMessages.scrollHeight;
    return indicator;
}

function removeTypingIndicator() {
    const el = document.getElementById('typing-indicator');
    if (el) el.remove();
}

// ── Stop All Speech (interruption) ──
function stopAllSpeech() {
    // Stop server TTS audio
    if (currentAudio) {
        currentAudio.pause();
        currentAudio.currentTime = 0;
        currentAudio = null;
    }
    // Clear queue
    speechQueue.length = 0;
    isSpeaking = false;
    // Stop browser TTS
    if ('speechSynthesis' in window) window.speechSynthesis.cancel();
    // Reset visual indicator
    const pulseEl = document.querySelector('.pulse-indicator');
    if (pulseEl) pulseEl.classList.remove('speaking');
}

async function sendChatMessage(inputText) {
    const text = (inputText || chatInput.value).trim();
    if (!text) return;

    // ★ STOP any current speech when user sends a message
    stopAllSpeech();

    // Show user message
    addUserMessage(text);
    chatInput.value = '';
    chatSendBtn.disabled = true;

    // Show typing indicator
    showTypingIndicator();

    // Build context from current analysis state
    const context = {
        detections: lastAnalysisData.detections || [],
        weather: lastAnalysisData.weather || currentWeather || {},
        advisory: lastAnalysisData.advisory || {},
        speed: Math.round(currentSpeed),
        speedStatus: currentSpeedReason,
    };

    try {
        const res = await fetch('/chat', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ message: text, context })
        });

        removeTypingIndicator();

        if (res.ok) {
            const data = await res.json();

            // Show AI response
            addBotMessage(data.response);

            // Play audio response (AI speaks back)
            if (data.audio_url) {
                const audio = new Audio(data.audio_url);
                audio.volume = 1.0;
                currentAudio = audio;  // Track so it can be interrupted
                const pulseEl = document.querySelector('.pulse-indicator');
                if (pulseEl) pulseEl.classList.add('speaking');
                audio.onended = () => {
                    currentAudio = null;
                    if (pulseEl) pulseEl.classList.remove('speaking');
                };
                audio.onerror = () => {
                    currentAudio = null;
                    if (pulseEl) pulseEl.classList.remove('speaking');
                    speak(data.response);
                };
                try {
                    await audio.play();
                } catch (e) {
                    currentAudio = null;
                    if (pulseEl) pulseEl.classList.remove('speaking');
                    speak(data.response);
                }
            } else {
                speak(data.response);
            }
        } else {
            addBotMessage("Sorry, I encountered an error processing your question. Please try again.", true);
        }
    } catch (e) {
        removeTypingIndicator();
        addBotMessage("Connection error. Please check if the server is running.", true);
        console.error('[Chat]', e);
    }

    chatSendBtn.disabled = false;
    chatInput.focus();
}

// Event listeners for chat
if (chatSendBtn) {
    chatSendBtn.addEventListener('click', () => sendChatMessage());
}
if (chatInput) {
    chatInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            sendChatMessage();
        }
    });
}

// ══════════════════════════════════════════════════════════════
// ── ALWAYS-ON Voice-to-Voice System (Continuous Listening) ──
// ══════════════════════════════════════════════════════════════
const chatMicBtn = document.getElementById('chat-mic-btn');
let recognition = null;
let isListening = false;
let isMicEnabled = false;
let voiceDebounceTimer = null;
let currentTranscript = '';
let isProcessingVoice = false;

const MIN_CONFIDENCE = 0.5;
const MIN_WORDS = 1;
const SILENCE_DELAY = 1500;

const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;

if (SpeechRecognition) {
    recognition = new SpeechRecognition();
    recognition.continuous = true;
    recognition.interimResults = true;
    recognition.lang = 'en-US';
    recognition.maxAlternatives = 1;

    let sentenceStartIndex = 0; // Tracks the start index of the current sentence

    recognition.onstart = () => {
        isListening = true;
        sentenceStartIndex = 0; // Reset index on fresh start
        if (chatMicBtn) chatMicBtn.classList.add('recording');
        chatInput.placeholder = '🎙️ Listening... speak anytime';
        console.log('[Voice] Mic is live');
    };

    recognition.onresult = (event) => {
        clearTimeout(voiceDebounceTimer);

        let interimTranscript = '';
        let finalTranscript = '';

        for (let i = sentenceStartIndex; i < event.results.length; ++i) {
            if (event.results[i].isFinal) {
                finalTranscript += event.results[i][0].transcript + ' ';
            } else {
                interimTranscript += event.results[i][0].transcript;
            }
        }

        const fullTranscript = (finalTranscript + interimTranscript).trim();

        if (fullTranscript.length > 0) {
            // Stop AI speech if the user interrupts and has uttered meaningful words
            stopAllSpeech();

            chatInput.value = fullTranscript;
            chatInput.style.color = (interimTranscript.length > 0) ? '#94a3b8' : '#ffffff'; // slate-400 during interim

            voiceDebounceTimer = setTimeout(() => {
                let messageToSend = fullTranscript.replace(/hey\s*(sentinel|sentinal|centinel|centinal)\b/gi, '').trim();
                
                // Fast-forward sentence start index to ignore these results going forward
                sentenceStartIndex = event.results.length;

                if (messageToSend.length === 0) messageToSend = "Hello";
                
                if (messageToSend.split(/\s+/).length >= MIN_WORDS && !isProcessingVoice) {
                    isProcessingVoice = true;
                    chatInput.value = '';
                    
                    // Instantly send message to AI Backend
                    sendChatMessage(messageToSend).then(() => {
                        isProcessingVoice = false;
                    });
                }
            }, SILENCE_DELAY);
        }
    };

    recognition.onerror = (event) => {
        console.warn('[Voice] Error:', event.error);
        if (event.error === 'not-allowed') {
            isListening = false;
            isMicEnabled = false;
            if (chatMicBtn) chatMicBtn.classList.remove('recording');
            chatInput.placeholder = 'Mic blocked — type your question...';
            addBotMessage("🎙️ Microphone access denied. Go to browser address bar → click the lock/info icon → allow Microphone → refresh page.", true);
        }
    };

    recognition.onend = () => {
        isListening = false;
        if (isMicEnabled) {
            setTimeout(() => {
                if (isMicEnabled && !isListening) {
                    try { recognition.start(); } catch (e) {}
                }
            }, 300);
        } else {
            if (chatMicBtn) chatMicBtn.classList.remove('recording');
            chatInput.placeholder = 'Ask about detections, weather, speed...';
        }
    };

    console.log('[Voice] Speech recognition available');
} else {
    if (chatMicBtn) {
        chatMicBtn.classList.add('unsupported');
        chatMicBtn.title = 'Voice input not supported — use Chrome or Edge';
    }
    console.warn('[Voice] SpeechRecognition not supported');
}

if (chatMicBtn && recognition) {
    chatMicBtn.addEventListener('click', () => {
        stopAllSpeech();
        if (isMicEnabled) {
            isMicEnabled = false;
            try { recognition.stop(); } catch (e) {}
            chatMicBtn.classList.remove('recording');
            chatInput.placeholder = 'Ask about detections, weather, speed...';
            addBotMessage("🎙️ Voice mode OFF. You can type your questions.");
        } else {
            isMicEnabled = true;
            try { recognition.start(); } catch (e) { console.warn('[Voice] Start error:', e); }
            addBotMessage("🎙️ Voice mode ON — I'm always listening. Just speak anytime.");
        }
    });

    // ── Auto-Start Voice System on Load ──
    const startMicAutomatically = () => {
        if (!isMicEnabled) {
            isMicEnabled = true;
            try { recognition.start(); } catch (e) { console.warn('[Voice] Auto-start error:', e); }
        }
    };

    // Try starting immediately
    setTimeout(startMicAutomatically, 1000);

    // Fallback: Start securely on first user interaction if browser blocked auto-start
    document.addEventListener('click', () => {
        if (!isMicEnabled) startMicAutomatically();
    }, { once: true });
}
