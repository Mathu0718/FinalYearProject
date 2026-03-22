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
    } else {
        videoPreview.src = url;
        videoPreview.classList.remove('hidden');
    }

    dropZone.classList.add('hidden');
    previewContainer.classList.remove('hidden');

    addBotMessage(`Stream loaded: ${file.name}. Select model and execute inference.`);
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

    aiMessages.innerHTML = '<div class="msg ai">Awaiting new data stream.</div>';

    // Reset speedometer & speech
    setSpeed(60, 'Cruising');
    speechQueue.length = 0;
    window.speechSynthesis.cancel();
    isSpeaking = false;
}

// ── Analysis ──
analyzeBtn.addEventListener('click', async () => {
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
    window.speechSynthesis.cancel();
    isSpeaking = false;

    addBotMessage(`Initiating ${modelTypeSelect.options[modelTypeSelect.selectedIndex].text} analysis...`, false);

    const formData = new FormData();
    formData.append('file', currentFile);
    formData.append('model_type', modelTypeSelect.value);
    formData.append('lat', userLat);
    formData.append('lon', userLon);

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

            // Speak advisory IMMEDIATELY (queued — no overlap)
            if (!spokenSet.has(rawName)) {
                spokenSet.add(rawName);
                const advisory = advisoriesMap[rawName] || `Detected: ${rawName}. Navigate safely.`;
                const isAlert = advisory.includes("Warning") || advisory.includes("Danger");
                addBotMessage(`🔴 LIVE: ${advisory}`, isAlert);
                speak(advisory);
            }
        });

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

            // Speak final combined advisory
            setTimeout(() => speak(event.advisory.combined_advisory), 500);
        }

        if (event.weather && event.weather.status === 'ok') {
            renderWeather(event.weather);
            const risk = event.advisory?.weather_risk;
            if (risk) {
                document.getElementById('weather-risk-text').textContent = risk.level.toUpperCase() + ' RISK';
                document.getElementById('weather-risk-badge').className = `weather-risk-badge risk-${risk.level}`;
            }
        }

        addBotMessage(`✅ Analysis complete. ${event.total_frames_processed} frames processed. ${event.detections.length} unique objects detected.`);
    }
}

// ── Render Results ──
function renderResults(data) {
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

// ── Speech Queue System ──
const speechQueue = [];
let isSpeaking = false;

function speak(text) {
    if (!('speechSynthesis' in window)) return;
    speechQueue.push(text);
    processQueue();
}

function processQueue() {
    if (isSpeaking || speechQueue.length === 0) return;
    isSpeaking = true;
    const text = speechQueue.shift();
    const utter = new SpeechSynthesisUtterance(text);
    utter.rate = 1.0;
    utter.pitch = 1.05;
    utter.onend = () => {
        isSpeaking = false;
        processQueue();
    };
    utter.onerror = () => {
        isSpeaking = false;
        processQueue();
    };
    window.speechSynthesis.speak(utter);
}

// ── Speedometer ──

function drawSpeedometer(speed) {
    const W = speedCanvas.width;
    const H = speedCanvas.height;
    const cx = W / 2;
    const cy = H - 10;
    const radius = 100;

    speedCtx.clearRect(0, 0, W, H);

    // Background arc
    speedCtx.beginPath();
    speedCtx.arc(cx, cy, radius, Math.PI, 0, false);
    speedCtx.lineWidth = 18;
    speedCtx.strokeStyle = 'rgba(255,255,255,0.05)';
    speedCtx.stroke();

    // Color zones: green (0-60), yellow (60-100), red (100-140)
    const zones = [
        { start: 0, end: 60, color: '#22C55E' },
        { start: 60, end: 100, color: '#F59E0B' },
        { start: 100, end: 140, color: '#EF4444' },
    ];

    zones.forEach(z => {
        const sAngle = Math.PI + (z.start / 140) * Math.PI;
        const eAngle = Math.PI + (z.end / 140) * Math.PI;
        speedCtx.beginPath();
        speedCtx.arc(cx, cy, radius, sAngle, eAngle, false);
        speedCtx.lineWidth = 18;
        speedCtx.strokeStyle = z.color + '30';
        speedCtx.stroke();
    });

    // Active arc
    const activeAngle = Math.PI + (Math.min(speed, 140) / 140) * Math.PI;
    let arcColor = '#22C55E';
    if (speed > 100) arcColor = '#EF4444';
    else if (speed > 60) arcColor = '#F59E0B';
    else if (speed <= 0) arcColor = '#EF4444';

    speedCtx.beginPath();
    speedCtx.arc(cx, cy, radius, Math.PI, activeAngle, false);
    speedCtx.lineWidth = 18;
    speedCtx.strokeStyle = arcColor;
    speedCtx.lineCap = 'round';
    speedCtx.stroke();

    // Tick marks & labels
    for (let i = 0; i <= 140; i += 20) {
        const angle = Math.PI + (i / 140) * Math.PI;
        const inner = radius - 28;
        const outer = radius - 12;
        const x1 = cx + Math.cos(angle) * inner;
        const y1 = cy + Math.sin(angle) * inner;
        const x2 = cx + Math.cos(angle) * outer;
        const y2 = cy + Math.sin(angle) * outer;

        speedCtx.beginPath();
        speedCtx.moveTo(x1, y1);
        speedCtx.lineTo(x2, y2);
        speedCtx.lineWidth = 2;
        speedCtx.strokeStyle = 'rgba(255,255,255,0.3)';
        speedCtx.lineCap = 'round';
        speedCtx.stroke();

        // Labels
        const lx = cx + Math.cos(angle) * (inner - 12);
        const ly = cy + Math.sin(angle) * (inner - 12);
        speedCtx.fillStyle = 'rgba(255,255,255,0.4)';
        speedCtx.font = '9px Inter, sans-serif';
        speedCtx.textAlign = 'center';
        speedCtx.textBaseline = 'middle';
        speedCtx.fillText(i, lx, ly);
    }

    // Needle
    const needleAngle = Math.PI + (Math.min(Math.max(speed, 0), 140) / 140) * Math.PI;
    const needleLen = radius - 35;
    const nx = cx + Math.cos(needleAngle) * needleLen;
    const ny = cy + Math.sin(needleAngle) * needleLen;

    speedCtx.beginPath();
    speedCtx.moveTo(cx, cy);
    speedCtx.lineTo(nx, ny);
    speedCtx.lineWidth = 3;
    speedCtx.strokeStyle = '#fff';
    speedCtx.lineCap = 'round';
    speedCtx.stroke();

    // Center dot
    speedCtx.beginPath();
    speedCtx.arc(cx, cy, 6, 0, Math.PI * 2);
    speedCtx.fillStyle = arcColor;
    speedCtx.fill();
    speedCtx.beginPath();
    speedCtx.arc(cx, cy, 3, 0, Math.PI * 2);
    speedCtx.fillStyle = '#fff';
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
