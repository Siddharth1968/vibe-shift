import React, { useState, useRef, useEffect } from 'react';
import {
  AlertCircle,
  Camera,
  CameraOff,
  CheckCircle2,
  RefreshCw,
  ShieldCheck,
  Sparkles,
  Upload,
} from 'lucide-react';
import { estimateEmotionFromImage } from '../services/emotionService';
import { MOCK_EMOTION_SCENARIOS } from '../mocks/mockData';
import { getMoodMeta } from '../config/constants';

export default function CameraPanel({
  apiMode = 'mock',
  onEmotionDetected,
  onSwitchToManual,
}) {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const streamRef = useRef(null);
  const inFlightRef = useRef(false);
  const capturePendingRef = useRef(false);
  const versionRef = useRef(0);
  const mountedRef = useRef(true);
  const sessionRef = useRef(crypto.randomUUID());
  const selectionRef = useRef(null);
  const notifiedMoodRef = useRef(null);

  const [cameraActive, setCameraActive] = useState(false);
  const [isVideoReady, setIsVideoReady] = useState(false);
  const [cameraError, setCameraError] = useState(null);
  const [analyzing, setAnalyzing] = useState(false);
  const [selectedMockScenario, setSelectedMockScenario] =
    useState('happy_success');
  const [result, setResult] = useState(null);
  const [inferenceMode, setInferenceMode] = useState('single_view');
  const [autoScan, setAutoScan] = useState(true);
  const [debugMode, setDebugMode] = useState(false);

  const stopCamera = () => {
    versionRef.current++;
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((track) => track.stop());
      streamRef.current = null;
    }
    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }
    setCameraActive(false);
    setIsVideoReady(false);
    setResult(null);
    notifiedMoodRef.current = null;
  };

  useEffect(() => {
    mountedRef.current = true;
    return () => {
      mountedRef.current = false;
      stopCamera();
    };
  }, []);

  useEffect(() => {
    if (!cameraActive || !isVideoReady || apiMode !== 'live' || !autoScan) return;
    let cancelled = false;
    let timer;
    const scan = async () => {
      await captureSnapshot();
      if (!cancelled) timer = setTimeout(scan, 250);
    };
    timer = setTimeout(scan, 250);
    return () => { cancelled = true; clearTimeout(timer); };
  }, [cameraActive, isVideoReady, apiMode, autoScan, inferenceMode, debugMode]);

  useEffect(() => {
    if (cameraActive && streamRef.current && videoRef.current) {
      if (videoRef.current.srcObject !== streamRef.current) {
        videoRef.current.srcObject = streamRef.current;
      }
      const playPromise = videoRef.current.play();
      if (playPromise !== undefined) {
        playPromise.catch(() => {});
      }
    }
  }, [cameraActive]);

  const startCamera = async () => {
    const version = ++versionRef.current;
    sessionRef.current = crypto.randomUUID();
    setCameraError(null);
    setResult(null);
    setIsVideoReady(false);

    if (
      !navigator.mediaDevices ||
      typeof navigator.mediaDevices.getUserMedia !== 'function'
    ) {
      setCameraError(
        'Webcam access is not supported in this browser environment. You can upload a photo or choose your mood manually below.'
      );
      return;
    }

    try {
      let stream;
      try {
        stream = await navigator.mediaDevices.getUserMedia({
          video: { facingMode: 'user', width: { ideal: 640 }, height: { ideal: 480 } },
          audio: false,
        });
      } catch (error) {
        if (error?.name === 'NotAllowedError') throw error;
        stream = await navigator.mediaDevices.getUserMedia({
          video: true,
          audio: false,
        });
      }

      if (!mountedRef.current || version !== versionRef.current) {
        stream.getTracks().forEach(track => track.stop());
        return;
      }

      streamRef.current = stream;
      setCameraActive(true);

      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        try {
          await videoRef.current.play();
        } catch {
          // Playback will automatically begin when metadata loads
        }
      }
    } catch (err) {
      setCameraError(
        err?.name === 'NotAllowedError'
          ? 'Camera permission was declined. That is completely fine — you can select your mood manually or upload an image.'
          : 'Could not open webcam stream. You can upload an image or use manual mood selection.'
      );
    }
  };

  const handleAnalyzeBlob = async (blob, snapshot = false) => {
    if (inFlightRef.current) return;
    const version = versionRef.current;
    inFlightRef.current = true;
    setAnalyzing(true);
    setCameraError(null);
    const selection = !snapshot && selectionRef.current && canvasRef.current
      ? [selectionRef.current[0] * canvasRef.current.width,
         selectionRef.current[1] * canvasRef.current.height]
      : null;
    selectionRef.current = null;
    try {
      const response = await estimateEmotionFromImage(blob, {
        apiMode,
        mockScenarioId: selectedMockScenario,
        liveSession: sessionRef.current,
        selection,
        snapshot,
        debug: debugMode,
        horizontalFlip: inferenceMode === 'horizontal_flip',
      });
      if (!mountedRef.current || version !== versionRef.current) return;
      setResult(response);
      if (onEmotionDetected && (response.emotion !== notifiedMoodRef.current || snapshot || apiMode === 'mock')) {
        onEmotionDetected(response);
        notifiedMoodRef.current = response.emotion;
      }
    } catch (err) {
      if (!mountedRef.current || version !== versionRef.current) return;
      setResult(null);
      setCameraError(
        err?.message ||
          'Emotion estimation service could not process the image. Please select your mood manually.'
      );
    } finally {
      inFlightRef.current = false;
      if (mountedRef.current) setAnalyzing(false);
    }
  };

  const captureSnapshot = async () => {
    if (inFlightRef.current || capturePendingRef.current) return;
    const version = versionRef.current;
    const video = videoRef.current;
    const canvas = canvasRef.current;

    if (!video || !canvas) {
      setCameraError('Camera preview is not available.');
      return;
    }

    const width = video.videoWidth;
    const height = video.videoHeight;

    if (!width || !height || video.readyState < 2) {
      setCameraError(
        'Camera video stream is still initializing. Please wait a moment and try again.'
      );
      return;
    }

    const scale = Math.min(1, 1280 / Math.max(width, height));
    canvas.width = Math.max(1, Math.round(width * scale));
    canvas.height = Math.max(1, Math.round(height * scale));
    const ctx = canvas.getContext('2d');
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

    capturePendingRef.current = true;
    try {
      const blob = await new Promise(resolve => canvas.toBlob(resolve, 'image/jpeg', 0.85));
      if (!mountedRef.current || version !== versionRef.current) return;
      if (blob && blob.size > 0) {
        await handleAnalyzeBlob(blob);
      } else {
        setCameraError('Failed to capture a valid camera snapshot. Please try again or upload a photo.');
      }
    } finally {
      capturePendingRef.current = false;
    }
  };

  const handleFileUpload = (e) => {
    const file = e.target.files?.[0];
    e.target.value = '';
    if (!file) return;
    handleAnalyzeBlob(file, true);
  };

  const detectedMoodMeta = result?.emotion ? getMoodMeta(result.emotion) : null;

  return (
    <div
      className="impact-camera-card"
      data-testid="camera-panel"
    >
      <div className="impact-camera-head">
        <div>
          <span className="badge badge-accent" style={{ marginBottom: '0.55rem' }}>
            <ShieldCheck size={13} />
            Optional & Privacy-First
          </span>
          <h3 className="impact-camera-title">
            Camera Expression Check-In
          </h3>
          <p className="impact-camera-sub">
            Your photo is only used for a single expression estimate and is never stored. Facial expression is an estimate — you always confirm or change your mood before music plays.
          </p>
        </div>

        <button
          type="button"
          className="btn btn-secondary btn-sm"
          onClick={onSwitchToManual}
        >
          Use Manual Mood Tiles
        </button>
      </div>

      {apiMode === 'live' && <div className="impact-camera-sim-box">
        <p role="status" data-testid="active-ai-model-status">
          Active AI model — {result?.modelName || 'CleanBTNVit (Phase 12)'} (YuNet + single-model inference)
        </p>
        <label htmlFor="live-inference-mode">Inference Mode</label>
        <select
          id="live-inference-mode"
          className="impact-select"
          value={inferenceMode}
          onChange={event => {
            versionRef.current++;
            selectionRef.current = null;
            notifiedMoodRef.current = null;
            setResult(null);
            setCameraError(null);
            setInferenceMode(event.target.value);
          }}
        >
          <option value="single_view">Single-view (Fast — default)</option>
          <option value="horizontal_flip">Horizontal-flip (Accuracy mode)</option>
        </select>
        <label><input type="checkbox" checked={autoScan} onChange={event => setAutoScan(event.target.checked)} /> Live expression scanning</label>
        <label><input type="checkbox" checked={debugMode} onChange={event => {
          versionRef.current++;
          setResult(null);
          setDebugMode(event.target.checked);
        }} /> Local prediction debug</label>
        <p>Click your face when multiple people are visible. Facial expressions are visual estimates — you always confirm your mood.</p>
        {result && <p>Face detection — {result.faceStatus || result.status}. Expression — {result.expression || 'uncertain'}.
          {result.inferenceMode && ` Mode — ${result.inferenceMode === 'horizontal_flip' ? 'horizontal-flip' : 'single-view'}.`}
          {result.inferenceMs != null && ` Inference ${result.inferenceMs.toFixed(1)} ms.`}</p>}
        {debugMode && result && <div role="status" data-testid="live-prediction-debug">
          <p>Raw prediction — {result.rawExpression || 'unavailable'} ({((result.rawConfidence || 0) * 100).toFixed(1)}%).
            Stabilized expression — {result.expression || 'uncertain'}. Music mood — {result.emotion || 'choose manually'}.</p>
          <pre style={{whiteSpace:'pre-wrap',overflowWrap:'anywhere'}}>{JSON.stringify(result.debug, null, 2)}</pre>
        </div>}
      </div>}

      {/* Demo Mode Simulation Selector (Explicitly Labelled) */}
      {apiMode === 'mock' && (
        <div
          className="impact-camera-sim-box"
          aria-label="Demo Mode Emotion API Simulator"
        >
          <div className="impact-camera-sim-title">
            <Sparkles size={15} />
            <span>
              Demo Mode Simulation Preset (Tests all{' '}
              <code>POST /api/emotion</code> response states)
            </span>
          </div>

          <div className="impact-camera-sim-controls">
            <select
              value={selectedMockScenario}
              onChange={(e) => setSelectedMockScenario(e.target.value)}
              aria-label="Select simulated emotion response"
              className="impact-select"
            >
              {MOCK_EMOTION_SCENARIOS.map((sc) => (
                <option key={sc.id} value={sc.id}>
                  {sc.label}
                </option>
              ))}
            </select>

            <button
              type="button"
              className="btn btn-primary btn-sm"
              disabled={analyzing}
              onClick={() =>
                handleAnalyzeBlob(
                  new Blob(['simulated-demo-image'], { type: 'image/jpeg' })
                )
              }
              data-testid="run-simulated-emotion-btn"
            >
              {analyzing ? 'Estimating...' : 'Run Simulated Check-In'}
            </button>
          </div>
        </div>
      )}

      {/* Webcam Preview Box */}
      <div className="camera-frame">
        <video
          ref={videoRef}
          autoPlay
          playsInline
          muted
          onClick={event => {
            const rect = event.currentTarget.getBoundingClientRect();
            const vw = event.currentTarget.videoWidth, vh = event.currentTarget.videoHeight;
            if (!vw || !vh) return;
            const displayedScale = Math.min(rect.width / vw, rect.height / vh);
            const x = (event.clientX - rect.left - (rect.width - vw * displayedScale) / 2) / displayedScale;
            const y = (event.clientY - rect.top - (rect.height - vh * displayedScale) / 2) / displayedScale;
            if (x >= 0 && x <= vw && y >= 0 && y <= vh) {
              selectionRef.current = [x / vw, y / vh];
            }
          }}
          onLoadedMetadata={() => {
            setIsVideoReady(true);
            if (videoRef.current) {
              videoRef.current.play().catch(() => {});
            }
          }}
          onCanPlay={() => setIsVideoReady(true)}
          style={{
            width: '100%',
            height: '100%',
            objectFit: 'contain',
            display: cameraActive ? 'block' : 'none',
          }}
        />
        {!cameraActive && (
          <div className="impact-camera-idle">
            <Camera size={38} style={{ margin: '0 auto 0.75rem', opacity: 0.85 }} />
            <p className="impact-camera-idle-title">
              Webcam is currently off
            </p>
            <p className="impact-camera-idle-sub">
              Start your camera to capture a snapshot, or upload a photo file below.
            </p>
          </div>
        )}
        <canvas ref={canvasRef} style={{ display: 'none' }} />
      </div>

      {/* Camera Action Buttons */}
      <div className="impact-camera-actions">
        {!cameraActive ? (
          <button
            type="button"
            className="btn btn-primary"
            onClick={startCamera}
          >
            <Camera size={17} />
            <span>Enable Camera</span>
          </button>
        ) : (
          <>
            <button
              type="button"
              className="btn btn-primary"
              disabled={analyzing || !isVideoReady}
              onClick={captureSnapshot}
            >
              <RefreshCw size={16} />
              <span>
                {analyzing
                  ? 'Analyzing Frame...'
                  : !isVideoReady
                  ? 'Starting Camera...'
                  : 'Capture & Estimate Mood'}
              </span>
            </button>
            <button
              type="button"
              className="btn btn-secondary"
              onClick={stopCamera}
            >
              <CameraOff size={16} />
              <span>Turn Off Camera</span>
            </button>
          </>
        )}

        <label
          className="btn btn-secondary"
          style={{ cursor: 'pointer' }}
        >
          <Upload size={16} />
          <span>Upload Photo</span>
          <input
            type="file"
            accept="image/jpeg,image/png,image/webp"
            onChange={handleFileUpload}
            style={{ display: 'none' }}
          />
        </label>
      </div>

      {/* Camera / Permission Error */}
      {cameraError && (
        <div
          role="alert"
          className="impact-alert impact-alert--warning"
        >
          <AlertCircle size={18} style={{ flexShrink: 0, marginTop: '2px' }} />
          <div>
            <strong>Camera Notice</strong>
            <p style={{ margin: '0.2rem 0 0', fontSize: '0.9rem' }}>
              {cameraError}
            </p>
          </div>
        </div>
      )}

      {/* Estimation Result Banner */}
      {result && (
        <div
          className={`impact-emotion-result ${
            detectedMoodMeta ? 'detected' : 'manual-needed'
          }`}
          data-testid="emotion-result-banner"
        >
          <div className="impact-emotion-result-row">
            <div>
              <div className="impact-emotion-status-line">
                {detectedMoodMeta ? (
                  <CheckCircle2 size={18} color="#171717" />
                ) : (
                  <AlertCircle size={18} color="#8B3252" />
                )}
                <strong>
                  {detectedMoodMeta
                    ? (result.status === 'low_confidence'
                        ? `AI Suggestion: ${detectedMoodMeta.label}`
                        : `Estimated Mood: ${detectedMoodMeta.label}`)
                    : 'Manual Mood Selection Needed'}
                </strong>
                {result.confidence > 0 && (
                  <span className="badge badge-neutral">
                    Confidence: {Math.round(result.confidence * 100)}%
                  </span>
                )}
              </div>
              <p className="impact-emotion-note">{result.note}</p>
            </div>

            {!detectedMoodMeta && (
              <button
                type="button"
                className="btn btn-primary btn-sm"
                onClick={onSwitchToManual}
              >
                Choose Mood Manually
              </button>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
