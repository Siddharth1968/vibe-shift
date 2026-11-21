import React from 'react';
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import CameraPanel from '../components/CameraPanel';
import * as emotionService from '../services/emotionService';

describe('CameraPanel Component', () => {
  let mockStream;
  let mockTrack;

  beforeEach(() => {
    mockTrack = {
      stop: vi.fn(),
      kind: 'video',
    };
    mockStream = {
      getTracks: vi.fn(() => [mockTrack]),
    };

    // Mock navigator.mediaDevices.getUserMedia
    Object.defineProperty(global.navigator, 'mediaDevices', {
      value: {
        getUserMedia: vi.fn().mockResolvedValue(mockStream),
      },
      writable: true,
      configurable: true,
    });

    // Mock HTMLMediaElement.prototype.play
    window.HTMLMediaElement.prototype.play = vi.fn().mockResolvedValue(undefined);
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  it('renders webcam idle placeholder when initially loaded', () => {
    render(<CameraPanel apiMode="live" />);
    expect(screen.getByText('Webcam is currently off')).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /enable camera/i })).toBeInTheDocument();
    expect(screen.getByText(/upload photo/i)).toBeInTheDocument();
  });

  it('starts webcam, attaches stream to video.srcObject, and calls play()', async () => {
    const { container } = render(<CameraPanel apiMode="live" />);
    const video = container.querySelector('video');
    expect(video).toBeInTheDocument();
    expect(video).toHaveStyle({ display: 'none' });

    const enableBtn = screen.getByRole('button', { name: /enable camera/i });
    fireEvent.click(enableBtn);

    await waitFor(() => {
      expect(navigator.mediaDevices.getUserMedia).toHaveBeenCalled();
      expect(video.srcObject).toBe(mockStream);
      expect(video).toHaveStyle({ display: 'block' });
      expect(window.HTMLMediaElement.prototype.play).toHaveBeenCalled();
    });

    expect(screen.getByRole('button', { name: /turn off camera/i })).toBeInTheDocument();
  });

  it('disables capture button until video metadata is loaded with non-zero dimensions', async () => {
    const { container } = render(<CameraPanel apiMode="live" />);
    const video = container.querySelector('video');

    fireEvent.click(screen.getByRole('button', { name: /enable camera/i }));

    await waitFor(() => {
      expect(video.srcObject).toBe(mockStream);
    });

    // Before metadata is loaded, button should indicate starting camera and be disabled
    const captureBtn = screen.getByRole('button', { name: /starting camera/i });
    expect(captureBtn).toBeDisabled();

    // Trigger metadata loaded
    Object.defineProperty(video, 'videoWidth', { value: 640, configurable: true });
    Object.defineProperty(video, 'videoHeight', { value: 480, configurable: true });
    Object.defineProperty(video, 'readyState', { value: 4, configurable: true });

    fireEvent.loadedMetadata(video);

    await waitFor(() => {
      expect(screen.getByRole('button', { name: /capture & estimate mood/i })).not.toBeDisabled();
    });
  });

  it('stops camera, stops all media tracks, and resets video.srcObject', async () => {
    const { container } = render(<CameraPanel apiMode="live" />);
    const video = container.querySelector('video');

    fireEvent.click(screen.getByRole('button', { name: /enable camera/i }));

    await waitFor(() => {
      expect(video.srcObject).toBe(mockStream);
    });

    fireEvent.click(screen.getByRole('button', { name: /turn off camera/i }));

    expect(mockTrack.stop).toHaveBeenCalled();
    expect(video.srcObject).toBeNull();
    expect(video).toHaveStyle({ display: 'none' });
    expect(screen.getByText('Webcam is currently off')).toBeInTheDocument();
  });

  it('handles capture snapshot and sends to emotion service', async () => {
    const mockEstimate = vi.spyOn(emotionService, 'estimateEmotionFromImage').mockResolvedValue({
      emotion: 'happy',
      confidence: 0.85,
      status: 'success',
      note: 'Facial-expression estimates are approximate.',
      requiresManualSelection: false,
    });

    const onDetected = vi.fn();
    const { container } = render(<CameraPanel apiMode="live" onEmotionDetected={onDetected} />);
    const video = container.querySelector('video');
    const canvas = container.querySelector('canvas');

    // Mock canvas methods
    const mockCtx = {
      drawImage: vi.fn(),
    };
    canvas.getContext = vi.fn(() => mockCtx);
    canvas.toBlob = vi.fn((callback) => {
      callback(new Blob(['mock-img'], { type: 'image/jpeg' }));
    });

    fireEvent.click(screen.getByRole('button', { name: /enable camera/i }));

    await waitFor(() => {
      expect(video.srcObject).toBe(mockStream);
    });

    Object.defineProperty(video, 'videoWidth', { value: 640, configurable: true });
    Object.defineProperty(video, 'videoHeight', { value: 480, configurable: true });
    Object.defineProperty(video, 'readyState', { value: 4, configurable: true });
    fireEvent.loadedMetadata(video);

    const captureBtn = await screen.findByRole('button', { name: /capture & estimate mood/i });
    fireEvent.click(captureBtn);

    await waitFor(() => {
      expect(mockCtx.drawImage).toHaveBeenCalledWith(video, 0, 0, 640, 480);
      expect(mockEstimate).toHaveBeenCalled();
      expect(onDetected).toHaveBeenCalledWith(expect.objectContaining({ emotion: 'happy' }));
    });

    expect(screen.getByText(/estimated mood: happy/i)).toBeInTheDocument();
  });

  it('displays single CleanBTNVit model and switches inference mode while retaining the existing camera stream', async () => {
    render(<CameraPanel apiMode="live" />);
    expect(screen.getByTestId('active-ai-model-status')).toHaveTextContent(/CleanBTNVit \(Phase 12\)/i);
    expect(screen.queryByLabelText('AI Model')).not.toBeInTheDocument();
    fireEvent.click(screen.getByRole('button', { name: /enable camera/i }));
    await waitFor(() => expect(navigator.mediaDevices.getUserMedia).toHaveBeenCalledTimes(1));
    expect(screen.getByLabelText('Inference Mode')).toHaveValue('single_view');
    fireEvent.change(screen.getByLabelText('Inference Mode'), { target: { value: 'horizontal_flip' } });
    expect(screen.getByLabelText('Inference Mode')).toHaveValue('horizontal_flip');
    expect(mockTrack.stop).not.toHaveBeenCalled();
    expect(navigator.mediaDevices.getUserMedia).toHaveBeenCalledTimes(1);
  });

  it('uploads a photo with snapshot=true and forwards optional horizontal-flip accuracy mode', async () => {
    const estimate = vi.spyOn(emotionService, 'estimateEmotionFromImage').mockResolvedValue({
      emotion: 'happy',
      expression: 'happy',
      confidence: 0.91,
      status: 'success',
      note: 'This is only an estimate of facial expression, not proof of feelings.',
      modelName: 'CleanBTNVit (Phase 12)',
      inferenceMode: 'horizontal_flip',
      inferenceMs: 52.1,
      requiresManualSelection: false,
    });
    const onDetected = vi.fn();
    const { container } = render(<CameraPanel apiMode="live" onEmotionDetected={onDetected} />);
    fireEvent.change(screen.getByLabelText('Inference Mode'), { target: { value: 'horizontal_flip' } });
    const fileInput = container.querySelector('input[type="file"]');
    const photo = new File(['fake-jpg'], 'portrait.jpg', { type: 'image/jpeg' });
    fireEvent.change(fileInput, { target: { files: [photo] } });
    await waitFor(() => expect(estimate).toHaveBeenCalledTimes(1));
    expect(estimate.mock.calls[0][1]).toEqual(
      expect.objectContaining({
        apiMode: 'live',
        snapshot: true,
        horizontalFlip: true,
      })
    );
    expect(onDetected).toHaveBeenCalledWith(expect.objectContaining({ emotion: 'happy' }));
  });

  it('releases a camera permission result arriving after unmount', async () => {
    let resolvePermission;
    navigator.mediaDevices.getUserMedia.mockImplementation(() => new Promise(resolve => { resolvePermission = resolve; }));
    const { unmount } = render(<CameraPanel apiMode="live" />);
    fireEvent.click(screen.getByRole('button', { name: /enable camera/i }));
    unmount();
    resolvePermission(mockStream);
    await waitFor(() => expect(mockTrack.stop).toHaveBeenCalled());
  });

  it('preserves mood confirmation when consecutive estimates give the same mood', async () => {
    vi.spyOn(emotionService, 'estimateEmotionFromImage').mockResolvedValue({
      emotion:'happy',confidence:.85,status:'success',note:'Estimated expression',requiresManualSelection:false,
    });
    const onDetected = vi.fn();
    const { container } = render(<CameraPanel apiMode="live" onEmotionDetected={onDetected} />);
    const video = container.querySelector('video'), canvas = container.querySelector('canvas');
    canvas.getContext = vi.fn(() => ({drawImage:vi.fn()}));
    canvas.toBlob = vi.fn(callback => callback(new Blob(['frame'],{type:'image/jpeg'})));
    fireEvent.click(screen.getByRole('button', { name: /enable camera/i }));
    await waitFor(() => expect(video.srcObject).toBe(mockStream));
    Object.defineProperty(video,'videoWidth',{value:640,configurable:true});
    Object.defineProperty(video,'videoHeight',{value:480,configurable:true});
    Object.defineProperty(video,'readyState',{value:4,configurable:true});
    fireEvent.loadedMetadata(video);
    fireEvent.click(screen.getByRole('button',{name:/capture & estimate mood/i}));
    await waitFor(() => expect(onDetected).toHaveBeenCalledTimes(1));
    await waitFor(() => expect(screen.getByRole('button',{name:/capture & estimate mood/i})).toBeEnabled());
    fireEvent.click(screen.getByRole('button',{name:/capture & estimate mood/i}));
    await waitFor(() => expect(emotionService.estimateEmotionFromImage).toHaveBeenCalledTimes(2));
    expect(onDetected).toHaveBeenCalledTimes(1);
  });

  it.each([[1280,720],[1920,1080]])('bounds capture and selection for %i by %i and discards stale responses', async (width,height) => {
    let completeRequest;
    const estimate = vi.spyOn(emotionService,'estimateEmotionFromImage').mockImplementation(
      () => new Promise(resolve => { completeRequest = resolve; }));
    const onDetected = vi.fn();
    const { container } = render(<CameraPanel apiMode="live" onEmotionDetected={onDetected} />);
    fireEvent.click(screen.getByLabelText('Live expression scanning'));
    const video=container.querySelector('video'),canvas=container.querySelector('canvas');
    video.getBoundingClientRect=()=>({width:640,height:480,left:0,top:0});
    const drawImage=vi.fn();
    canvas.getContext=vi.fn(() => ({drawImage}));
    canvas.toBlob=vi.fn(callback => callback(new Blob(['frame'],{type:'image/jpeg'})));
    fireEvent.click(screen.getByRole('button',{name:/enable camera/i}));
    await waitFor(() => expect(video.srcObject).toBe(mockStream));
    Object.defineProperty(video,'videoWidth',{value:width,configurable:true});
    Object.defineProperty(video,'videoHeight',{value:height,configurable:true});
    fireEvent.click(video,{clientX:320,clientY:240});
    Object.defineProperty(video,'readyState',{value:4,configurable:true});
    fireEvent.loadedMetadata(video);
    fireEvent.click(screen.getByRole('button',{name:/capture & estimate mood/i}));
    await waitFor(() => expect(estimate).toHaveBeenCalledTimes(1));
    expect(drawImage).toHaveBeenCalledWith(video,0,0,1280,720);
    expect(estimate.mock.calls[0][1].selection).toEqual([640,360]);
    expect(canvas.toBlob).toHaveBeenCalledWith(expect.any(Function),'image/jpeg',.85);
    fireEvent.change(screen.getByLabelText('Inference Mode'),{target:{value:'horizontal_flip'}});
    completeRequest({emotion:'happy',confidence:.9,status:'success',note:'old response'});
    await waitFor(() => expect(screen.getByRole('button',{name:/capture & estimate mood/i})).toBeEnabled());
    expect(onDetected).not.toHaveBeenCalled();
    expect(screen.queryByTestId('emotion-result-banner')).not.toBeInTheDocument();
    expect(mockTrack.stop).not.toHaveBeenCalled();
  });

  it('displays raw expression separately from smoothing and music mapping in debug mode', async () => {
    vi.spyOn(emotionService,'estimateEmotionFromImage').mockResolvedValue({
      emotion:null,expression:'fear',rawExpression:'fear',rawConfidence:.82,
      confidence:.78,status:'success',note:'Choose a music mood manually',debug:{output:'output_VIT'},
    });
    const { container }=render(<CameraPanel apiMode="live" />);
    fireEvent.click(screen.getByLabelText('Live expression scanning'));
    fireEvent.click(screen.getByLabelText('Local prediction debug'));
    const video=container.querySelector('video'),canvas=container.querySelector('canvas');
    canvas.getContext=vi.fn(() => ({drawImage:vi.fn()}));
    canvas.toBlob=vi.fn(callback => callback(new Blob(['frame'],{type:'image/jpeg'})));
    fireEvent.click(screen.getByRole('button',{name:/enable camera/i}));
    await waitFor(() => expect(video.srcObject).toBe(mockStream));
    Object.defineProperty(video,'videoWidth',{value:640,configurable:true});
    Object.defineProperty(video,'videoHeight',{value:480,configurable:true});
    Object.defineProperty(video,'readyState',{value:4,configurable:true});
    fireEvent.loadedMetadata(video);
    fireEvent.click(screen.getByRole('button',{name:/capture & estimate mood/i}));
    const debug=await screen.findByTestId('live-prediction-debug');
    expect(debug).toHaveTextContent('Raw prediction — fear (82.0%)');
    expect(debug).toHaveTextContent('Stabilized expression — fear');
    expect(debug).toHaveTextContent('Music mood — choose manually');
  });

  it('displays AI suggestion banner when prediction returns low_confidence with valid mood', async () => {
    vi.spyOn(emotionService, 'estimateEmotionFromImage').mockResolvedValue({
      emotion: 'neutral',
      confidence: 0.28,
      status: 'low_confidence',
      note: 'Facial expression confidence is low (28%). AI suggests \'neutral\'.',
      requiresManualSelection: true,
    });

    const onDetected = vi.fn();
    const { container } = render(<CameraPanel apiMode="live" onEmotionDetected={onDetected} />);
    const video = container.querySelector('video');
    const canvas = container.querySelector('canvas');

    const mockCtx = { drawImage: vi.fn() };
    canvas.getContext = vi.fn(() => mockCtx);
    canvas.toBlob = vi.fn((callback) => {
      callback(new Blob(['mock-img'], { type: 'image/jpeg' }));
    });

    fireEvent.click(screen.getByRole('button', { name: /enable camera/i }));
    await waitFor(() => {
      expect(video.srcObject).toBe(mockStream);
    });

    Object.defineProperty(video, 'videoWidth', { value: 640, configurable: true });
    Object.defineProperty(video, 'videoHeight', { value: 480, configurable: true });
    Object.defineProperty(video, 'readyState', { value: 4, configurable: true });
    fireEvent.loadedMetadata(video);

    const captureBtn = await screen.findByRole('button', { name: /capture & estimate mood/i });
    fireEvent.click(captureBtn);

    await waitFor(() => {
      expect(onDetected).toHaveBeenCalledWith(expect.objectContaining({
        emotion: 'neutral',
        status: 'low_confidence',
      }));
    });

    expect(screen.getByText(/ai suggestion: neutral/i)).toBeInTheDocument();
    expect(screen.getByText(/confidence: 28%/i)).toBeInTheDocument();
  });
});

