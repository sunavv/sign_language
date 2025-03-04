"use client";

import React, { useState, useRef, useEffect } from 'react';
import { Keyboard, Trash2, WifiOff, Hand, Upload, Camera, Image as ImageIcon, Eye, EyeOff, Volume2 } from 'lucide-react';
import { Switch } from '@/components/ui/switch';
import { Label } from '@/components/ui/label';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Button } from '@/components/ui/button';

const SignLanguageTranslator = () => {
  const [state, setState] = useState({
    current_word: '',
    words: [],
    current_gesture: null,
    landmarks: null
  });
  
  const [connectionStatus, setConnectionStatus] = useState({
    isConnected: false,
    error: null,
    retryCount: 0,
    isConnecting: false
  });
  
  const [activeCameraId, setActiveCameraId] = useState(null);
  const [availableCameras, setAvailableCameras] = useState([]);
  const [selectedImage, setSelectedImage] = useState(null);
  const [isProcessingImage, setIsProcessingImage] = useState(false);
  const [showLandmarks, setShowLandmarks] = useState(true);
  
  // Video upload states
  const [selectedVideo, setSelectedVideo] = useState(null);
  const [isProcessingVideo, setIsProcessingVideo] = useState(false);
  const [videoFile, setVideoFile] = useState(null);
  const [videoProcessingStatus, setVideoProcessingStatus] = useState('');
  
  // Audio caption states
  const [selectedAudio, setSelectedAudio] = useState(null);
  const [audioFile, setAudioFile] = useState(null);
  const [isProcessingAudio, setIsProcessingAudio] = useState(false);
  const [audioProcessingStatus, setAudioProcessingStatus] = useState('');
  const [audioCaption, setAudioCaption] = useState('');
  
  const videoRef = useRef(null);
  const streamRef = useRef(null);
  const wsRef = useRef(null);
  const canvasRef = useRef(null);
  const landmarksCanvasRef = useRef(null);
  const fileInputRef = useRef(null);
  const audioInputRef = useRef(null);
  const audioRef = useRef(null);
  const animationFrameRef = useRef(null);
  const reconnectTimeoutRef = useRef(null);
  const processingIntervalRef = useRef(null);
  
  const MAX_RETRY_ATTEMPTS = 5;
  const RETRY_DELAY = 3000;
  const FRAME_PROCESSING_INTERVAL = 100; // Process frames every 100ms for better performance
  const WS_URL = 'ws://127.0.0.1:5000/ws';
  const API_URL = 'http://127.0.0.1:5000';

  // Fetch available cameras
  const getAvailableCameras = async () => {
    try {
      const devices = await navigator.mediaDevices.enumerateDevices();
      const videoDevices = devices.filter(device => device.kind === 'videoinput');
      setAvailableCameras(videoDevices);
      
      // Set the default camera if we have devices
      if (videoDevices.length > 0 && !activeCameraId) {
        setActiveCameraId(videoDevices[0].deviceId);
      }
    } catch (error) {
      console.error("Error getting camera devices:", error);
    }
  };

  // Connect to WebSocket
  const connectWebSocket = () => {
    if (wsRef.current?.readyState === WebSocket.OPEN || connectionStatus.isConnecting) return;

    try {
      setConnectionStatus(prev => ({
        ...prev,
        isConnecting: true,
        error: null
      }));

      console.log('Attempting WebSocket connection...');
      const ws = new WebSocket(WS_URL);
      wsRef.current = ws;

      ws.onopen = () => {
        console.log('WebSocket Connected');
        setConnectionStatus({
          isConnected: true,
          error: null,
          retryCount: 0,
          isConnecting: false
        });
        startFrameCapture();
      };

      ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          setState(data);
          if (data.landmarks && showLandmarks) {
            drawLandmarks(data.landmarks);
          } else if (!showLandmarks && landmarksCanvasRef.current) {
            // Clear landmarks if they're hidden
            const ctx = landmarksCanvasRef.current.getContext('2d');
            ctx.clearRect(0, 0, landmarksCanvasRef.current.width, landmarksCanvasRef.current.height);
          }
        } catch (e) {
          console.error('Error parsing WebSocket message:', e);
          handleError('Invalid data received from server');
        }
      };

      ws.onerror = (error) => {
        console.error('WebSocket Error:', error);
        handleError('Connection failed. Please check if the server is running.');
      };

      ws.onclose = (event) => {
        console.log('WebSocket Disconnected:', event.code, event.reason);
        handleDisconnect();
      };
    } catch (error) {
      console.error('Error creating WebSocket:', error);
      handleError('Failed to initialize connection');
    }
  };

  const handleError = (errorMessage) => {
    setConnectionStatus(prev => ({
      isConnected: false,
      error: errorMessage,
      retryCount: prev.retryCount + 1,
      isConnecting: false
    }));
    stopFrameCapture();
  };

  const handleDisconnect = () => {
    setConnectionStatus(prev => ({
      isConnected: false,
      error: prev.error || 'Connection lost',
      retryCount: prev.retryCount,
      isConnecting: false
    }));
    stopFrameCapture();

    // Attempt reconnection if we haven't exceeded max retries
    if (connectionStatus.retryCount < MAX_RETRY_ATTEMPTS) {
      console.log(`Attempting reconnection in ${RETRY_DELAY}ms...`);
      reconnectTimeoutRef.current = setTimeout(connectWebSocket, RETRY_DELAY);
    }
  };

  // Draw landmarks on the canvas
  const drawLandmarks = (landmarks) => {
    if (!landmarksCanvasRef.current || !showLandmarks) return;
    
    const canvas = landmarksCanvasRef.current;
    const ctx = canvas.getContext('2d');
    const { width, height } = canvas;
    
    // Clear previous drawing
    ctx.clearRect(0, 0, width, height);
    
    if (!landmarks || landmarks.length === 0) return;
    
    // Define connections between landmarks for hand skeleton
    const connections = [
      // Thumb
      [0, 1], [1, 2], [2, 3], [3, 4],
      // Index finger
      [0, 5], [5, 6], [6, 7], [7, 8],
      // Middle finger
      [0, 9], [9, 10], [10, 11], [11, 12],
      // Ring finger
      [0, 13], [13, 14], [14, 15], [15, 16],
      // Pinky
      [0, 17], [17, 18], [18, 19], [19, 20],
      // Palm
      [0, 5], [5, 9], [9, 13], [13, 17]
    ];
    
    // Draw connections (skeleton)
    ctx.beginPath();
    ctx.strokeStyle = 'rgba(0, 119, 255, 0.8)';
    ctx.lineWidth = 3;
    
    connections.forEach(([i, j]) => {
      if (landmarks[i] && landmarks[j]) {
        ctx.moveTo(landmarks[i].x * width, landmarks[i].y * height);
        ctx.lineTo(landmarks[j].x * width, landmarks[j].y * height);
      }
    });
    ctx.stroke();
    
    // Draw landmarks
    landmarks.forEach((point, index) => {
      ctx.beginPath();
      // Different colors for different finger landmarks
      if (index === 0) {
        // Wrist
        ctx.fillStyle = 'rgba(255, 0, 0, 0.7)';
        ctx.arc(point.x * width, point.y * height, 8, 0, 2 * Math.PI);
      } else if (index % 4 === 0) {
        // Fingertips
        ctx.fillStyle = 'rgba(0, 255, 0, 0.7)';
        ctx.arc(point.x * width, point.y * height, 6, 0, 2 * Math.PI);
      } else {
        // Other joints
        ctx.fillStyle = 'rgba(0, 119, 255, 0.5)';
        ctx.arc(point.x * width, point.y * height, 4, 0, 2 * Math.PI);
      }
      ctx.fill();
    });
  };

  const startFrameCapture = () => {
    if (!canvasRef.current || !videoRef.current) return;

    // Clear any existing interval
    if (processingIntervalRef.current) {
      clearInterval(processingIntervalRef.current);
    }

    // Use interval instead of requestAnimationFrame for more controlled frame rate
    processingIntervalRef.current = setInterval(() => {
      try {
        if (videoRef.current && wsRef.current?.readyState === WebSocket.OPEN) {
          const context = canvasRef.current.getContext('2d');
          context.drawImage(videoRef.current, 0, 0, canvasRef.current.width, canvasRef.current.height);
          
          // Lower quality for better performance
          const frameData = canvasRef.current.toDataURL('image/jpeg', 0.6);
          wsRef.current.send(frameData);
        }
      } catch (error) {
        console.error('Error capturing frame:', error);
        handleError('Error capturing video frame');
        stopFrameCapture();
      }
    }, FRAME_PROCESSING_INTERVAL);
  };

  const stopFrameCapture = () => {
    if (processingIntervalRef.current) {
      clearInterval(processingIntervalRef.current);
      processingIntervalRef.current = null;
    }
    
    if (animationFrameRef.current) {
      cancelAnimationFrame(animationFrameRef.current);
      animationFrameRef.current = null;
    }
  };

  const handleRetryConnection = () => {
    // Clear any existing timeouts
    if (reconnectTimeoutRef.current) {
      clearTimeout(reconnectTimeoutRef.current);
      reconnectTimeoutRef.current = null;
    }

    // Close existing connection if any
    if (wsRef.current) {
      wsRef.current.close();
      wsRef.current = null;
    }

    setConnectionStatus(prev => ({
      ...prev,
      retryCount: 0,
      error: null,
      isConnecting: false
    }));

    // Attempt new connection
    connectWebSocket();
  };

  // Initialize camera with selected device
  const initializeCamera = async (deviceId = null) => {
    try {
      // Stop any existing stream
      if (streamRef.current) {
        streamRef.current.getTracks().forEach(track => track.stop());
      }

      // Create canvas for frame capture if it doesn't exist
      if (!canvasRef.current) {
        const canvas = document.createElement('canvas');
        canvas.width = 640;
        canvas.height = 480;
        canvasRef.current = canvas;
      }

      // Get camera access with specific device if provided
      const constraints = {
        video: deviceId ? 
          { deviceId: { exact: deviceId }, width: 640, height: 480, frameRate: { ideal: 10 } } : 
          { width: 640, height: 480, frameRate: { ideal: 10 } }
      };

      const stream = await navigator.mediaDevices.getUserMedia(constraints);

      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        streamRef.current = stream;
        
        // Wait for video to be ready
        if (videoRef.current.readyState >= 2) {
          connectWebSocket();
        } else {
          videoRef.current.onloadeddata = () => {
            connectWebSocket();
          };
        }
      }
    } catch (error) {
      console.error("Error setting up camera:", error);
      handleError(error.message || "Failed to access camera");
    }
  };

  // Handle switching cameras
  const handleCameraSwitch = (deviceId) => {
    setActiveCameraId(deviceId);
    initializeCamera(deviceId);
  };

  // Handle file selection for image processing
  const handleFileChange = (e) => {
    const file = e.target.files[0];
    if (!file) return;
    
    // Check if the file is an image
    if (!file.type.startsWith('image/')) {
      alert('Please select an image file');
      return;
    }
    
    const reader = new FileReader();
    reader.onload = (event) => {
      setSelectedImage(event.target.result);
    };
    reader.readAsDataURL(file);
  };

  // Process selected image
  const processImage = async () => {
    if (!selectedImage) return;
    
    setIsProcessingImage(true);
    
    try {
      const response = await fetch(`${API_URL}/process_image`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ image_data: selectedImage }),
      });
      
      if (!response.ok) {
        throw new Error(`Server returned ${response.status}`);
      }
      
      const data = await response.json();

      // Ensure data has the expected structure
      if (data && data.text !== undefined) {
        setState(prev => ({
          ...prev,
          words: [...prev.words, data.text],
          current_word: data.current_word || '',
          landmarks: data.landmarks || null
        }));
      } else {
        console.error('Unexpected response structure:', data);
      }

      // Draw landmarks if available and landmarks are enabled
      if (data.landmarks && showLandmarks) {
        drawLandmarks(data.landmarks);
      }
      
    } catch (error) {
      console.error('Error processing image:', error);
      alert('Failed to process image. Please try again.');
    } finally {
      setIsProcessingImage(false);
    }
  };

  const handleVideoChange = (e) => {
    const file = e.target.files[0];
    if (!file) return;
  
    // Check if the file is a video
    if (!file.type.startsWith('video/')) {
      alert('Please select a video file');
      return;
    }
  
    // Clear previous results
    setState({
      current_word: '',
      words: [],
      current_gesture: null,
      landmarks: null
    });
  
    setVideoFile(file); // Store the actual file
    setSelectedVideo(URL.createObjectURL(file)); // For preview
  };

  const processVideo = async () => {
    if (!videoFile) return;
  
    setIsProcessingVideo(true);
    setState(prev => ({...prev, words: []})); // Clear previous results
    setVideoProcessingStatus('Processing video... This may take some time.');
  
    try {
      // Create a FormData object to send the file
      const formData = new FormData();
      formData.append('video', videoFile);
  
      const response = await fetch(`${API_URL}/process_video`, {
        method: 'POST',
        body: formData, // Send the actual file
      });
  
      if (!response.ok) {
        const errorText = await response.text();
        let errorMessage;
        try {
          const errorData = JSON.parse(errorText);
          errorMessage = errorData.detail || `Server returned ${response.status}`;
        } catch (e) {
          errorMessage = `Server returned ${response.status}: ${errorText.substring(0, 100)}`;
        }
        throw new Error(errorMessage);
      }
  
      const data = await response.json();
  
      // Ensure data has the expected structure
      if (data && Array.isArray(data.words)) {
        setState(prev => ({
          ...prev,
          words: data.words,
          current_word: data.current_word || '',
          landmarks: data.landmarks || null
        }));
        
        setVideoProcessingStatus(
        //   data.words.length > 0 
        //     ? `Processing complete. Detected ${data.words.length} word(s).` 
        //     : 'Processing complete. No words detected.'
        );
      } else {
        console.error('Unexpected response structure:', data);
        setVideoProcessingStatus('Processing complete, but received unexpected data format.');
      }
  
    } catch (error) {
      console.error('Error processing video:', error);
      setVideoProcessingStatus(`Error: ${error.message}`);
      alert(`Failed to process video: ${error.message}`);
    } finally {
      setIsProcessingVideo(false);
    }
  };

  // Handle audio file selection
  const handleAudioChange = (e) => {
    const file = e.target.files[0];
    if (!file) return;
    
    // Check if the file is an audio
    if (!file.type.startsWith('audio/')) {
      alert('Please select an audio file');
      return;
    }
    
    // Clear previous results
    setAudioCaption('');
    setAudioProcessingStatus('');
    
    setAudioFile(file); // Store the actual file
    setSelectedAudio(URL.createObjectURL(file)); // For preview
  };

  // Process selected audio for captioning
  const processAudio = async () => {
    if (!audioFile) return;
    
    setIsProcessingAudio(true);
    setAudioProcessingStatus('Processing audio... This may take some time.');
    
    try {
      // Create a FormData object to send the file
      const formData = new FormData();
      formData.append('audio', audioFile);
      
      const response = await fetch(`${API_URL}/process_audio`, {
        method: 'POST',
        body: formData,
      });
      
      if (!response.ok) {
        const errorText = await response.text();
        let errorMessage;
        try {
          const errorData = JSON.parse(errorText);
          errorMessage = errorData.detail || `Server returned ${response.status}`;
        } catch (e) {
          errorMessage = `Server returned ${response.status}: ${errorText.substring(0, 100)}`;
        }
        throw new Error(errorMessage);
      }
      
      const data = await response.json();
      
      // Update the caption with the transcribed text
      if (data && data.caption) {
        setAudioCaption(data.caption);
        setAudioProcessingStatus('Processing complete.');
        
        // Add the caption to the sign language words
        setState(prev => ({
          ...prev,
          words: [...prev.words, ...data.caption.split(' ')],
        }));
      } else {
        console.error('Unexpected response structure:', data);
        setAudioProcessingStatus('Processing complete, but received unexpected data format.');
      }
      
    } catch (error) {
      console.error('Error processing audio:', error);
      setAudioProcessingStatus(`Error: ${error.message}`);
      alert(`Failed to process audio: ${error.message}`);
    } finally {
      setIsProcessingAudio(false);
    }
  };

  const handleClear = async () => {
    try {
      const response = await fetch(`${API_URL}/clear`, {
        method: 'POST'
      });
      
      if (response.ok) {
        setState({
          current_word: '',
          words: [],
          current_gesture: null,
          landmarks: null
        });
        
        // Clear image if one is selected
        setSelectedImage(null);
        setSelectedVideo(null);
        setVideoFile(null);
        setVideoProcessingStatus('');
        
        // Clear audio if one is selected
        setSelectedAudio(null);
        setAudioFile(null);
        setAudioCaption('');
        setAudioProcessingStatus('');
        
        if (fileInputRef.current) {
          fileInputRef.current.value = '';
        }
        
        if (audioInputRef.current) {
          audioInputRef.current.value = '';
        }
        
        // Clear landmarks canvas
        if (landmarksCanvasRef.current) {
          const ctx = landmarksCanvasRef.current.getContext('2d');
          ctx.clearRect(0, 0, landmarksCanvasRef.current.width, landmarksCanvasRef.current.height);
        }
      }
    } catch (error) {
      console.error('Error clearing state:', error);
      alert('Failed to clear text');
    }
  };

  // Toggle landmarks visibility
  const handleToggleLandmarks = () => {
    setShowLandmarks(!showLandmarks);
    
    // Clear landmarks if turning off
    if (showLandmarks && landmarksCanvasRef.current) {
      const ctx = landmarksCanvasRef.current.getContext('2d');
      ctx.clearRect(0, 0, landmarksCanvasRef.current.width, landmarksCanvasRef.current.height);
    }
  };

  // Initialize on component mount
  useEffect(() => {
    getAvailableCameras();
    
    return () => {
      // Cleanup function
      stopFrameCapture();
      
      if (streamRef.current) {
        streamRef.current.getTracks().forEach(track => track.stop());
      }
      
      if (wsRef.current) {
        wsRef.current.close();
        wsRef.current = null;
      }
      
      if (reconnectTimeoutRef.current) {
        clearTimeout(reconnectTimeoutRef.current);
        reconnectTimeoutRef.current = null;
      }
      
      if (processingIntervalRef.current) {
        clearInterval(processingIntervalRef.current);
        processingIntervalRef.current = null;
      }
    };
  }, []);

  // Initialize camera when active camera changes
  useEffect(() => {
    if (activeCameraId) {
      initializeCamera(activeCameraId);
    }
  }, [activeCameraId]);

  // Effect to handle landmark visibility changes
  useEffect(() => {
    if (!showLandmarks && landmarksCanvasRef.current) {
      const ctx = landmarksCanvasRef.current.getContext('2d');
      ctx.clearRect(0, 0, landmarksCanvasRef.current.width, landmarksCanvasRef.current.height);
    } else if (showLandmarks && state.landmarks) {
      drawLandmarks(state.landmarks);
    }
  }, [showLandmarks]);

  return (
    <div className="max-w-4xl mx-auto p-4 w-full">
      {connectionStatus.error && (
        <div className="mb-4 p-4 bg-red-50 border border-red-200 rounded-lg">
          <div className="flex items-center gap-2 text-red-800 font-semibold mb-1">
            <WifiOff className="h-4 w-4" />
            <span>Connection Error</span>
          </div>
          <p className="text-red-600">{connectionStatus.error}</p>
          {connectionStatus.retryCount >= MAX_RETRY_ATTEMPTS && (
            <div className="mt-2">
              <button 
                onClick={handleRetryConnection}
                className="px-3 py-1 text-sm bg-red-100 hover:bg-red-200 text-red-800 rounded-md transition-colors"
                disabled={connectionStatus.isConnecting}
              >
                {connectionStatus.isConnecting ? 'Connecting...' : 'Retry Connection'}
              </button>
            </div>
          )}
        </div>
      )}
      
      <div className="bg-white shadow-lg rounded-lg mb-6 p-4">
        <div className="flex items-center justify-between border-b pb-3 mb-4">
          <div className="flex items-center text-lg font-semibold">
            <Keyboard className="mr-2" />
            Sign Language Recognition
            <div className={`ml-2 h-2 w-2 rounded-full ${connectionStatus.isConnected ? 'bg-green-500' : 'bg-red-500'}`} />
          </div>
          <Button 
            variant="ghost" 
            size="icon"
            onClick={handleClear}
            className="hover:bg-red-100"
          >
            <Trash2 className="h-5 w-5" />
          </Button>
        </div>
        
        <div className="min-h-[120px] bg-gray-100 rounded-lg p-6 shadow-inner mb-4">
          <div className="text-lg">
            {state.words && state.words.length > 0 ? state.words.join(' ') : ''}
            {state.current_word && (
              <span className="text-blue-600">
                {state.words.length > 0 ? ' ' : ''} 
                {state.current_word}
              </span>
            )}
            <span className="animate-pulse">|</span>
          </div>
        </div>
      </div>

      <Tabs defaultValue="live" className="mb-6">
        <TabsList className="mb-4">
          <TabsTrigger value="live" className="flex items-center gap-2">
            <Camera className="h-4 w-4" />
            Live Camera
          </TabsTrigger>
          <TabsTrigger value="image" className="flex items-center gap-2">
            <ImageIcon className="h-4 w-4" />
            Image Upload
          </TabsTrigger>
          <TabsTrigger value="video" className="flex items-center gap-2">
            <Upload className="h-4 w-4" />
            Video Upload
          </TabsTrigger>
          <TabsTrigger value="audio" className="flex items-center gap-2">
            <Volume2 className="h-4 w-4" />
            Audio Caption
          </TabsTrigger>
        </TabsList>
        
        <TabsContent value="live" className="space-y-4">
          <div className="bg-white shadow-lg rounded-lg p-4 mb-4">
            <div className="flex items-center justify-between mb-3">
              <div className="flex items-center text-lg font-semibold">
                <Hand className="mr-2 h-5 w-5" />
                <h3>Live Hand Tracking</h3>
              </div>
              
              <div className="flex items-center space-x-4">
                <div className="flex items-center space-x-2">
                  <Switch 
                    id="landmarks-toggle" 
                    checked={showLandmarks} 
                    onCheckedChange={handleToggleLandmarks}
                  />
                  <Label htmlFor="landmarks-toggle" className="flex items-center cursor-pointer">
                    {showLandmarks ? (
                      <Eye className="h-4 w-4 mr-1" />
                    ) : (
                      <EyeOff className="h-4 w-4 mr-1" />
                    )}
                    <span className="text-sm">
                      {showLandmarks ? 'Hide Landmarks' : 'Show Landmarks'}
                    </span>
                  </Label>
                </div>
                
                {availableCameras.length > 1 && (
                  <div className="flex items-center space-x-2">
                    <span className="text-sm text-gray-600">Camera:</span>
                    <select 
                      className="text-sm border rounded p-1"
                      value={activeCameraId || ''}
                      onChange={(e) => handleCameraSwitch(e.target.value)}
                    >
                      {availableCameras.map(camera => (
                        <option key={camera.deviceId} value={camera.deviceId}>
                          {camera.label || `Camera ${availableCameras.indexOf(camera) + 1}`}
                        </option>
                      ))}
                    </select>
                  </div>
                )}
              </div>
            </div>
            
            <div className="flex justify-center">
              <div className="relative">
                <video
                  ref={videoRef}
                  autoPlay
                  playsInline
                  width="640"
                  height="480"
                  className="rounded-lg"
                  muted
                />
                <canvas
                  ref={landmarksCanvasRef}
                  width="640"
                  height="480"
                  className="absolute top-0 left-0 rounded-lg"
                />
              </div>
            </div>
          </div>
        </TabsContent>
        
        <TabsContent value="image">
          <div className="bg-white shadow-lg rounded-lg p-4 mb-4">
            <div className="flex items-center justify-between mb-3">
              <div className="flex items-center text-lg font-semibold">
                <Upload className="mr-2 h-5 w-5" />
                <h3>Upload Sign Language Image</h3>
              </div>
              
              <div className="flex items-center space-x-2">
                <Switch 
                  id="image-landmarks-toggle" 
                  checked={showLandmarks} 
                  onCheckedChange={handleToggleLandmarks}
                />
                <Label htmlFor="image-landmarks-toggle" className="flex items-center cursor-pointer">
                  {showLandmarks ? (
                    <Eye className="h-4 w-4 mr-1" />
                  ) : (
                    <EyeOff className="h-4 w-4 mr-1" />
                  )}
                  <span className="text-sm">
                    {showLandmarks ? 'Hide Landmarks' : 'Show Landmarks'}
                  </span>
                </Label>
              </div>
            </div>
            
            <div className="border-2 border-dashed border-gray-300 rounded-lg p-6 mb-4 text-center">
              {selectedImage ? (
                <div className="relative">
                  <img 
                    src={selectedImage} 
                    alt="Selected sign language image" 
                    className="max-w-full max-h-[400px] mx-auto rounded-lg"
                  />
                  <canvas
                    ref={landmarksCanvasRef}
                    width="640"
                    height="480"
                    className="absolute top-0 left-0 w-full h-full rounded-lg"
                  />
                </div>
              ) : (
                <div className="py-8">
                  <ImageIcon className="h-12 w-12 mx-auto mb-3 text-gray-400" />
                  <p className="text-gray-600 mb-2">Upload an image with sign language gestures</p>
                  <p className="text-gray-500 text-sm mb-4">Supported formats: JPG, PNG, JPEG</p>
                </div>
              )}
              
              <div className="flex flex-col sm:flex-row justify-center gap-3 mt-4">
                <Button 
                  variant="outline"
                  onClick={() => fileInputRef.current?.click()}
                >
                  <Upload className="h-4 w-4 mr-2" />
                  Select Image
                </Button>
                
                <Button 
                  disabled={!selectedImage || isProcessingImage}
                  onClick={processImage}
                >
                  {isProcessingImage ? 'Processing...' : 'Process Image'}
                </Button>
                
                <input
                  type="file"
                  ref={fileInputRef}
                  onChange={handleFileChange}
                  accept="image/*"
                  className="hidden"
                />
              </div>
            </div>
          </div>
        </TabsContent>

        <TabsContent value="video">
          <div className="bg-white shadow-lg rounded-lg p-4 mb-4">
            <div className="flex items-center justify-between mb-3">
              <div className="flex items-center text-lg font-semibold">
                <Upload className="mr-2 h-5 w-5" />
                <h3>Upload Sign Language Video</h3>
              </div>
              
              <div className="flex items-center space-x-2">
                <Switch 
                  id="video-landmarks-toggle" 
                  checked={showLandmarks} 
                  onCheckedChange={handleToggleLandmarks}
                />
                <Label htmlFor="video-landmarks-toggle" className="flex items-center cursor-pointer">
                  {showLandmarks ? (
                    <Eye className="h-4 w-4 mr-1" />
                  ) : (
                    <EyeOff className="h-4 w-4 mr-1" />
                  )}
                  <span className="text-sm">
                    {showLandmarks ? 'Hide Landmarks' : 'Show Landmarks'}
                  </span>
                </Label>
              </div>
            </div>

            <div className="border-2 border-dashed border-gray-300 rounded-lg p-6 mb-4 text-center">
              {selectedVideo ? (
                <div>
                  <video 
                    controls 
                    className="max-w-full max-h-[400px] mx-auto rounded-lg mb-3"
                  >
                    <source src={selectedVideo} />
                    Your browser does not support the video tag.
                  </video>
                  
                  {videoProcessingStatus && (
                    <div className={`mt-2 p-2 rounded text-sm ${
                      videoProcessingStatus.includes('Error') 
                        ? 'bg-red-50 text-red-700' 
                        : 'bg-blue-50 text-blue-700'
                    }`}>
                      {videoProcessingStatus}
                    </div>
                  )}
                </div>
              ) : (
                <div className="py-8">
                  <Upload className="h-12 w-12 mx-auto mb-3 text-gray-400" />
                  <p className="text-gray-600 mb-2">Upload a video with sign language gestures</p>
                  <p className="text-gray-500 text-sm mb-4">Supported formats: MP4, MOV, AVI</p>
                  <p className="text-amber-600 text-sm">Note: Processing may take some time depending on video length</p>
                </div>
              )}

              <div className="flex flex-col sm:flex-row justify-center gap-3 mt-4">
                <Button 
                  variant="outline"
                  onClick={() => fileInputRef.current?.click()}
                >
                  <Upload className="h-4 w-4 mr-2" />
                  Select Video
                </Button>

                <Button 
                  disabled={!selectedVideo || isProcessingVideo}
                  onClick={processVideo}
                >
                  {isProcessingVideo ? 'Processing...' : 'Process Video'}
                </Button>

                <input
                  type="file"
                  ref={fileInputRef}
                  onChange={handleVideoChange}
                  accept="video/*"
                  className="hidden"
                />
              </div>
            </div>
          </div>
        </TabsContent>
        
        <TabsContent value="audio">
          <div className="bg-white shadow-lg rounded-lg p-4 mb-4">
            <div className="flex items-center justify-between mb-3">
              <div className="flex items-center text-lg font-semibold">
                <Volume2 className="mr-2 h-5 w-5" />
                <h3>Audio to Caption</h3>
              </div>
            </div>
            
            <div className="border-2 border-dashed border-gray-300 rounded-lg p-6 mb-4 text-center">
              {selectedAudio ? (
                <div>
                  <audio 
                    ref={audioRef}
                    controls 
                    className="w-full max-w-md mx-auto mb-4"
                  >
                    <source src={selectedAudio} />
                    Your browser does not support the audio element.
                  </audio>
                  
                  {audioCaption && (
                    <div className="mt-4 p-4 bg-blue-50 rounded-lg text-left">
                      <h4 className="font-semibold mb-2 text-blue-800">Generated Caption:</h4>
                      <p className="text-gray-800">{audioCaption}</p>
                    </div>
                  )}
                  
                  {audioProcessingStatus && (
                    <div className={`mt-2 p-2 rounded text-sm ${
                      audioProcessingStatus.includes('Error') 
                        ? 'bg-red-50 text-red-700' 
                        : 'bg-blue-50 text-blue-700'
                    }`}>
                      {audioProcessingStatus}
                    </div>
                  )}
                </div>
              ) : (
                <div className="py-8">
                  <Volume2 className="h-12 w-12 mx-auto mb-3 text-gray-400" />
                  <p className="text-gray-600 mb-2">Upload an audio file to generate captions</p>
                  <p className="text-gray-500 text-sm mb-4">Supported formats: MP3, WAV, M4A</p>
                  <p className="text-amber-600 text-sm">Note: Processing may take some time depending on audio length</p>
                </div>
              )}
              
              <div className="flex flex-col sm:flex-row justify-center gap-3 mt-4">
                <Button 
                  variant="outline"
                  onClick={() => audioInputRef.current?.click()}
                >
                  <Upload className="h-4 w-4 mr-2" />
                  Select Audio
                </Button>
                
                <Button 
                  disabled={!selectedAudio || isProcessingAudio}
                  onClick={processAudio}
                >
                  {isProcessingAudio ? 'Processing...' : 'Generate Caption'}
                </Button>
                
                <input
                  type="file"
                  ref={audioInputRef}
                  onChange={handleAudioChange}
                  accept="audio/*"
                  className="hidden"
                />
              </div>
            </div>
          </div>
        </TabsContent>
      </Tabs>

      {state.current_gesture && (
        <div className="fixed bottom-4 right-4 bg-blue-50 shadow-lg rounded-lg p-4">
          <div className="text-2xl font-bold text-blue-600">
            {state.current_gesture}
          </div>
          <div className="text-sm text-gray-600">
            Current Gesture
          </div>
        </div>
      )}

      <div className="bg-white shadow-lg rounded-lg p-4">
        <h3 className="font-semibold mb-2">Instructions:</h3>
        <ul className="list-disc pl-5 space-y-1 text-sm text-gray-700">
          <li>Use the <strong>Live Camera</strong> tab to recognize sign language in real-time</li>
          <li>Use the <strong>Image Upload</strong> tab to process sign language from images</li>
          <li>Use the <strong>Video Upload</strong> tab to process sign language from videos</li>
          <li>Use the <strong>Audio Caption</strong> tab to generate text from audio files</li>
          <li>Toggle the <strong>Show/Hide Landmarks</strong> switch to control visualization</li>
          <li>Hold a gesture steady for 1 second to add it to the current word</li>
          <li>Pause for 2 seconds to complete the current word</li>
          <li>Click the trash icon to clear all text</li>
          <li>Blue lines show hand skeleton tracking</li>
          <li>Green dots highlight fingertips</li>
        </ul>
      </div>
    </div>
  );
};

export default SignLanguageTranslator;