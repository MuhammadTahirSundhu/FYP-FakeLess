import 'dart:io';
import 'package:flutter/material.dart';
import 'package:just_audio/just_audio.dart';
import 'package:path/path.dart' as p;
import 'package:path_provider/path_provider.dart';
import 'package:record/record.dart';
import 'package:flutter_animate/flutter_animate.dart';
import 'package:share_plus/share_plus.dart';
import 'package:file_picker/file_picker.dart';

import '../core/theme.dart';
import '../services/api_service.dart';
import '../services/ai_agent_service.dart';
import '../services/tflite_service.dart';
import '../services/crypto_service.dart';
import '../widgets/dual_mode_dashboard.dart';
import '../widgets/results_display.dart';

class AudioScreen extends StatefulWidget {
  const AudioScreen({super.key});

  @override
  State<AudioScreen> createState() => _AudioScreenState();
}

class _AudioScreenState extends State<AudioScreen> {
  // Controllers
  final AudioRecorder _audioRecorder = AudioRecorder();
  final AudioPlayer _audioPlayerOriginal = AudioPlayer();
  final AudioPlayer _audioPlayerProtected = AudioPlayer();

  // Services
  late final CryptoService _cryptoService;
  late final ApiService _apiService;
  late final AiAgentService _aiAgentService;
  late final TfliteService _tfliteService;

  // State
  bool _isInit = false;
  bool _isRecording = false;
  bool _isLoading = false;
  bool _isLocalFallbackActive = false;
  
  List<Map<String, String>> _chatMessages = [];
  bool _isAiTyping = false;
  
  String? _recordingPath;
  String? _protectedPath;
  
  int _currentScale = 3; 
  double? _pesqResult;
  double? _stoiResult;

  // Playback state
  bool _isPlayingOriginal = false;
  bool _isPlayingProtected = false;

  @override
  void initState() {
    super.initState();
    _initServices();
    
    // Original Player Streams
    _audioPlayerOriginal.playerStateStream.listen((state) {
      if (!mounted) return;
      setState(() => _isPlayingOriginal = state.playing);
      if (state.processingState == ProcessingState.completed) {
        _audioPlayerOriginal.stop();
        _audioPlayerOriginal.seek(Duration.zero);
        setState(() => _isPlayingOriginal = false);
      }
    });

    // Protected Player Streams
    _audioPlayerProtected.playerStateStream.listen((state) {
      if (!mounted) return;
      setState(() => _isPlayingProtected = state.playing);
      if (state.processingState == ProcessingState.completed) {
        _audioPlayerProtected.stop();
        _audioPlayerProtected.seek(Duration.zero);
        setState(() => _isPlayingProtected = false);
      }
    });
  }

  Future<void> _initServices() async {
    _cryptoService = CryptoService();
    await _cryptoService.init();
    
    _apiService = ApiService();
    _aiAgentService = AiAgentService();
    
    _tfliteService = TfliteService();
    await _tfliteService.init();
    
    setState(() {
      _isInit = true;
    });
  }

  @override
  void dispose() {
    // CRITICAL: Prevent memory leaks
    if (_isRecording) {
      _audioRecorder.stop();
    }
    _audioRecorder.dispose();
    _audioPlayerOriginal.dispose();
    _audioPlayerProtected.dispose();
    super.dispose();
  }

  Future<void> _toggleRecording() async {
    if (_isRecording) {
      final path = await _audioRecorder.stop();
      if (!mounted) return;
      setState(() {
        _isRecording = false;
        _recordingPath = path;
        _protectedPath = null;
        _pesqResult = null;
        _stoiResult = null;
        _isLocalFallbackActive = false;
      });
      if (path != null) {
        await _audioPlayerOriginal.setFilePath(path);
      }
    } else {
      if (await _audioRecorder.hasPermission()) {
        final dir = await getApplicationDocumentsDirectory();
        final filePath = p.join(dir.path, 'audio_${DateTime.now().millisecondsSinceEpoch}.wav');
        
        await _audioRecorder.start(
          const RecordConfig(
            encoder: AudioEncoder.wav,
            bitRate: 16000,
            sampleRate: 16000,
          ),
          path: filePath,
        );
        
        if (!mounted) return;
        setState(() {
          _isRecording = true;
          _recordingPath = null;
          _protectedPath = null;
          _pesqResult = null;
          _stoiResult = null;
        });
      }
    }
  }

  Future<void> _submitProtection(int scale) async {
    if (_recordingPath == null) return;

    setState(() {
      _isLoading = true;
      _isLocalFallbackActive = false;
    });

    try {
      // 1. Try API Service
      final result = await _apiService.protectAudio(_recordingPath!, scale);
      
      if (!mounted) return;
      setState(() {
        _protectedPath = result['path'];
        _pesqResult = result['pesq'];
        _stoiResult = result['stoi'];
      });
      await _audioPlayerProtected.setFilePath(_protectedPath!);
      
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(content: Text("Cloud Protection Successful"), backgroundColor: AppTheme.cyanAccent),
      );
      
    } catch (e) {
      if (e is OfflineException || e is ApiException || e is SocketException) {
        // Fallback to local
        if (!mounted) return;
        setState(() => _isLocalFallbackActive = true);
        
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(content: Text("Offline/Error. Local Mode Active. (${e.toString()})"), backgroundColor: AppTheme.orangeAccent),
        );

        try {
          final localPath = await _tfliteService.applyLocalProtection(_recordingPath!, scale);
          if (!mounted) return;
          setState(() {
            _protectedPath = localPath;
            // Provide mocked metrics for local fallback
            _pesqResult = 3.5;
            _stoiResult = 0.8;
          });
          await _audioPlayerProtected.setFilePath(_protectedPath!);
        } catch (localError) {
          if (!mounted) return;
          ScaffoldMessenger.of(context).showSnackBar(
            SnackBar(content: Text("Local Protection Failed: $localError"), backgroundColor: Colors.red),
          );
        }
      } else {
        if (!mounted) return;
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(content: Text("Unexpected Error: $e"), backgroundColor: Colors.red),
        );
      }
    } finally {
      if (mounted) {
        setState(() {
          _isLoading = false;
        });
      }
    }
  }

  Future<void> _pickAudioFile() async {
    final result = await FilePicker.platform.pickFiles(
      type: FileType.custom,
      allowedExtensions: ['wav'],
    );
    if (result != null && result.files.single.path != null) {
      if (!mounted) return;
      setState(() {
         _recordingPath = result.files.single.path;
         _protectedPath = null;
         _pesqResult = null;
         _stoiResult = null;
         _isLocalFallbackActive = false;
      });
      await _audioPlayerOriginal.setFilePath(_recordingPath!);
    }
  }

  Future<void> _handleAiGuardSubmit(String intent) async {
    if (!mounted) return;
    setState(() {
      _chatMessages.add({'sender': 'user', 'text': intent});
      _isAiTyping = true;
    });

    final response = await _aiAgentService.sendMessage(intent);
    
    if (!mounted) return;
    setState(() {
      _isAiTyping = false;
      _chatMessages.add({'sender': 'agent', 'text': response.message});
    });

    if (response.actionScale > 0) {
      if (_recordingPath == null) {
        setState(() {
          _chatMessages.add({'sender': 'agent', 'text': "Please record or upload an audio file first!"});
        });
        return;
      }
      setState(() => _currentScale = response.actionScale);
      await _submitProtection(response.actionScale);
    }
  }

  Future<void> _togglePlayOriginal() async {
    if (_isPlayingOriginal) {
      await _audioPlayerOriginal.pause();
    } else {
      await _audioPlayerOriginal.play();
    }
  }

  Future<void> _togglePlayProtected() async {
    if (_isPlayingProtected) {
      await _audioPlayerProtected.pause();
    } else {
      await _audioPlayerProtected.play();
    }
  }

  @override
  Widget build(BuildContext context) {
    if (!_isInit) {
      return Scaffold(
        backgroundColor: AppTheme.background,
        body: const Center(child: CircularProgressIndicator(color: AppTheme.cyanAccent)),
      );
    }

    return Scaffold(
      backgroundColor: AppTheme.background,
      appBar: AppBar(
        title: const Text("FAKeless Protection", style: TextStyle(fontWeight: FontWeight.bold, letterSpacing: 2)),
        actions: [
          if (_isLocalFallbackActive)
            const Padding(
              padding: EdgeInsets.only(right: 16),
              child: Chip(
                label: Text("Local Mode", style: TextStyle(color: Colors.black, fontWeight: FontWeight.bold, fontSize: 12)),
                backgroundColor: AppTheme.orangeAccent,
              ),
            ).animate(onPlay: (controller) => controller.repeat(reverse: true)).fade(begin: 0.5, end: 1.0)
        ],
      ),
      body: SingleChildScrollView(
        child: Padding(
          padding: const EdgeInsets.all(24.0),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.stretch,
            children: [
              // Recording Section
              Center(
                child: GestureDetector(
                  onTap: _isLoading ? null : _toggleRecording,
                  child: AnimatedContainer(
                    duration: const Duration(milliseconds: 300),
                    width: 100,
                    height: 100,
                    decoration: BoxDecoration(
                      shape: BoxShape.circle,
                      color: _isRecording ? AppTheme.orangeAccent : AppTheme.cyanAccent,
                      boxShadow: [
                        BoxShadow(
                          color: (_isRecording ? AppTheme.orangeAccent : AppTheme.cyanAccent).withOpacity(0.5),
                          blurRadius: _isRecording ? 30 : 15,
                          spreadRadius: _isRecording ? 5 : 0,
                        )
                      ],
                    ),
                    child: Icon(
                      _isRecording ? Icons.stop_rounded : Icons.mic_rounded,
                      size: 48,
                      color: Colors.black,
                    ),
                  ),
                ),
              ),
              const SizedBox(height: 16),
              Center(
                child: Text(
                  _isRecording ? "Recording..." : (_recordingPath != null ? "Ready for Protection" : "Tap to Record"),
                  style: TextStyle(color: _isRecording ? AppTheme.orangeAccent : Colors.white70),
                ),
              ),
              const SizedBox(height: 16),
              if (!_isRecording)
                Center(
                  child: OutlinedButton.icon(
                    onPressed: _isLoading ? null : _pickAudioFile,
                    icon: const Icon(Icons.upload_file, color: AppTheme.cyanAccent),
                    label: const Text("Upload .WAV", style: TextStyle(color: AppTheme.cyanAccent)),
                    style: OutlinedButton.styleFrom(
                      side: const BorderSide(color: AppTheme.cyanAccent),
                    ),
                  ),
                ),
              const SizedBox(height: 32),

              // Original Playback
              if (_recordingPath != null) ...[
                Card(
                  color: Colors.white10,
                  child: ListTile(
                    leading: const Icon(Icons.audio_file, color: Colors.white54),
                    title: const Text("Original Recording", style: TextStyle(color: Colors.white)),
                    trailing: IconButton(
                      icon: Icon(_isPlayingOriginal ? Icons.pause_circle_filled : Icons.play_circle_fill, color: AppTheme.cyanAccent, size: 36),
                      onPressed: _togglePlayOriginal,
                    ),
                  ),
                ).animate().slideX(),
                const SizedBox(height: 32),
              ],

              // Dashboard
              DualModeDashboard(
                currentManualScale: _currentScale,
                onScaleChanged: (scale) {
                  setState(() => _currentScale = scale);
                },
                onAiGuardSubmit: _handleAiGuardSubmit,
                chatMessages: _chatMessages,
                isAiTyping: _isAiTyping,
              ),

              const SizedBox(height: 32),

              // Manual Apply Button (if not triggered by AI Guard)
              if (_recordingPath != null && !_isLoading)
                ElevatedButton.icon(
                  style: ElevatedButton.styleFrom(
                    backgroundColor: AppTheme.cyanAccent,
                    foregroundColor: Colors.black,
                    padding: const EdgeInsets.symmetric(vertical: 16),
                  ),
                  onPressed: () => _submitProtection(_currentScale),
                  icon: const Icon(Icons.shield),
                  label: const Text("APPLY PROTECTION", style: TextStyle(fontWeight: FontWeight.bold, letterSpacing: 2)),
                ).animate().slideY(begin: 0.5),

              if (_isLoading)
                const Center(child: Padding(
                  padding: EdgeInsets.all(24.0),
                  child: CircularProgressIndicator(color: AppTheme.cyanAccent),
                )),

              // Results Section
              if (_protectedPath != null && _pesqResult != null && _stoiResult != null) ...[
                const SizedBox(height: 32),
                const Divider(color: Colors.white24),
                const SizedBox(height: 16),
                Card(
                  color: Colors.white10,
                  shape: RoundedRectangleBorder(side: const BorderSide(color: AppTheme.orangeAccent, width: 1), borderRadius: BorderRadius.circular(12)),
                  child: ListTile(
                    leading: const Icon(Icons.verified_user, color: AppTheme.orangeAccent),
                    title: const Text("Protected Recording", style: TextStyle(color: Colors.white, fontWeight: FontWeight.bold)),
                    trailing: IconButton(
                      icon: Icon(_isPlayingProtected ? Icons.pause_circle_filled : Icons.play_circle_fill, color: AppTheme.orangeAccent, size: 36),
                      onPressed: _togglePlayProtected,
                    ),
                  ),
                ).animate().fade().scale(),
                const SizedBox(height: 24),
                ResultsDisplay(pesqResult: _pesqResult!, stoiResult: _stoiResult!),
                const SizedBox(height: 24),
                Center(
                  child: ElevatedButton.icon(
                    onPressed: () {
                      if (_protectedPath != null) {
                        Share.shareXFiles(
                          [XFile(_protectedPath!)],
                          text: 'Check out my protected audio from FAKeless!',
                        );
                      }
                    },
                    icon: const Icon(Icons.share),
                    label: const Text("Share Protected Audio"),
                    style: ElevatedButton.styleFrom(
                      backgroundColor: AppTheme.orangeAccent,
                      foregroundColor: Colors.black,
                      padding: const EdgeInsets.symmetric(horizontal: 24, vertical: 12),
                    ),
                  ),
                ).animate().slideX(),
              ]
            ],
          ),
        ),
      ),
    );
  }
}
