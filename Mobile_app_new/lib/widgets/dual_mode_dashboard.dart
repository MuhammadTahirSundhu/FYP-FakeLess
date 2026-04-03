import 'package:flutter/cupertino.dart';
import 'package:flutter/material.dart';
import 'package:syncfusion_flutter_sliders/sliders.dart';
import 'package:flutter_animate/flutter_animate.dart';
import '../core/theme.dart';

class DualModeDashboard extends StatefulWidget {
  final ValueChanged<int> onScaleChanged;
  final ValueChanged<String> onAiGuardSubmit;
  final int currentManualScale;
  final List<Map<String, String>> chatMessages;
  final bool isAiTyping;

  const DualModeDashboard({
    super.key,
    required this.onScaleChanged,
    required this.onAiGuardSubmit,
    required this.currentManualScale,
    required this.chatMessages,
    this.isAiTyping = false,
  });

  @override
  State<DualModeDashboard> createState() => _DualModeDashboardState();
}

class _DualModeDashboardState extends State<DualModeDashboard> {
  int _selectedMode = 0; // 0 for Manual, 1 for AI Guard
  final TextEditingController _chatController = TextEditingController();

  @override
  void dispose() {
    _chatController.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Column(
      children: [
        // Mode Selector
        SizedBox(
          width: double.infinity,
          child: CupertinoSlidingSegmentedControl<int>(
            groupValue: _selectedMode,
            backgroundColor: Colors.white10,
            thumbColor: AppTheme.cyanAccent.withOpacity(0.2),
            children: {
              0: Padding(
                padding: const EdgeInsets.symmetric(vertical: 12),
                child: Text(
                  "Manual Precision",
                  style: TextStyle(
                    color: _selectedMode == 0 ? AppTheme.cyanAccent : Colors.white54,
                    fontWeight: FontWeight.bold,
                  ),
                ),
              ),
              1: Padding(
                padding: const EdgeInsets.symmetric(vertical: 12),
                child: Text(
                  "AI Guard",
                  style: TextStyle(
                    color: _selectedMode == 1 ? AppTheme.orangeAccent : Colors.white54,
                    fontWeight: FontWeight.bold,
                  ),
                ),
              ),
            },
            onValueChanged: (int? value) {
              if (value != null) {
                setState(() {
                  _selectedMode = value;
                  // Clear focus when switching
                  FocusScope.of(context).unfocus();
                });
              }
            },
          ).animate().fadeIn(),
        ),
        
        const SizedBox(height: 24),
        
        // Active Area
        AnimatedSwitcher(
          duration: const Duration(milliseconds: 400),
          switchInCurve: Curves.easeIn,
          switchOutCurve: Curves.easeOut,
          transitionBuilder: (Widget child, Animation<double> animation) {
            return FadeTransition(
              opacity: animation,
              child: SlideTransition(
                position: Tween<Offset>(
                  begin: const Offset(0.0, 0.1),
                  end: Offset.zero,
                ).animate(animation),
                child: child,
              ),
            );
          },
          child: _selectedMode == 0
              ? _buildManualMode(key: const ValueKey('ManualMode'))
              : _buildAiGuardMode(key: const ValueKey('AiGuardMode')),
        ),
      ],
    );
  }

  Widget _buildManualMode({required Key key}) {
    // Dynamic glow based on scale intensity
    final intensityGlow = AppTheme.cyanAccent.withOpacity(
      widget.currentManualScale / 5.0,
    );

    return Container(
      key: key,
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: AppTheme.background,
        borderRadius: BorderRadius.circular(16),
        boxShadow: [
          BoxShadow(
            color: intensityGlow,
            blurRadius: 10 + (widget.currentManualScale * 5.0),
            spreadRadius: -5,
          )
        ],
        border: Border.all(color: Colors.white12),
      ),
      child: Column(
        children: [
          const Text("Protection Intensity", style: TextStyle(fontWeight: FontWeight.bold)),
          const SizedBox(height: 16),
          SfSlider(
            min: 1.0,
            max: 5.0,
            value: widget.currentManualScale.toDouble(),
            interval: 1,
            showTicks: true,
            showLabels: true,
            activeColor: intensityGlow.withOpacity(1.0),
            inactiveColor: Colors.white24,
            onChanged: (dynamic value) {
              widget.onScaleChanged((value as double).toInt());
            },
          ),
        ],
      ),
    );
  }

  Widget _buildAiGuardMode({required Key key}) {
    return Container(
      key: key,
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: AppTheme.background,
        borderRadius: BorderRadius.circular(16),
        border: Border.all(color: AppTheme.orangeAccent.withOpacity(0.5)),
        boxShadow: [
          BoxShadow(
            color: AppTheme.orangeAccent.withOpacity(0.1),
            blurRadius: 20,
          )
        ],
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: const [
              Icon(Icons.smart_toy, color: AppTheme.orangeAccent),
              SizedBox(width: 8),
              Text("FAKeless Assistant", style: TextStyle(fontWeight: FontWeight.bold)),
            ],
          ),
          const SizedBox(height: 16),
          // Chat History
          Container(
            height: 200, // Fixed height for chat area
            margin: const EdgeInsets.only(bottom: 16),
            padding: const EdgeInsets.all(8),
            decoration: BoxDecoration(
              color: Colors.black26,
              borderRadius: BorderRadius.circular(12),
            ),
            child: ListView.builder(
              reverse: true,
              itemCount: widget.chatMessages.length + (widget.isAiTyping ? 1 : 0),
              itemBuilder: (context, index) {
                if (widget.isAiTyping && index == 0) {
                  return const Padding(
                    padding: EdgeInsets.symmetric(vertical: 8.0, horizontal: 12.0),
                    child: Align(
                      alignment: Alignment.centerLeft,
                      child: Text("Agent is typing...", style: TextStyle(color: Colors.white54, fontStyle: FontStyle.italic)),
                    ),
                  ).animate(onPlay: (controller) => controller.repeat(reverse: true)).fade(begin: 0.3, end: 1.0);
                }
                
                final msgIndex = widget.isAiTyping ? index - 1 : index;
                final msg = widget.chatMessages[widget.chatMessages.length - 1 - msgIndex];
                final isUser = msg['sender'] == 'user';
                
                return Padding(
                  padding: const EdgeInsets.symmetric(vertical: 4.0),
                  child: Align(
                    alignment: isUser ? Alignment.centerRight : Alignment.centerLeft,
                    child: Container(
                      padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 8),
                      decoration: BoxDecoration(
                        color: isUser ? AppTheme.cyanAccent.withOpacity(0.2) : AppTheme.orangeAccent.withOpacity(0.2),
                        borderRadius: BorderRadius.circular(12),
                        border: Border.all(color: isUser ? AppTheme.cyanAccent.withOpacity(0.5) : AppTheme.orangeAccent.withOpacity(0.5)),
                      ),
                      child: Text(
                        msg['text']!,
                        style: const TextStyle(color: Colors.white),
                      ),
                    ),
                  ),
                );
              },
            ),
          ),
          TextField(
            controller: _chatController,
            style: const TextStyle(color: Colors.white),
            decoration: InputDecoration(
              hintText: "e.g. 'Protect this audio please'",
              hintStyle: const TextStyle(color: Colors.white38),
              filled: true,
              fillColor: Colors.black26,
              suffixIcon: IconButton(
                icon: const Icon(Icons.send, color: AppTheme.orangeAccent),
                onPressed: () {
                  if (_chatController.text.trim().isNotEmpty) {
                    widget.onAiGuardSubmit(_chatController.text.trim());
                    // Clear the text field immediately for better UX
                    _chatController.clear();
                    FocusScope.of(context).unfocus();
                  }
                },
              ),
              border: OutlineInputBorder(
                borderRadius: BorderRadius.circular(12),
                borderSide: BorderSide.none,
              ),
            ),
            onSubmitted: (val) {
              if (val.trim().isNotEmpty) {
                widget.onAiGuardSubmit(val.trim());
                _chatController.clear();
              }
            },
          ),
        ],
      ),
    );
  }
}
