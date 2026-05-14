import 'package:flutter/material.dart';
import 'package:flutter_animate/flutter_animate.dart';
import 'package:google_fonts/google_fonts.dart';
import '../core/theme.dart';

class FloatingAiAssistant extends StatefulWidget {
  final List<Map<String, String>> chatMessages;
  final bool isAiTyping;
  final bool isRecording;
  final bool hasAudio;
  final Future<String> Function(String) onSendMessage;
  final Future<void> Function() onToggleRecording;
  final Future<void> Function() onPickFile;

  const FloatingAiAssistant({
    super.key,
    required this.chatMessages,
    required this.isAiTyping,
    required this.isRecording,
    required this.hasAudio,
    required this.onSendMessage,
    required this.onToggleRecording,
    required this.onPickFile,
  });

  @override
  State<FloatingAiAssistant> createState() => _FloatingAiAssistantState();
}

class _FloatingAiAssistantState extends State<FloatingAiAssistant> with SingleTickerProviderStateMixin {
  int _unreadCount = 0;

  void _openChat() {
    setState(() => _unreadCount = 0);
    showModalBottomSheet(
      context: context,
      isScrollControlled: true,
      backgroundColor: Colors.transparent,
      barrierColor: Colors.black54,
      builder: (_) => _ChatPanel(
        chatMessages: widget.chatMessages,
        isAiTyping: widget.isAiTyping,
        isRecording: widget.isRecording,
        hasAudio: widget.hasAudio,
        onSendMessage: widget.onSendMessage,
        onToggleRecording: widget.onToggleRecording,
        onPickFile: widget.onPickFile,
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    return GestureDetector(
      onTap: _openChat,
      child: Stack(clipBehavior: Clip.none, children: [
        // Outer pulsing ring
        Container(
          width: 64, height: 64,
          decoration: BoxDecoration(
            shape: BoxShape.circle,
            border: Border.all(color: AppTheme.cyanAccent.withOpacity(0.3), width: 2),
          ),
        ).animate(onPlay: (c) => c.repeat(reverse: true))
          .scaleXY(begin: 1.0, end: 1.25, duration: 1600.ms, curve: Curves.easeInOut)
          .fade(begin: 0.6, end: 0.0, duration: 1600.ms),

        // Main FAB
        Container(
          width: 58, height: 58,
          decoration: BoxDecoration(
            shape: BoxShape.circle,
            gradient: AppTheme.cyanGradient,
            boxShadow: [
              BoxShadow(color: AppTheme.cyanAccent.withOpacity(0.5), blurRadius: 20, spreadRadius: 2),
            ],
          ),
          child: const Icon(Icons.security, color: Colors.black, size: 28),
        ).animate(onPlay: (c) => c.repeat(reverse: true))
          .scaleXY(begin: 0.96, end: 1.04, duration: 2000.ms, curve: Curves.easeInOut),

        // Unread badge
        if (_unreadCount > 0)
          Positioned(
            top: 0, right: 0,
            child: Container(
              width: 18, height: 18,
              decoration: const BoxDecoration(shape: BoxShape.circle, color: AppTheme.orangeAccent),
              child: Center(
                child: Text('$_unreadCount',
                    style: const TextStyle(color: Colors.black, fontSize: 10, fontWeight: FontWeight.bold)),
              ),
            ),
          ),
      ]),
    ).animate().fadeIn(delay: 600.ms).slideX(begin: 0.5);
  }
}

// ─────────────────────────────────────────── CHAT PANEL ──────────────────────
class _ChatPanel extends StatefulWidget {
  final List<Map<String, String>> chatMessages;
  final bool isAiTyping;
  final bool isRecording;
  final bool hasAudio;
  final Future<String> Function(String) onSendMessage;
  final Future<void> Function() onToggleRecording;
  final Future<void> Function() onPickFile;

  const _ChatPanel({
    required this.chatMessages,
    required this.isAiTyping,
    required this.isRecording,
    required this.hasAudio,
    required this.onSendMessage,
    required this.onToggleRecording,
    required this.onPickFile,
  });

  @override
  State<_ChatPanel> createState() => _ChatPanelState();
}

class _ChatPanelState extends State<_ChatPanel> {
  final TextEditingController _textController = TextEditingController();
  final ScrollController _scrollController = ScrollController();

  // Local copies — updated immediately without relying on parent setState
  late List<Map<String, String>> _messages;
  bool _isTyping = false;
  bool _isRecording = false;
  bool _hasAudio = false;

  @override
  void initState() {
    super.initState();
    // Seed from the current snapshot
    _messages = List.from(widget.chatMessages);
    _isTyping = widget.isAiTyping;
    _isRecording = widget.isRecording;
    _hasAudio = widget.hasAudio;
  }

  @override
  void dispose() {
    _textController.dispose();
    _scrollController.dispose();
    super.dispose();
  }

  void _scrollToBottom() {
    WidgetsBinding.instance.addPostFrameCallback((_) {
      if (_scrollController.hasClients) {
        _scrollController.animateTo(
          _scrollController.position.maxScrollExtent,
          duration: const Duration(milliseconds: 300),
          curve: Curves.easeOut,
        );
      }
    });
  }

  Future<void> _send() async {
    final text = _textController.text.trim();
    if (text.isEmpty) return;
    _textController.clear();

    // 1. Show user message immediately
    setState(() {
      _messages.add({'sender': 'user', 'text': text});
      _isTyping = true;
    });
    _scrollToBottom();

    // 2. Await AI response (returned directly — no parent sync needed)
    final agentReply = await widget.onSendMessage(text);

    // 3. Append the agent reply locally
    if (!mounted) return;
    setState(() {
      _messages.add({'sender': 'agent', 'text': agentReply});
      _isTyping = false;
      _hasAudio = widget.hasAudio;
    });
    _scrollToBottom();
  }

  Future<void> _handleToggleRecording() async {
    final wasRecording = _isRecording;
    await widget.onToggleRecording();
    if (!mounted) return;

    // Stopped recording → audio is now ready
    if (wasRecording) {
      setState(() {
        _isRecording = false;
        _hasAudio = true;
        _messages.add({
          'sender': 'agent',
          'text': '🎙️ Got it! Your recording is ready. Just say "protect this audio" and I\'ll secure it for you!',
        });
      });
      _scrollToBottom();
    } else {
      // Started recording
      setState(() {
        _isRecording = true;
        _messages.add({'sender': 'agent', 'text': '🔴 Recording started. Tap the mic again when you\'re done!'});
      });
      _scrollToBottom();
    }
  }

  Future<void> _handlePickFile() async {
    await widget.onPickFile();
    if (!mounted) return;
    final audioLoaded = widget.hasAudio;
    setState(() {
      _hasAudio = audioLoaded;
      if (audioLoaded) {
        _messages.add({
          'sender': 'agent',
          'text': '📁 Audio file loaded successfully! You can now ask me to protect it — just say "protect this" and choose your protection level.',
        });
      }
    });
    if (audioLoaded) _scrollToBottom();
  }

  @override
  Widget build(BuildContext context) {
    final bottomPadding = MediaQuery.of(context).viewInsets.bottom;

    return Padding(
      padding: EdgeInsets.only(bottom: bottomPadding),
      child: DraggableScrollableSheet(
        initialChildSize: 0.58,
        minChildSize: 0.35,
        maxChildSize: 1.0,
        builder: (_, scrollController) => Container(
        decoration: BoxDecoration(
          color: AppTheme.surface,
          borderRadius: const BorderRadius.vertical(top: Radius.circular(28)),
          border: Border.all(color: AppTheme.cyanAccent.withOpacity(0.2)),
          boxShadow: [BoxShadow(color: AppTheme.cyanAccent.withOpacity(0.1), blurRadius: 32, spreadRadius: 2)],
        ),
        child: Column(children: [
          // ── Handle ──
          const SizedBox(height: 12),
          Container(width: 40, height: 4, decoration: BoxDecoration(color: AppTheme.textSecondary.withOpacity(0.4), borderRadius: BorderRadius.circular(2))),
          const SizedBox(height: 12),

          // ── Header ──
          Padding(
            padding: const EdgeInsets.symmetric(horizontal: 20),
            child: Row(children: [
              Container(
                padding: const EdgeInsets.all(8),
                decoration: BoxDecoration(
                  color: AppTheme.cyanAccent.withOpacity(0.15),
                  borderRadius: BorderRadius.circular(10),
                ),
                child: const Icon(Icons.security, color: AppTheme.cyanAccent, size: 20),
              ),
              const SizedBox(width: 12),
              Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
                Text('FAKeless Guard', style: GoogleFonts.inter(color: AppTheme.textMain, fontWeight: FontWeight.w700, fontSize: 16)),
                Row(children: [
                  Container(width: 6, height: 6, decoration: const BoxDecoration(shape: BoxShape.circle, color: AppTheme.successGreen)),
                  const SizedBox(width: 5),
                  Text('Online', style: GoogleFonts.inter(color: AppTheme.successGreen, fontSize: 11)),
                ]),
              ]),
              const Spacer(),
              IconButton(
                icon: const Icon(Icons.close, color: AppTheme.textSecondary),
                onPressed: () => Navigator.pop(context),
              ),
            ]),
          ),

          const Divider(color: AppTheme.surfaceHigh, height: 24),

          // ── Messages ──
          Expanded(
            child: ListView.builder(
              controller: _scrollController,
              padding: const EdgeInsets.symmetric(horizontal: 16),
              itemCount: _messages.length + (_isTyping ? 1 : 0),
              itemBuilder: (ctx, i) {
                if (i == _messages.length && _isTyping) {
                  return _TypingIndicator();
                }
                final msg = _messages[i];
                final isUser = msg['sender'] == 'user';
                return _MessageBubble(text: msg['text'] ?? '', isUser: isUser, index: i);
              },
            ),
          ),

          // ── Status pill ──
          if (_isRecording)
            Container(
              margin: const EdgeInsets.fromLTRB(16, 8, 16, 0),
              padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 8),
              decoration: BoxDecoration(
                color: AppTheme.errorRed.withOpacity(0.12),
                borderRadius: BorderRadius.circular(20),
                border: Border.all(color: AppTheme.errorRed.withOpacity(0.4)),
              ),
              child: Row(mainAxisAlignment: MainAxisAlignment.center, children: [
                Container(width: 7, height: 7, decoration: const BoxDecoration(shape: BoxShape.circle, color: AppTheme.errorRed))
                    .animate(onPlay: (c) => c.repeat(reverse: true)).fade(begin: 0.3, end: 1.0, duration: 600.ms),
                const SizedBox(width: 8),
                Text('Recording in progress — tap \u25A0 to stop',
                    style: GoogleFonts.inter(color: AppTheme.errorRed, fontSize: 12, fontWeight: FontWeight.w600)),
              ]),
            ).animate().fadeIn().slideY(begin: 0.3)
          else if (_hasAudio)
            Container(
              margin: const EdgeInsets.fromLTRB(16, 8, 16, 0),
              padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 8),
              decoration: BoxDecoration(
                color: AppTheme.successGreen.withOpacity(0.12),
                borderRadius: BorderRadius.circular(20),
                border: Border.all(color: AppTheme.successGreen.withOpacity(0.4)),
              ),
              child: Row(mainAxisAlignment: MainAxisAlignment.center, children: [
                const Icon(Icons.check_circle_outline, color: AppTheme.successGreen, size: 15),
                const SizedBox(width: 6),
                Text('Audio ready \u2014 say "protect this" to secure it!',
                    style: GoogleFonts.inter(color: AppTheme.successGreen, fontSize: 12, fontWeight: FontWeight.w500)),
              ]),
            ).animate().fadeIn().slideY(begin: 0.3)
          else
            Container(
              margin: const EdgeInsets.fromLTRB(16, 8, 16, 0),
              padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 8),
              decoration: BoxDecoration(
                color: AppTheme.textSecondary.withOpacity(0.08),
                borderRadius: BorderRadius.circular(20),
                border: Border.all(color: AppTheme.textSecondary.withOpacity(0.2)),
              ),
              child: Row(mainAxisAlignment: MainAxisAlignment.center, children: [
                const Icon(Icons.mic_none, color: AppTheme.textSecondary, size: 14),
                const SizedBox(width: 6),
                Text('Tap \u25CB mic to record or \u25A1 upload a .WAV file',
                    style: GoogleFonts.inter(color: AppTheme.textSecondary, fontSize: 12)),
              ]),
            ),

          // ── Input row ──
          Padding(
            padding: const EdgeInsets.fromLTRB(12, 12, 12, 16),
            child: Row(children: [
              // Mic button
              _InputIconButton(
                icon: _isRecording ? Icons.stop_circle : Icons.mic,
                color: _isRecording ? AppTheme.errorRed : AppTheme.cyanAccent,
                onTap: _handleToggleRecording,
                animate: _isRecording,
              ),
              const SizedBox(width: 6),
              // Upload button
              _InputIconButton(
                icon: Icons.upload_file,
                color: AppTheme.cyanDim,
                onTap: _handlePickFile,
              ),
              const SizedBox(width: 8),
              // Text field
              Expanded(
                child: TextField(
                  controller: _textController,
                  onSubmitted: (_) => _send(),
                  style: GoogleFonts.inter(color: AppTheme.textMain, fontSize: 14),
                  decoration: InputDecoration(
                    hintText: 'Ask FAKeless Guard...',
                    hintStyle: GoogleFonts.inter(color: AppTheme.textSecondary, fontSize: 13),
                    filled: true,
                    fillColor: AppTheme.surfaceHigh,
                    contentPadding: const EdgeInsets.symmetric(horizontal: 16, vertical: 10),
                    border: OutlineInputBorder(borderRadius: BorderRadius.circular(24), borderSide: BorderSide.none),
                  ),
                ),
              ),
              const SizedBox(width: 8),
              // Send button
              GestureDetector(
                onTap: _send,
                child: Container(
                  width: 42, height: 42,
                  decoration: BoxDecoration(
                    gradient: AppTheme.cyanGradient,
                    shape: BoxShape.circle,
                    boxShadow: [BoxShadow(color: AppTheme.cyanAccent.withOpacity(0.3), blurRadius: 8)],
                  ),
                  child: const Icon(Icons.send_rounded, color: Colors.black, size: 18),
                ),
              ),
            ]),
          ),
        ]),
      ),
    ));
  }
}


class _InputIconButton extends StatelessWidget {
  final IconData icon;
  final Color color;
  final VoidCallback onTap;
  final bool animate;

  const _InputIconButton({required this.icon, required this.color, required this.onTap, this.animate = false});

  @override
  Widget build(BuildContext context) {
    final btn = GestureDetector(
      onTap: onTap,
      child: Container(
        width: 40, height: 40,
        decoration: BoxDecoration(
          shape: BoxShape.circle,
          color: color.withOpacity(0.15),
          border: Border.all(color: color.withOpacity(0.4)),
        ),
        child: Icon(icon, color: color, size: 20),
      ),
    );
    if (animate) {
      return btn.animate(onPlay: (c) => c.repeat(reverse: true)).scaleXY(begin: 0.9, end: 1.1, duration: 600.ms);
    }
    return btn;
  }
}

class _MessageBubble extends StatelessWidget {
  final String text;
  final bool isUser;
  final int index;
  const _MessageBubble({required this.text, required this.isUser, required this.index});

  @override
  Widget build(BuildContext context) {
    return Padding(
      padding: const EdgeInsets.only(bottom: 10),
      child: Row(
        mainAxisAlignment: isUser ? MainAxisAlignment.end : MainAxisAlignment.start,
        crossAxisAlignment: CrossAxisAlignment.end,
        children: [
          if (!isUser) ...[
            Container(
              width: 28, height: 28,
              decoration: BoxDecoration(shape: BoxShape.circle, color: AppTheme.cyanAccent.withOpacity(0.2)),
              child: const Icon(Icons.security, color: AppTheme.cyanAccent, size: 14),
            ),
            const SizedBox(width: 8),
          ],
          Flexible(
            child: Container(
              padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 10),
              decoration: BoxDecoration(
                gradient: isUser
                    ? AppTheme.cyanGradient
                    : const LinearGradient(colors: [AppTheme.surfaceHigh, AppTheme.surface]),
                borderRadius: BorderRadius.only(
                  topLeft: const Radius.circular(18),
                  topRight: const Radius.circular(18),
                  bottomLeft: isUser ? const Radius.circular(18) : Radius.zero,
                  bottomRight: isUser ? Radius.zero : const Radius.circular(18),
                ),
              ),
              child: Text(
                text,
                style: GoogleFonts.inter(
                  color: isUser ? Colors.black : AppTheme.textMain,
                  fontSize: 13,
                  height: 1.4,
                ),
              ),
            ),
          ),
          if (isUser) const SizedBox(width: 8),
        ],
      ),
    ).animate().fadeIn(delay: (index * 30).ms).slideY(begin: 0.2, delay: (index * 30).ms);
  }
}

class _TypingIndicator extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return Padding(
      padding: const EdgeInsets.only(bottom: 10),
      child: Row(children: [
        Container(
          width: 28, height: 28,
          decoration: BoxDecoration(shape: BoxShape.circle, color: AppTheme.cyanAccent.withOpacity(0.2)),
          child: const Icon(Icons.security, color: AppTheme.cyanAccent, size: 14),
        ),
        const SizedBox(width: 8),
        Container(
          padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 12),
          decoration: BoxDecoration(color: AppTheme.surfaceHigh, borderRadius: BorderRadius.circular(18)),
          child: Row(children: List.generate(3, (i) => Padding(
            padding: const EdgeInsets.symmetric(horizontal: 2),
            child: Container(width: 6, height: 6, decoration: const BoxDecoration(shape: BoxShape.circle, color: AppTheme.cyanAccent))
                .animate(onPlay: (c) => c.repeat(reverse: true))
                .scaleXY(begin: 0.5, end: 1.0, delay: (i * 200).ms, duration: 400.ms),
          ))),
        ),
      ]),
    ).animate().fadeIn();
  }
}
