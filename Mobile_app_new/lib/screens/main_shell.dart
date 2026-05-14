import 'package:flutter/material.dart';
import 'package:google_fonts/google_fonts.dart';
import '../core/theme.dart';
import '../screens/home_screen.dart';
import '../screens/protect_screen.dart';
import '../screens/history_screen.dart';
import '../screens/detect_screen.dart';
import '../screens/faq_screen.dart';
import '../widgets/floating_ai_assistant.dart';

class MainShell extends StatefulWidget {
  const MainShell({super.key});

  @override
  State<MainShell> createState() => _MainShellState();
}

class _MainShellState extends State<MainShell> {
  int _selectedIndex = 0;
  final GlobalKey<ProtectScreenState> _protectKey = GlobalKey<ProtectScreenState>();

  void _goToProtect() => setState(() => _selectedIndex = 1);

  @override
  Widget build(BuildContext context) {
    final protectScreen = ProtectScreen(key: _protectKey);

    final screens = [
      HomeScreen(onStartProtecting: _goToProtect),
      protectScreen,
      const HistoryScreen(),
      const DetectScreen(),
      const FaqScreen(),
    ];

    return Scaffold(
      backgroundColor: AppTheme.background,
      body: Stack(
        children: [
          IndexedStack(
            index: _selectedIndex,
            children: screens,
          ),

          // Floating AI Security Agent
          Positioned(
            bottom: 90,
            right: 20,
            child: _buildFloatingAgent(),
          ),
        ],
      ),
      bottomNavigationBar: _BottomNav(
        selectedIndex: _selectedIndex,
        onTap: (i) => setState(() {
          _selectedIndex = i;
        }),
      ),
    );
  }

  Widget _buildFloatingAgent() {
    final state = _protectKey.currentState;
    return FloatingAiAssistant(
      chatMessages: state?.chatMessages ?? [],
      isAiTyping: state?.isAiTyping ?? false,
      isRecording: state?.isRecording ?? false,
      hasAudio: state?.hasAudio ?? false,
      onSendMessage: (msg) async {
        if (_protectKey.currentState != null) {
          final reply = await _protectKey.currentState!.handleAiGuardSubmit(msg);
          setState(() {});
          return reply;
        }
        return 'FAKeless Guard is not ready yet. Please try again.';
      },
      onToggleRecording: () async {
        if (_protectKey.currentState != null) {
          await _protectKey.currentState!.toggleRecording();
          setState(() {});
        }
      },
      onPickFile: () async {
        if (_protectKey.currentState != null) {
          await _protectKey.currentState!.pickAudioFile();
          setState(() {});
        }
      },
    );
  }
}

class _BottomNav extends StatelessWidget {
  final int selectedIndex;
  final ValueChanged<int> onTap;
  const _BottomNav({required this.selectedIndex, required this.onTap});

  @override
  Widget build(BuildContext context) {
    final items = [
      {'icon': Icons.home_outlined,    'activeIcon': Icons.home,    'label': 'Home'},
      {'icon': Icons.shield_outlined,  'activeIcon': Icons.shield,  'label': 'Protect'},
      {'icon': Icons.history_outlined, 'activeIcon': Icons.history, 'label': 'History'},
      {'icon': Icons.search_outlined,  'activeIcon': Icons.search,  'label': 'Detect'},
      {'icon': Icons.help_outline,     'activeIcon': Icons.help,    'label': 'FAQ'},
    ];

    return Container(
      decoration: BoxDecoration(
        color: AppTheme.surface,
        border: Border(top: BorderSide(color: AppTheme.cyanAccent.withOpacity(0.15), width: 1)),
        boxShadow: [BoxShadow(color: Colors.black.withOpacity(0.4), blurRadius: 16)],
      ),
      child: SafeArea(
        child: SizedBox(
          height: 64,
          child: Row(
            children: items.asMap().entries.map((e) {
              final isSelected = selectedIndex == e.key;
              final item = e.value;
              return Expanded(
                child: GestureDetector(
                  behavior: HitTestBehavior.opaque,
                  onTap: () => onTap(e.key),
                  child: Column(mainAxisAlignment: MainAxisAlignment.center, children: [
                    AnimatedContainer(
                      duration: const Duration(milliseconds: 200),
                      padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 4),
                      decoration: BoxDecoration(
                        color: isSelected ? AppTheme.cyanAccent.withOpacity(0.12) : Colors.transparent,
                        borderRadius: BorderRadius.circular(20),
                      ),
                      child: Icon(
                        isSelected ? item['activeIcon'] as IconData : item['icon'] as IconData,
                        color: isSelected ? AppTheme.cyanAccent : AppTheme.textSecondary,
                        size: 22,
                      ),
                    ),
                    const SizedBox(height: 3),
                    Text(
                      item['label'] as String,
                      style: GoogleFonts.inter(
                        color: isSelected ? AppTheme.cyanAccent : AppTheme.textSecondary,
                        fontSize: 10,
                        fontWeight: isSelected ? FontWeight.w700 : FontWeight.w400,
                      ),
                    ),
                  ]),
                ),
              );
            }).toList(),
          ),
        ),
      ),
    );
  }
}
