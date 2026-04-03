import 'package:flutter/material.dart';
import 'package:flutter_animate/flutter_animate.dart';
import 'package:google_fonts/google_fonts.dart';
import 'package:shared_preferences/shared_preferences.dart';
import '../core/theme.dart';
import '../screens/main_shell.dart';

class OnboardingScreen extends StatefulWidget {
  const OnboardingScreen({super.key});

  @override
  State<OnboardingScreen> createState() => _OnboardingScreenState();
}

class _OnboardingScreenState extends State<OnboardingScreen> {
  final PageController _pageController = PageController();
  int _currentPage = 0;

  final List<Map<String, dynamic>> _pages = [
    {
      'title': 'Protect Your Voice',
      'body': 'Voice cloning AI can steal your identity from just a 3-second audio clip. We stop them before they start.',
      'icon': Icons.security,
      'color': AppTheme.cyanAccent,
    },
    {
      'title': 'Invisible Watermarks',
      'body': 'We inject an invisible acoustic watermark into your audio. To humans, it sounds identical. To AI, it is garbled noise.',
      'icon': Icons.graphic_eq,
      'color': AppTheme.purpleAccent,
    },
    {
      'title': 'Intelligent AI Guard',
      'body': 'Chat with the FAKeless AI Guard. It analyzes your audio duration and context to recommend the perfect protection scale.',
      'icon': Icons.smart_toy_outlined,
      'color': AppTheme.orangeAccent,
    },
  ];

  Future<void> _completeOnboarding() async {
    final prefs = await SharedPreferences.getInstance();
    await prefs.setBool('has_seen_onboarding', true);
    if (!mounted) return;
    Navigator.pushReplacement(
      context,
      MaterialPageRoute(builder: (_) => const MainShell()),
    );
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: AppTheme.background,
      body: Stack(
        children: [
          // Background Rings
          Positioned(
            top: -100, right: -50,
            child: _GlowRing(color: _pages[_currentPage]['color'].withOpacity(0.1), size: 300),
          ),
          Positioned(
            bottom: -50, left: -100,
            child: _GlowRing(color: _pages[_currentPage]['color'].withOpacity(0.05), size: 250),
          ),
          
          SafeArea(
            child: Column(
              children: [
                Align(
                  alignment: Alignment.topRight,
                  child: TextButton(
                    onPressed: _completeOnboarding,
                    child: Text('Skip', style: GoogleFonts.inter(color: AppTheme.textSecondary, fontWeight: FontWeight.w600)),
                  ),
                ),
                
                Expanded(
                  child: PageView.builder(
                    controller: _pageController,
                    onPageChanged: (idx) => setState(() => _currentPage = idx),
                    itemCount: _pages.length,
                    itemBuilder: (context, index) {
                      final page = _pages[index];
                      final color = page['color'] as Color;
                      return Padding(
                        padding: const EdgeInsets.all(40),
                        child: Column(
                          mainAxisAlignment: MainAxisAlignment.center,
                          children: [
                            Container(
                              padding: const EdgeInsets.all(32),
                              decoration: BoxDecoration(
                                shape: BoxShape.circle,
                                color: color.withOpacity(0.15),
                                border: Border.all(color: color.withOpacity(0.4), width: 2),
                                boxShadow: [BoxShadow(color: color.withOpacity(0.2), blurRadius: 40)],
                              ),
                              child: Icon(page['icon'], size: 80, color: color),
                            ).animate().scale(delay: 200.ms, begin: const Offset(0.8, 0.8), curve: Curves.easeOutBack),
                            
                            const SizedBox(height: 60),
                            
                            Text(
                              page['title'],
                              textAlign: TextAlign.center,
                              style: GoogleFonts.inter(color: AppTheme.textMain, fontWeight: FontWeight.w800, fontSize: 28),
                            ).animate().fadeIn(delay: 400.ms).slideY(begin: 0.2),
                            
                            const SizedBox(height: 20),
                            
                            Text(
                              page['body'],
                              textAlign: TextAlign.center,
                              style: GoogleFonts.inter(color: AppTheme.textSecondary, fontSize: 16, height: 1.5),
                            ).animate().fadeIn(delay: 500.ms).slideY(begin: 0.2),
                          ],
                        ),
                      );
                    },
                  ),
                ),
                
                Padding(
                  padding: const EdgeInsets.all(30),
                  child: Row(
                    mainAxisAlignment: MainAxisAlignment.spaceBetween,
                    children: [
                      // Dots
                      Row(
                        children: List.generate(
                          _pages.length,
                          (index) => AnimatedContainer(
                            duration: const Duration(milliseconds: 300),
                            margin: const EdgeInsets.only(right: 8),
                            width: _currentPage == index ? 24 : 8,
                            height: 8,
                            decoration: BoxDecoration(
                              color: _currentPage == index ? _pages[_currentPage]['color'] : AppTheme.surfaceHigh,
                              borderRadius: BorderRadius.circular(4),
                            ),
                          ),
                        ),
                      ),
                      
                      // Next / Complete Button
                      GestureDetector(
                        onTap: () {
                          if (_currentPage < _pages.length - 1) {
                            _pageController.nextPage(duration: const Duration(milliseconds: 400), curve: Curves.easeInOut);
                          } else {
                            _completeOnboarding();
                          }
                        },
                        child: AnimatedContainer(
                          duration: const Duration(milliseconds: 300),
                          padding: const EdgeInsets.symmetric(horizontal: 24, vertical: 14),
                          decoration: BoxDecoration(
                            color: _pages[_currentPage]['color'],
                            borderRadius: BorderRadius.circular(30),
                            boxShadow: [BoxShadow(color: _pages[_currentPage]['color'].withOpacity(0.4), blurRadius: 16, offset: const Offset(0, 4))],
                          ),
                          child: Row(
                            mainAxisSize: MainAxisSize.min,
                            children: [
                              Text(
                                _currentPage == _pages.length - 1 ? 'Get Started' : 'Next',
                                style: GoogleFonts.inter(color: Colors.black, fontWeight: FontWeight.w700, fontSize: 15),
                              ),
                              const SizedBox(width: 8),
                              Icon(_currentPage == _pages.length - 1 ? Icons.check : Icons.arrow_forward, color: Colors.black, size: 18),
                            ],
                          ),
                        ),
                      ),
                    ],
                  ),
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }
}

class _GlowRing extends StatelessWidget {
  final Color color;
  final double size;
  const _GlowRing({required this.color, required this.size});

  @override
  Widget build(BuildContext context) {
    return Container(
      width: size, height: size,
      decoration: BoxDecoration(
        shape: BoxShape.circle,
        border: Border.all(color: color, width: 2),
      ),
      child: Center(
        child: Container(
          width: size * 0.7, height: size * 0.7,
          decoration: BoxDecoration(
            shape: BoxShape.circle,
            border: Border.all(color: color, width: 2),
          ),
        ),
      ),
    );
  }
}
