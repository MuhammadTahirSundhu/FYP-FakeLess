import 'package:animated_splash_screen/animated_splash_screen.dart';
import 'package:flutter/material.dart';
import 'package:lottie/lottie.dart';
import 'package:mobile_app_fakeless/main.dart';

class SplashScreen extends StatelessWidget {
  const SplashScreen({super.key});

  @override
  Widget build(BuildContext context) {
    return AnimatedSplashScreen(
        splash: LottieBuilder.asset("assets/animation/phone-voice.json"),
        nextScreen: const MyHomePage(title: 'Fakeless Home Page'),
        splashIconSize: 400,
        backgroundColor: Theme.of(context).colorScheme.primaryContainer,
      );
  }
}