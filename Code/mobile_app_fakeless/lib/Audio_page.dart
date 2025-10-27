import 'package:flutter/material.dart';

class AudioPage extends StatelessWidget {
  const AudioPage({super.key});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text("Audio Page")),
      body: const Center(
        child: Text(
          "This is the new page!",
          style: TextStyle(fontSize: 20),
        ),
      ),
    );
  }
}
