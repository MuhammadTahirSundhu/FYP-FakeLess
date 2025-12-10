import 'package:flutter/material.dart';
import 'package:mobile_app_fakeless/splashscreen.dart';
import 'audio_page.dart';

void main() {
  runApp(const MyApp());
}

class MyApp extends StatelessWidget {
  const MyApp({super.key});

  // Root widget: follow system theme (light/dark)
  @override
  Widget build(BuildContext context) {
    final seed = const Color.fromARGB(255, 99, 195, 244);

    final lightTheme = ThemeData(
      colorScheme: ColorScheme.fromSeed(seedColor: seed, brightness: Brightness.light),
      useMaterial3: true,
      brightness: Brightness.light,
    );

    final darkTheme = ThemeData(
      colorScheme: ColorScheme.fromSeed(seedColor: seed, brightness: Brightness.dark),
      useMaterial3: true,
      brightness: Brightness.dark,
    );

    return MaterialApp(
      title: 'Fakeless',
      theme: lightTheme,
      darkTheme: darkTheme,
      themeMode: ThemeMode.system, // follow device/system setting
      home: const SplashScreen(),
    );
  }
}

class MyHomePage extends StatefulWidget {
  const MyHomePage({super.key, required this.title});

  // This widget is the home page of your application. It is stateful, meaning
  // that it has a State object (defined below) that contains fields that affect
  // how it looks.

  // This class is the configuration for the state. It holds the values (in this
  // case the title) provided by the parent (in this case the App widget) and
  // used by the build method of the State. Fields in a Widget subclass are
  // always marked "final".

  final String title;

  @override
  State<MyHomePage> createState() => _MyHomePageState();
}

class _MyHomePageState extends State<MyHomePage> {
  String message = "";

  void _uploadAudio() {
    // Implement your audio upload logic here
    setState(() {
      message = "Audio uploaded successfully!";
    });
    print("Upload Audio button pressed");
  }
  void _recordAudio() {
    // Implement your audio upload logic here
    Navigator.push(
      context,
      MaterialPageRoute(builder: (context) => const AudioPage()),
      );
    setState(() {
      message = "Audio recorded successfully!";
    });
  }

  @override
  Widget build(BuildContext context) {
    // This method is rerun every time setState is called, for instance as done
    // by the _incrementCounter method above.
    //
    // The Flutter framework has been optimized to make rerunning build methods
    // fast, so that you can just rebuild anything that needs updating rather
    // than having to individually change instances of widgets.
    final screenWidth = MediaQuery.of(context).size.width;
    return Scaffold(
      appBar: AppBar(
        // TRY THIS: Try changing the color here to a specific color (to
        // Colors.amber, perhaps?) and trigger a hot reload to see the AppBar
        // change color while the other colors stay the same.
        backgroundColor: Theme.of(context).colorScheme.primary,
        // Here we take the value from the MyHomePage object that was created by
        // the App.build method, and use it to set our appbar title.
        title: Center(
          child: Text(
            widget.title,
            style: TextStyle(
              color: Theme.of(context).colorScheme.onPrimary,
            )
          ),
        ),
      ),
      body: Center(
        // Center is a layout widget. It takes a single child and positions it
        // in the middle of the parent.
        child: Column(
          // Column is also a layout widget. It takes a list of children and
          // arranges them vertically. By default, it sizes itself to fit its
          // children horizontally, and tries to be as tall as its parent.
          //
          // Column has various properties to control how it sizes itself and
          // how it positions its children. Here we use mainAxisAlignment to
          // center the children vertically; the main axis here is the vertical
          // axis because Columns are vertical (the cross axis would be
          // horizontal).
          //
          // TRY THIS: Invoke "debug painting" (choose the "Toggle Debug Paint"
          // action in the IDE, or press "p" in the console), to see the
          // wireframe for each widget.
          mainAxisAlignment: MainAxisAlignment.center,
          spacing: 10, // space between elements
          
          children: <Widget>[
            Row(
              mainAxisAlignment: MainAxisAlignment.center, // center horizontally
              spacing: 0.05 * screenWidth, // space between buttons
              children: [
                FloatingActionButton.extended(
                  onPressed: _uploadAudio,
                  foregroundColor: Color.fromARGB(255, 0, 0, 0),
                  backgroundColor: Theme.of(context).colorScheme.primary,
                  hoverColor: const Color.fromARGB(255, 162, 216, 245),
                  splashColor: const Color.fromARGB(255, 3, 159, 243),
                  elevation: 0,
                  hoverElevation: 2,
                  highlightElevation: 0,
                  label: Text(
                    'Upload Audio',
                    style: TextStyle(
                      // fontSize: 16,
                      // fontWeight: FontWeight.bold,
                      color: Theme.of(context).colorScheme.onPrimary,
                    ),
                  ),
                  heroTag: 'uploadButton',
                ),

                FloatingActionButton.extended(
                  onPressed: _recordAudio,
                  foregroundColor: Color.fromARGB(255, 0, 0, 0),
                  backgroundColor: Theme.of(context).colorScheme.primary,
                  hoverColor: const Color.fromARGB(255, 162, 216, 245),
                  splashColor: const Color.fromARGB(255, 3, 159, 243),
                  elevation: 0,
                  hoverElevation: 2,
                  highlightElevation: 0,
                  label: Text(
                    'Record Audio',
                    style: TextStyle(
                      // fontSize: 16,
                      // fontWeight: FontWeight.bold,
                      color: Theme.of(context).colorScheme.onPrimary,
                    ),
                  ),
                  heroTag: 'recordButton',
                ),
              ],
            ),
            Text(
              message,
              style: Theme.of(context).textTheme.headlineMedium,
            ),
            // Text(
            //   '$_counter',
            //   style: Theme.of(context).textTheme.headlineMedium,
            // ),
          ],
        ),
      ),
      // floatingActionButton: FloatingActionButton(
      //   onPressed: _incrementCounter,
      //   tooltip: 'Increment',
      //   heroTag: 'incrementButton',
      //   child: const Icon(Icons.add),
      // ), // This trailing comma makes auto-formatting nicer for build methods.
      backgroundColor: Theme.of(context).colorScheme.primaryContainer,
    );
  }
}
