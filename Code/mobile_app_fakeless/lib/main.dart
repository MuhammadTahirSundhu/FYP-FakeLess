import 'package:flutter/material.dart';
import 'Audio_page.dart';

void main() {
  runApp(const MyApp());
}

class MyApp extends StatelessWidget {
  const MyApp({super.key});

  // This widget is the root of your application.
  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Fakeless',
      theme: ThemeData(
        // This is the theme of your application.
        //
        // TRY THIS: Try running your application with "flutter run". You'll see
        // the application has a purple toolbar. Then, without quitting the app,
        // try changing the seedColor in the colorScheme below to Colors.green
        // and then invoke "hot reload" (save your changes or press the "hot
        // reload" button in a Flutter-supported IDE, or press "r" if you used
        // the command line to start the app).
        //
        // Notice that the counter didn't reset back to zero; the application
        // state is not lost during the reload. To reset the state, use hot
        // restart instead.
        //
        // This works for code too, not just values: Most code changes can be
        // tested with just a hot reload.
        colorScheme: ColorScheme.fromSeed(seedColor: const Color.fromARGB(255, 87, 186, 236)),
      ),
      home: const MyHomePage(title: 'Fakeless Home Page'),
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
  int _counter = 0;
  String message = "";

  void _incrementCounter() {
    setState(() {
      // This call to setState tells the Flutter framework that something has
      // changed in this State, which causes it to rerun the build method below
      // so that the display can reflect the updated values. If we changed
      // _counter without calling setState(), then the build method would not be
      // called again, and so nothing would appear to happen.
      _counter++;
    });
  }

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
    print("Record Audio button pressed");
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
        backgroundColor: Theme.of(context).colorScheme.inversePrimary,
        // Here we take the value from the MyHomePage object that was created by
        // the App.build method, and use it to set our appbar title.
        title: Center(
          child: Text(widget.title)
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
                  backgroundColor: Theme.of(context).colorScheme.inversePrimary,
                  hoverColor: const Color.fromARGB(255, 162, 216, 245),
                  splashColor: const Color.fromARGB(255, 3, 159, 243),
                  elevation: 0,
                  hoverElevation: 2,
                  highlightElevation: 0,
                  label: Text('Upload Audio'),
                  heroTag: 'uploadButton',
                ),

                FloatingActionButton.extended(
                  onPressed: _recordAudio,
                  foregroundColor: Color.fromARGB(255, 0, 0, 0),
                  backgroundColor: Theme.of(context).colorScheme.inversePrimary,
                  hoverColor: const Color.fromARGB(255, 162, 216, 245),
                  splashColor: const Color.fromARGB(255, 3, 159, 243),
                  elevation: 0,
                  hoverElevation: 2,
                  highlightElevation: 0,
                  label: Text('Record Audio'),
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
      floatingActionButton: FloatingActionButton(
        onPressed: _incrementCounter,
        tooltip: 'Increment',
        heroTag: 'incrementButton',
        child: const Icon(Icons.add),
      ), // This trailing comma makes auto-formatting nicer for build methods.
    );
  }
}
