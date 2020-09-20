import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:camera/camera.dart';
import 'package:tflite/tflite.dart';

void main() {
  runApp(MyApp());
}

class MyApp extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    SystemChrome.setPreferredOrientations([
      DeviceOrientation.portraitUp,
    ]);
    return MaterialApp(
      debugShowCheckedModeBanner: false,
      home: MyHomePage(),
    );
  }
}

class MyHomePage extends StatefulWidget {
  MyHomePage({Key key}) : super(key: key);
  @override
  _MyHomePageState createState() => _MyHomePageState();
}

class _MyHomePageState extends State<MyHomePage> {
  CameraController controller;
  bool _cameraInitialized = false;
  bool isDetecting = false;
  var data = [];

  @override
  void initState() {
    super.initState();
    SystemChrome.setEnabledSystemUIOverlays([SystemUiOverlay.bottom]);
    SystemChrome.setSystemUIOverlayStyle(SystemUiOverlayStyle.dark);
    _initializeCamera();
  }

  void _initializeCamera() async {
    String res = await Tflite.loadModel(
      model: "assets/m1.tflite",
      labels: "assets/label.txt",
      useGpuDelegate: true,
    );
    print(res);

    List<CameraDescription> cameras = await availableCameras();
    controller = CameraController(cameras[1], ResolutionPreset.medium);

    controller.initialize().then((_) {
      if (!mounted) {
        return;
      }
      _cameraInitialized = true;
      print('Camera initialization complete');

      setState(() {});
      controller.startImageStream(_scan);
    });
    // controller.stopImageStream();
  }

  _scan(CameraImage img) async {
    if (!isDetecting) {
      isDetecting = true;

      Tflite.runModelOnFrame(
        bytesList: img.planes.map((plane) => plane.bytes).toList(),
      ).then((recognitions) {
        setState(() {
          data = recognitions;
        });
        // print(recognitions);
        isDetecting = false;
      });
      /* var rec = await Tflite.runModelOnFrame(
        bytesList: img.planes.map((plane) {
          return plane.bytes;
        }).toList(),
      );
      print(rec);
      isDetecting = false; */
    }

    /* setState(() {
      text = recognitions;
    }); */
  }

  @override
  void dispose() {
    controller.dispose();
    Tflite.close();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      floatingActionButtonLocation: FloatingActionButtonLocation.centerFloat,
      body: Container(
        color: Colors.black,
        // constraints: const BoxConstraints.expand(),
        child: Column(
          children: <Widget>[
            /* Text(
              _topText,
              style: TextStyle(color: Colors.white, fontSize: 18),
            ), */
            _cameraInitialized
                ? Expanded(
                    child: OverflowBox(
                      maxWidth: double.infinity,
                      child: AspectRatio(
                        aspectRatio: controller.value.aspectRatio,
                        child: CameraPreview(controller),
                      ),
                    ),
                  )
                : Container(),
            Text(
              data.length != 0 ? data[0]['label'].toString() : '',
              style: TextStyle(color: Colors.white, fontSize: 18),
            ),
          ],
        ),
      ),
      /* floatingActionButton: FloatingActionButton(
        backgroundColor: Colors.white,
        onPressed: () {},
      ), */
    );
  }
}
