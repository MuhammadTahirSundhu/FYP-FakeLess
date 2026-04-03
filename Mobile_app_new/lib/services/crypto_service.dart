import 'dart:convert';
import 'dart:typed_data';
import 'package:encrypt/encrypt.dart' as encrypt;
import 'package:flutter_secure_storage/flutter_secure_storage.dart';

class CryptoService {
  static const _storage = FlutterSecureStorage();
  static const _keyName = 'fernet_encryption_key';
  
  // The default hardcoded key from previous implementation.
  // In a real production scenario, this would be negotiated or injected dynamically.
  static const _defaultPlainKeyString = '0123456789abcdefghijklmnopqrstuv'; 

  late encrypt.Encrypter _encrypter;
  bool _initialized = false;

  Future<void> init() async {
    if (_initialized) return;

    String? b64KeyString = await _storage.read(key: _keyName);
    
    if (b64KeyString == null) {
      // Create base64 string from the 32-byte plain key
      b64KeyString = base64Url.encode(utf8.encode(_defaultPlainKeyString));
      await _storage.write(key: _keyName, value: b64KeyString);
    }

    final fernetKey = encrypt.Key.fromBase64(b64KeyString);
    final fernet = encrypt.Fernet(fernetKey);
    _encrypter = encrypt.Encrypter(fernet);
    _initialized = true;
  }

  String encryptFileToBase64(Uint8List fileBytes) {
    if (!_initialized) throw Exception("CryptoService not initialized");
    final encrypted = _encrypter.encryptBytes(fileBytes);
    return encrypted.base64;
  }

  Uint8List decryptBase64ToBytes(String base64Token) {
    if (!_initialized) throw Exception("CryptoService not initialized");
    final encrypted = encrypt.Encrypted.fromBase64(base64Token);
    final decrypted = _encrypter.decryptBytes(encrypted);
    return Uint8List.fromList(decrypted);
  }
}
