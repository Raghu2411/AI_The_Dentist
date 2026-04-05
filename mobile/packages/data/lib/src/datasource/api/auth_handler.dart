import 'dart:async';

import 'package:domain/common/auth_handler.dart';
import 'package:injectable/injectable.dart';

@Singleton(as: AuthHandler)
class AuthHandlerImpl implements AuthHandler {
  final _controller = StreamController<void>.broadcast();

  @override
  void triggerUnauthorized() {
    if (!_controller.isClosed) {
      _controller.add(null);
    }
  }

  @override
  void dispose() {
    _controller.close();
  }

  @override
  Stream<void> get onUnauthorized => _controller.stream;
}
