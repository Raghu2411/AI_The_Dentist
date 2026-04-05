abstract class AuthHandler {
  Stream<void> get onUnauthorized;

  void triggerUnauthorized();

  void dispose();
}
