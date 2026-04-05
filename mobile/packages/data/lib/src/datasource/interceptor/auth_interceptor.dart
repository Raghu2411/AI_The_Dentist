import 'package:dio/dio.dart';
import 'package:domain/common/auth_handler.dart';
import 'package:domain/domain.dart';
import 'package:injectable/injectable.dart';

@singleton
class AuthInterceptor extends InterceptorsWrapper {
  final StorageRepository storageRepository;
  final AuthHandler authHandler;

  AuthInterceptor(this.storageRepository, this.authHandler);

  @override
  Future<void> onRequest(
    RequestOptions options,
    RequestInterceptorHandler handler,
  ) async {
    super.onRequest(options, handler);
  }
}
