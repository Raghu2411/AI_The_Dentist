//@GeneratedMicroModule;DomainPackageModule;package:domain/di/domain_module.module.dart
// GENERATED CODE - DO NOT MODIFY BY HAND
// ignore_for_file: type=lint
// coverage:ignore-file

// ignore_for_file: no_leading_underscores_for_library_prefixes
import 'dart:async' as _i687;

import 'package:domain/domain.dart' as _i494;
import 'package:domain/model/chat_history.dart' as _i114;
import 'package:domain/usecase/chatbot_communication_usecase.dart' as _i583;
import 'package:domain/usecase/check_opg_image_analyses_usecase.dart' as _i86;
import 'package:domain/usecase/download_file_from_url_usecase.dart' as _i257;
import 'package:domain/usecase/submit_feedback_usecase.dart' as _i645;
import 'package:injectable/injectable.dart' as _i526;

class DomainPackageModule extends _i526.MicroPackageModule {
// initializes the registration of main-scope dependencies inside of GetIt
  @override
  _i687.FutureOr<void> init(_i526.GetItHelper gh) {
    gh.factory<_i114.ChatHistory>(
        () => _i114.ChatHistory(history: gh<List<Map<String, String>>>()));
    gh.factory<_i257.DownloadFileFromUrlUseCase>(() =>
        _i257.DownloadFileFromUrlUseCase(gh<_i494.EssexDentalApiRepository>()));
    gh.factory<_i86.CheckOPGImageAnalysesUseCase>(() =>
        _i86.CheckOPGImageAnalysesUseCase(
            gh<_i494.EssexDentalApiRepository>()));
    gh.factory<_i645.SubmitFeedbackUseCase>(() =>
        _i645.SubmitFeedbackUseCase(gh<_i494.EssexDentalApiRepository>()));
    gh.factory<_i583.ChatbotCommunicationUseCase>(() =>
        _i583.ChatbotCommunicationUseCase(
            gh<_i494.EssexDentalApiRepository>()));
  }
}
