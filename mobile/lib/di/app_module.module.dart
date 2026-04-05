//@GeneratedMicroModule;AiDentalStudioPackageModule;package:ai_dental_studio/di/app_module.module.dart
// GENERATED CODE - DO NOT MODIFY BY HAND
// ignore_for_file: type=lint
// coverage:ignore-file

// ignore_for_file: no_leading_underscores_for_library_prefixes
import 'dart:async' as _i687;

import 'package:ai_dental_studio/core/cubit/bottom_sheet_cubit.dart' as _i434;
import 'package:ai_dental_studio/core/styles/custom_input_style.dart' as _i116;
import 'package:ai_dental_studio/core/styles/theme.dart' as _i912;
import 'package:ai_dental_studio/di/app_module.dart' as _i633;
import 'package:ai_dental_studio/feature/main_screen/chatbot_screen/chatbot_cubit/chatbot_cubit.dart'
    as _i838;
import 'package:ai_dental_studio/feature/main_screen/chatbot_screen/chatbot_cubit/chatbot_text_field_cubit.dart'
    as _i812;
import 'package:ai_dental_studio/feature/main_screen/feedback_screen/feedback_cubit/feedback_cubit.dart'
    as _i809;
import 'package:ai_dental_studio/feature/select_radio_graph_screen/cubit/select_image_bottom_sheet_cubit.dart'
    as _i58;
import 'package:ai_dental_studio/feature/select_radio_graph_screen/cubit/select_image_cubit.dart'
    as _i675;
import 'package:ai_dental_studio/feature/select_radio_graph_screen/cubit/select_model_cubit.dart'
    as _i312;
import 'package:ai_dental_studio/navigation/app_router.dart' as _i910;
import 'package:ai_dental_studio/views/share_file/cubit/share_file_cubit.dart'
    as _i888;
import 'package:domain/usecase/chatbot_communication_usecase.dart' as _i583;
import 'package:domain/usecase/check_opg_image_analyses_usecase.dart' as _i86;
import 'package:domain/usecase/download_file_from_url_usecase.dart' as _i257;
import 'package:domain/usecase/submit_feedback_usecase.dart' as _i645;
import 'package:injectable/injectable.dart' as _i526;
import 'package:logger/logger.dart' as _i974;

class AiDentalStudioPackageModule extends _i526.MicroPackageModule {
// initializes the registration of main-scope dependencies inside of GetIt
  @override
  _i687.FutureOr<void> init(_i526.GetItHelper gh) {
    final appModule = _$AppModule();
    gh.factory<_i434.BottomSheetCubit>(() => _i434.BottomSheetCubit());
    gh.factory<_i312.SelectModelCubit>(() => _i312.SelectModelCubit());
    gh.factory<_i58.SelectImageBottomSheetCubit>(
        () => _i58.SelectImageBottomSheetCubit());
    gh.factory<_i812.ChatbotTextFieldCubit>(
        () => _i812.ChatbotTextFieldCubit());
    gh.singleton<_i912.ThemeBuilder>(() => appModule.themeBuilder);
    gh.singleton<_i116.CustomInputStyle>(() => appModule.customInputStyle);
    gh.singleton<_i910.AppRouter>(() => _i910.AppRouter());
    gh.factory<_i675.SelectImageCubit>(() => _i675.SelectImageCubit(
          gh<_i86.CheckOPGImageAnalysesUseCase>(),
          gh<_i974.Logger>(),
        ));
    gh.factory<_i809.FeedbackCubit>(
        () => _i809.FeedbackCubit(gh<_i645.SubmitFeedbackUseCase>()));
    gh.factory<_i838.ChatbotCubit>(
        () => _i838.ChatbotCubit(gh<_i583.ChatbotCommunicationUseCase>()));
    gh.factory<_i888.ShareFileCubit>(
        () => _i888.ShareFileCubit(gh<_i257.DownloadFileFromUrlUseCase>()));
  }
}

class _$AppModule extends _i633.AppModule {}
