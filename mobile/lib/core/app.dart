import 'package:ai_dental_studio/core/cubit/bottom_sheet_cubit.dart';
import 'package:ai_dental_studio/core/styles/theme.dart';
import 'package:ai_dental_studio/feature/main_screen/chatbot_screen/chatbot_cubit/chatbot_text_field_cubit.dart';
import 'package:ai_dental_studio/feature/main_screen/feedback_screen/feedback_cubit/feedback_cubit.dart';
import 'package:ai_dental_studio/views/share_file/cubit/share_file_cubit.dart';
import 'package:flutter/material.dart';
import 'package:flutter_bloc/flutter_bloc.dart';
import 'package:kernel/kernel.dart';

import '../feature/main_screen/chatbot_screen/chatbot_cubit/chatbot_cubit.dart';
import '../feature/select_radio_graph_screen/cubit/select_image_bottom_sheet_cubit.dart';
import '../feature/select_radio_graph_screen/cubit/select_image_cubit.dart';
import '../feature/select_radio_graph_screen/cubit/select_model_cubit.dart';
import '../navigation/app_router.dart';

class App extends StatelessWidget {
  final AppRouter appRouter;

  const App({super.key, required this.appRouter});

  @override
  Widget build(BuildContext context) {
    return MultiBlocProvider(
      providers: [
        BlocProvider<BottomSheetCubit>(
          create: (BuildContext context) => getIt.get<BottomSheetCubit>(),
        ),
        BlocProvider<SelectImageBottomSheetCubit>(
          create: (BuildContext context) =>
              getIt.get<SelectImageBottomSheetCubit>(),
        ),
        BlocProvider<SelectImageCubit>(
          create: (BuildContext context) => getIt.get<SelectImageCubit>(),
        ),
        BlocProvider<ShareFileCubit>(
          create: (BuildContext context) => getIt.get<ShareFileCubit>(),
        ),
        BlocProvider<SelectModelCubit>(
          create: (BuildContext context) => getIt.get<SelectModelCubit>(),
        ),
        BlocProvider<ChatbotCubit>(
          create: (BuildContext context) => getIt.get<ChatbotCubit>(),
        ),
        BlocProvider<ChatbotTextFieldCubit>(
          create: (BuildContext context) => getIt.get<ChatbotTextFieldCubit>(),
        ),
        BlocProvider<FeedbackCubit>(
          create: (BuildContext context) => getIt.get<FeedbackCubit>(),
        ),
      ],
      child: MaterialApp.router(
        debugShowCheckedModeBanner: false,
        theme: getIt<ThemeBuilder>().getDefault(),
        routerConfig: appRouter.router,
      ),
    );
  }
}
