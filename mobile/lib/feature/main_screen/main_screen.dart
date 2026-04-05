import 'dart:io';

import 'package:ai_dental_studio/core/constants/app_colors.dart';
import 'package:ai_dental_studio/core/constants/dimens.dart';
import 'package:ai_dental_studio/core/constants/strings.dart';
import 'package:ai_dental_studio/feature/select_radio_graph_screen/cubit/select_image_cubit.dart';
import 'package:domain/domain.dart';
import 'package:domain/model/api_urls.dart';
import 'package:flutter/material.dart';
import 'package:flutter_bloc/flutter_bloc.dart';
import 'package:go_router/go_router.dart';

import '../../navigation/routes.dart';

class MainScreen extends StatefulWidget {
  final Widget child;

  const MainScreen({super.key, required this.child});

  @override
  State<MainScreen> createState() => _MainScreenState();
}

class _MainScreenState extends State<MainScreen> {
  late int currentIndex;
  late String location;

  @override
  void initState() {
    super.initState();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      bottomNavigationBar: Padding(
        padding: EdgeInsets.only(
          bottom: Platform.isAndroid ? Dimens.marginSmall : Dimens.zero,
        ),
        child: Container(
          height: 90,
          decoration: BoxDecoration(
            color: Colors.white,
            boxShadow: [
              BoxShadow(
                color: Colors.black.withValues(alpha: 0.19),
                offset: const Offset(0, 3.75),
                blurRadius: Dimens.medium,
              ),
              BoxShadow(
                color: Colors.black.withValues(alpha: 0.039),
                offset: const Offset(0, 0.5),
                blurRadius: 5.0,
              ),
            ],
          ),
          child: NavigationBar(
            selectedIndex: selectedIndex,
            destinations: _buildItems(context, selectedIndex),
            onDestinationSelected: (index) {
              setState(() {});
              switch (index) {
                case 0:
                  context.go(Routes.homeScreenPath);
                  break;
                case 1:
                  final cubit = context.read<SelectImageCubit>();
                  final jobId =
                      cubit.state.whenOrNull(
                        imageAnalyzedSuccessfully: (analyses, _) =>
                            analyses.jobId,
                      ) ??
                      '';
                  context.go(Routes.chatbotScreenPath, extra: jobId);
                  break;
                case 2:
                  final cubit = context.read<SelectImageCubit>();
                  final emptyAnalyses = Analyses(
                    apiUrls: ApiUrls(
                      caption_text: '',
                      original_image: '',
                      pdf_report: '',
                      predicted_image: '',
                      report_text: '',
                    ),
                    predictions: [],
                    jobId: '',
                  );
                  final analyses =
                      cubit.state.whenOrNull(
                        imageAnalyzedSuccessfully: (analyses, _) => analyses,
                      ) ??
                      emptyAnalyses;
                  context.go(Routes.feedbackScreenPath, extra: analyses);
                  break;
              }
            },
          ),
        ),
      ),
      body: widget.child,
    );
  }

  List<Widget> _buildItems(BuildContext context, int selectedIndex) {
    List<Widget> items = [];

    // Home
    items.add(
      NavigationDestination(
        icon: Icon(Icons.home, color: AppColors.labelColor),
        selectedIcon: Icon(Icons.home, color: AppColors.primarySwatch),
        label: Strings.home,
      ),
    );

    // Chatbot
    items.add(
      NavigationDestination(
        icon: Icon(Icons.chat_rounded, color: AppColors.labelColor),
        selectedIcon: Icon(Icons.chat_rounded, color: AppColors.primarySwatch),
        label: Strings.chatbot,
      ),
    );

    // Feedback
    items.add(
      NavigationDestination(
        icon: Icon(Icons.feedback, color: AppColors.labelColor),
        selectedIcon: Icon(Icons.feedback, color: AppColors.primarySwatch),
        label: Strings.feedback,
      ),
    );

    return items;
  }

  int get selectedIndex {
    final location = GoRouterState.of(context).uri.toString();
    if (location.startsWith(Routes.chatbotScreenPath)) {
      return 1;
    } else if (location.startsWith(Routes.feedbackScreenPath)) {
      return 2;
    }
    return 0;
  }
}
