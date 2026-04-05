import 'package:ai_dental_studio/feature/main_screen/chatbot_screen/ChatbotScreen.dart';
import 'package:ai_dental_studio/feature/main_screen/feedback_screen/feedback_screen.dart';
import 'package:ai_dental_studio/feature/pdf_viewer_screen/pdf_viewer_screen.dart';
import 'package:ai_dental_studio/feature/zoomable_image/zoomable_image_screen.dart';
import 'package:ai_dental_studio/navigation/routes.dart';
import 'package:domain/domain.dart';
import 'package:go_router/go_router.dart';
import 'package:injectable/injectable.dart';

import '../feature/app_screen/app_screen.dart';
import '../feature/main_screen/home_screen/home_creen.dart';
import '../feature/main_screen/main_screen.dart';
import '../feature/select_radio_graph_screen/select_radio_graph_screen.dart';

@singleton
class AppRouter {
  final _router = GoRouter(
    initialLocation: Routes.rootPath,
    routes: [
      GoRoute(
        name: Routes.root,
        path: Routes.rootPath,
        builder: (context, state) => AppScreen(),
      ),
      ShellRoute(
        builder: (context, state, child) {
          return MainScreen(child: child); // stays alive
        },
        routes: [
          GoRoute(
            name: Routes.homeScreen,
            path: Routes.homeScreenPath,
            builder: (context, state) => const HomeScreen(),
            routes: [
              // any routes
            ],
          ),
          GoRoute(
            name: Routes.chatbotScreen,
            path: Routes.chatbotScreenPath,
            builder: (context, state) {
              final jobId = state.extra as String;
              return ChatbotScreen(jobId: jobId);
            },
            routes: [
              // any routes
            ],
          ),
          GoRoute(
            name: Routes.feedbackScreen,
            path: Routes.feedbackScreenPath,
            builder: (context, state) {
              final analyses = state.extra as Analyses;
              return FeedbackScreen(analyses: analyses);
            },
            routes: [
              // any routes
            ],
          ),
        ],
      ),
      GoRoute(
        name: Routes.selectRadioGraphScreen,
        path: Routes.selectRadioGraphScreenPath,
        builder: (context, state) => const SelectRadioGraphScreen(),
        routes: [
          // any routes
        ],
      ),
      GoRoute(
        path: Routes.zoomableImageScreenPath,
        builder: (context, state) {
          final extras = state.extra as Map<String, dynamic>;
          final imageUrl = extras['imageUrl'] as String;
          final jobId = extras['jobId'] as String;
          return ZoomableImageScreen(imageUrl: imageUrl, jobId: jobId);
        },
      ),
      GoRoute(
        path: Routes.pdfViewerScreenPath,
        builder: (context, state) {
          final extras = state.extra as Map<String, dynamic>;
          final pdfUrl = extras['pdfUrl'] as String;
          final jobId = extras['jobId'] as String;
          return PdfViewerScreen(pdfUrl: pdfUrl, jobId: jobId);
        },
      ),
    ],
  );

  GoRouter get router => _router;
}
