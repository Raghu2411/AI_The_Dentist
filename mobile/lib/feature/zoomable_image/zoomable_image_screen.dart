import 'package:ai_dental_studio/core/constants/app_colors.dart';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:go_router/go_router.dart';

import '../../core/constants/dimens.dart';
import '../../core/constants/strings.dart';
import '../../navigation/routes.dart';

class ZoomableImageScreen extends StatelessWidget {
  final String imageUrl;
  final String jobId;

  const ZoomableImageScreen({
    super.key,
    required this.imageUrl,
    required this.jobId,
  });

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        automaticallyImplyLeading: false,
        systemOverlayStyle: const SystemUiOverlayStyle(
          statusBarColor: Colors.black,
          statusBarIconBrightness: Brightness.light,
          statusBarBrightness: Brightness.dark,
        ),
        leading: Padding(
          padding: const EdgeInsets.only(left: Dimens.xLarge),
          child: Row(
            children: [
              InkWell(
                onTap: () {
                  context.pop();
                },
                child: Icon(Icons.arrow_back, color: AppColors.white),
              ),
            ],
          ),
        ),
        actions: [
          IconButton(
            icon: Icon(Icons.chat, color: Colors.white),
            tooltip: Strings.chat,
            onPressed: () {
              if (context.canPop()) context.pop();
              if (context.canPop()) context.pop();
              context.go(Routes.chatbotScreenPath, extra: jobId);
            },
          ),
          const SizedBox(width: Dimens.marginSmall),
        ],
      ),
      backgroundColor: Colors.black,
      body: Center(
        child: InteractiveViewer(
          minScale: 0.8,
          maxScale: 6.0,
          panEnabled: true,
          scaleEnabled: true,
          boundaryMargin: const EdgeInsets.all(Dimens.zero),
          child: loadNetWorkImage(imageUrl),
        ),
      ),
    );
  }

  Widget loadNetWorkImage(String image) {
    return Padding(
      padding: const EdgeInsets.only(top: Dimens.medium, bottom: Dimens.medium),
      child: Image.network(
        image,
        fit: BoxFit.contain,
        loadingBuilder: (context, child, loadingProgress) {
          if (loadingProgress == null) return child;
          return const Center(child: CircularProgressIndicator());
        },
        errorBuilder: (context, error, stackTrace) =>
            const Center(child: Icon(Icons.broken_image, size: 50)),

        width: double.infinity,
        height: double.infinity,
      ),
    );
  }
}
