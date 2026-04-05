import 'package:flutter/material.dart';
import 'package:go_router/go_router.dart';

import '../../../core/constants/app_colors.dart';
import '../../../core/constants/dimens.dart';
import '../../../core/constants/font_sizes.dart';
import '../../../core/constants/strings.dart';
import '../../../navigation/routes.dart';

class HomeScreen extends StatelessWidget {
  const HomeScreen({super.key});

  @override
  Widget build(BuildContext context) {
    final textTheme = Theme.of(context).textTheme;

    return Scaffold(
      backgroundColor: Colors.white,
      body: SafeArea(
        child: Padding(
          padding: const EdgeInsets.all(Dimens.marginMedium),
          child: Column(
            children: [
              Expanded(
                child: Column(
                  mainAxisAlignment: MainAxisAlignment.center,
                  children: [
                    Text(
                      Strings.aiAnalyses,
                      style: textTheme.headlineSmall?.copyWith(
                        color: AppColors.black,
                        fontSize: FontSizes.xxMedium,
                      ),
                      textAlign: TextAlign.center,
                    ),
                    SizedBox(height: 20),
                    Text(
                      Strings.description,
                      style: textTheme.titleMedium?.copyWith(
                        color: AppColors.black,
                      ),
                      textAlign: TextAlign.center,
                    ),
                    SizedBox(height: 8),

                    Text(
                      Strings.aiIdentify,
                      style: textTheme.bodySmall?.copyWith(
                        color: AppColors.black,
                      ),
                      textAlign: TextAlign.center,
                    ),

                    //
                  ],
                ),
              ),
              Padding(
                padding: EdgeInsets.only(
                  top: Dimens.marginLarge,
                  left: Dimens.marginXxLarge,
                  right: Dimens.marginXxLarge,
                ),
                child: ElevatedButton.icon(
                  icon: Icon(Icons.upload, color: AppColors.white),

                  onPressed: () {
                    context.push(Routes.selectRadioGraphScreenPath);
                  },
                  style: ElevatedButton.styleFrom(
                    minimumSize: const Size.fromHeight(50),
                    backgroundColor: AppColors.primarySwatch,
                  ),
                  label: Text(
                    Strings.upload,
                    style: textTheme.titleLarge?.copyWith(
                      color: AppColors.white,
                    ),
                  ),
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }
}
