import 'dart:io';
import 'dart:typed_data';

import 'package:ai_dental_studio/navigation/routes.dart';
import 'package:domain/model/prediction.dart';
import 'package:flutter/material.dart';
import 'package:go_router/go_router.dart';

class CustomGridView extends StatelessWidget {
  final List<String> selectedImages;
  final String? originalImage;
  final String? predictedImage;
  final List<Prediction> predictions;
  final String? jobId;

  const CustomGridView({
    super.key,
    required this.selectedImages,
    required this.jobId,
    this.originalImage,
    this.predictedImage,
    this.predictions = const [],
  });

  @override
  Widget build(BuildContext context) {
    return Column(
      mainAxisAlignment: MainAxisAlignment.center,
      crossAxisAlignment: CrossAxisAlignment.center,
      children: [
        predictedImage != null
            ? Column(
                children: [
                  InkWell(
                    onTap: () {
                      context.push(
                        Routes.zoomableImageScreenPath,
                        extra: {'imageUrl': originalImage, 'jobId': jobId},
                      );
                    },
                    child: loadNetWorkImage(originalImage!),
                  ),
                  InkWell(
                    onTap: () {
                      context.push(
                        Routes.zoomableImageScreenPath,
                        extra: {'imageUrl': predictedImage, 'jobId': jobId},
                      );
                    },
                    child: loadNetWorkImage(predictedImage!),
                  ),
                ],
              )
            : Padding(
                padding: const EdgeInsets.only(top: 10, bottom: 10),
                child: Image.file(
                  File(selectedImages[0]),
                  fit: BoxFit.cover,
                  errorBuilder: (context, error, stackTrace) => const Center(
                    child: Icon(Icons.image_not_supported, size: 50),
                  ),
                ),
              ),
      ],
    );
  }

  Widget loadBase64Image(Uint8List image) {
    return Padding(
      padding: const EdgeInsets.only(top: 10, bottom: 10),
      child: Image.memory(
        image,
        fit: BoxFit.cover,
        errorBuilder: (context, error, stackTrace) =>
            const Center(child: Icon(Icons.image_not_supported, size: 50)),
      ),
    );
  }

  Widget loadNetWorkImage(String image) {
    return Padding(
      padding: const EdgeInsets.only(top: 10, bottom: 10),
      child: Image.network(
        image,
        fit: BoxFit.cover,
        loadingBuilder: (context, child, loadingProgress) {
          if (loadingProgress == null) return child;
          return const Center(child: CircularProgressIndicator());
        },
        errorBuilder: (context, error, stackTrace) =>
            const Center(child: Icon(Icons.broken_image, size: 50)),
      ),
    );
  }
}
