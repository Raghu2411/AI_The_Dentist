import 'dart:io';

import 'package:data/src/dto/feedback_dto.dart';
import 'package:dio/dio.dart';
import 'package:domain/domain.dart';
import 'package:http_parser/http_parser.dart';
import 'package:injectable/injectable.dart';

import '../../dto/analyses_dto.dart';
import '../../dto/chatbot_dto.dart';

@singleton
class EssexDentalApi {
  static const String kRouteApiExample = 'some-api-end-point';

  static const String kRoutePredict = 'analyze';
  static const String kRouteChat = 'api/v1/chat';
  static const String kFeedback = 'feedback';

  final Dio dio;

  const EssexDentalApi(this.dio);

  ///
  /// Get analyses
  ///
  Future<AnalysesDto> checkOPGAnalyses(
    String OPGImagePath,
    List<String> selectedModels,
  ) async {
    MultipartFile multipartFile = await MultipartFile.fromFile(
      OPGImagePath,
      filename: OPGImagePath.split("/").last,
      contentType: MediaType('image', 'jpeg'),
    );

    /// ----------------------------Local Server--------------------------------
    // final data = FormData.fromMap({
    //   'image': multipartFile,
    //   // 'model_name': 'YOLOv9',
    //   'model_name': selectedModel,
    //   'confidence': '0.6',
    // });
    /// ------------------------------------------------------------------------

    final models = selectedModels.map((item) => item.toLowerCase()).toList();
    final data = FormData.fromMap({'image': multipartFile, 'models': models});
    final response = await dio.post(kRoutePredict, data: data);
    return AnalysesDto.fromJson(response.data);
  }

  ///
  ///  Download file from url
  ///
  Future<String> downloadFileFromUrl(
    String fileUrl,
    Directory appDocDir, {
    Options? options,
  }) async {
    final downloadFileName = DateTime.now().toIso8601String();
    final fullPath = "${appDocDir.path}/$downloadFileName.pdf";
    await dio.download(fileUrl, fullPath, options: options);
    return fullPath;
  }

  ///
  /// Chatbot messages
  ///
  Future<ChatbotDto> chat(
    String query,
    String jobId,
    List<Map<String, String>> history,
  ) async {
    final data = {'job_id': jobId, 'query': query, 'history': history};
    final response = await dio.post(kRouteChat, data: data);
    return ChatbotDto.fromJson(response.data);
  }

  ///
  /// Submit feedback
  ///
  Future<FeedbackDto> submitFeedback(
    Analyses analyses,
    String name,
    String feedbackText,
  ) async {
    final data = {
      'source': 'mobile_api',
      'name': name,
      'feedback_text': feedbackText,
      'job_id': analyses.jobId,
      'medical_report': analyses.apiUrls.report_text,
      'pdf_url': analyses.apiUrls.pdf_report,
      'original_image_url': analyses.apiUrls.original_image,
      'predicted_image_url': analyses.apiUrls.predicted_image,
    };
    final response = await dio.post(kFeedback, data: data);
    return FeedbackDto.fromJson(response.data);
  }
}
