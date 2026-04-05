import 'dart:io';

import 'package:data/src/common/data_object_mapper.dart';
import 'package:domain/domain.dart';
import 'package:domain/model/chatbot_messages.dart';
import 'package:domain/model/feedback.dart';
import 'package:injectable/injectable.dart';
import 'package:logger/logger.dart';

import '../datasource/api/essex_dental_api.dart';

@Singleton(as: EssexDentalApiRepository)
@injectable
class EssexDentalApiRepositoryImpl extends EssexDentalApiRepository {
  final EssexDentalApi _essexDentalApi;
  final DataObjectMapper _objectMapper;
  final Logger _logger;

  EssexDentalApiRepositoryImpl({
    required EssexDentalApi essexDentalApi,
    required DataObjectMapper objectMapper,
    required Logger logger,
  }) : _objectMapper = objectMapper,
       _essexDentalApi = essexDentalApi,
       _logger = logger;

  @override
  Future<Result<Analyses>> checkPrediction(
    String OPGImagePath,
    List<String> selectedModels,
  ) async {
    try {
      final response = await _essexDentalApi.checkOPGAnalyses(
        OPGImagePath,
        selectedModels,
      );
      final analyses = _objectMapper.toAnalyses(response);
      return Result.success(analyses);
    } on Exception catch (e) {
      _logger.e('Exception e: $e');
      return Result.failed(_objectMapper.toError(e));
    }
  }

  @override
  Future<Result> downloadFileFromUrl(
    String fileUrl,
    Directory appDocDir, {
    String? filePath,
    String? fileName,
  }) async {
    try {
      final filePath = await _essexDentalApi.downloadFileFromUrl(
        fileUrl,
        appDocDir,
      );
      return Result.success(filePath);
    } on Exception catch (e) {
      _logger.e('Exception e: $e');
      return Result.failed(_objectMapper.toError(e));
    }
  }

  @override
  Future<Result<ChatbotMessage>> chat(
    String query,
    String jobId,
    List<Map<String, String>> history,
  ) async {
    try {
      final response = await _essexDentalApi.chat(query, jobId, history);
      final analyses = _objectMapper.toChatbotMessages(response);

      return Result.success(analyses);
    } on Exception catch (e) {
      _logger.e('Exception e: $e');
      return Result.failed(_objectMapper.toError(e));
    }
  }

  @override
  Future<Result<Feedback>> submitFeedback(
    Analyses analyses,
    String name,
    String feedbackText,
  ) async {
    try {
      final response = await _essexDentalApi.submitFeedback(
        analyses,
        name,
        feedbackText,
      );
      final feedback = _objectMapper.toFeedback(response);

      return Result.success(feedback);
    } on Exception catch (e) {
      _logger.e('Exception e: $e');
      return Result.failed(_objectMapper.toError(e));
    }
  }
}
