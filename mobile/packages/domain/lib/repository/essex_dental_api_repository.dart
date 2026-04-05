import 'dart:io';

import 'package:domain/domain.dart';
import 'package:domain/model/chatbot_messages.dart';
import 'package:domain/model/feedback.dart';
import 'package:domain/repository/repository.dart';

///
/// Repository that contains all possible API actions
///
abstract class EssexDentalApiRepository extends Repository {
  ///
  /// checkPrediction to get the the OPG analyses results
  ///
  Future<Result<Analyses>> checkPrediction(
    String OPGImagePath,
    List<String> selectedModels,
  );

  ///
  /// download file from url
  ///
  Future<Result> downloadFileFromUrl(String fileUrl, Directory appDocDir);

  ///
  /// Ask query via chatbot
  ///
  Future<Result<ChatbotMessage>> chat(
    String query,
    String jobId,
    List<Map<String, String>> history,
  );

  ///
  /// Submit feedback
  ///
  Future<Result<Feedback>> submitFeedback(
    Analyses analyses,
    String name,
    String feedbackText,
  );
}
