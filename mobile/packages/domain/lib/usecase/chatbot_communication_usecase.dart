import 'package:domain/domain.dart';
import 'package:domain/model/chatbot_messages.dart';
import 'package:domain/usecase/usecase.dart';
import 'package:injectable/injectable.dart';

@injectable
class ChatbotCommunicationUseCase extends UseCase {
  final EssexDentalApiRepository _repository;

  ChatbotCommunicationUseCase(this._repository);

  Future<Result<ChatbotMessage>> call(
    String query,
    String jobId,
    List<Map<String, String>> history,
  ) async {
    return _repository.chat(query, jobId, history);
  }
}
