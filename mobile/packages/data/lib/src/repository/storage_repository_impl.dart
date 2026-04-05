import 'package:domain/domain.dart';
import 'package:injectable/injectable.dart';

@Singleton(as: StorageRepository)
class StorageRepositoryImpl implements StorageRepository {
  static const keySession = 'session';

  @override
  Future<void> example() async {
    /// do what is required
  }
}
